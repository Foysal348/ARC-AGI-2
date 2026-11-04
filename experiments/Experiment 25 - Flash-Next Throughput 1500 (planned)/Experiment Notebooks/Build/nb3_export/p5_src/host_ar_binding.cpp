// host_ar_binding.cpp - torch ops for the P5 host-staged all-reduce (library "p5_host_ar", loaded with
// torch.ops.load_library). Shared host buffer: a POSIX shm file (/dev/shm/<name>) created by rank 0 (ftruncate +
// posix_fallocate, so a full /dev/shm fails here with ENOSPC instead of SIGBUS later), mapped MAP_SHARED by every rank
// and registered in each process with cudaHostRegister(Portable | Mapped) - the same recipe as NCCL's legacy SHM path
// (src/os/linux.cc ncclOsShmOpen). The kernel gets the mapping's device pointer from cudaHostGetDevicePointer.
//
// Ops:
//   info() -> str
//   init(rank, world, name, max_bytes, device, timeout_ms) -> handle      (collective: every rank must call it)
//   all_reduce(handle, inp, out, mode, blocks) -> ()  stream ordered on the current stream, graph capturable
//   status(handle) -> int[]   [abort, status_0 .. status_{world-1}, max_coresident_blocks]; abort is sticky: every
//                             later call on the handle returns all-NaN until destroy + init (collective)
//   epoch(handle) -> int      completed calls on this rank (synchronous device read; not inside capture)
//   destroy(handle) -> ()     synchronizes the device first; drop every CUDA graph that captured this handle BEFORE
//                             destroy (a replay afterwards touches unmapped host memory / freed seq = illegal address)
//   cover_check(n, world, dtype, mode, blocks) -> int[]   CPU model of the partition (host_ar_check.h)
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <torch/library.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "host_ar_check.h"
#include "host_ar_layout.h"

// host_ar.cu
cudaError_t p5_launch(const void* in, void* out, char* dbase, int64_t slot_bytes, unsigned long long* seq,
                      unsigned int* arrive, int64_t n, int world, int rank, int dtype, int mode, int blocks,
                      uint64_t timeout_ns, uint64_t sig_value, cudaStream_t stream);
int p5_max_coresident_blocks(int world, int device);

namespace {

using namespace p5;

struct Handle {
  int rank = 0, world = 0, device = 0;
  int64_t max_bytes = 0, slot_bytes = 0, total = 0;
  uint64_t timeout_ns = 0;
  int max_blocks = 0;
  std::string name;
  int fd = -1;
  char* host = nullptr;
  char* dev = nullptr;
  bool registered = false;
  unsigned long long* seq = nullptr;  // device: [0] = seq, [1] = arrive (as unsigned int)
  bool alive = false;
};

std::mutex g_mu;
std::vector<std::unique_ptr<Handle>> g_handles;

#define P5_CUDA_CHECK(expr, what)                                                                     \
  do {                                                                                                \
    cudaError_t _e = (expr);                                                                          \
    TORCH_CHECK(_e == cudaSuccess, "p5_host_ar: ", what, " failed: ", cudaGetErrorString(_e), " (", \
                (int)_e, ")");                                                                        \
  } while (0)

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void release(Handle& h) {
  if (h.registered) {
    cudaHostUnregister(h.host);
    h.registered = false;
  }
  if (h.host) {
    munmap(h.host, (size_t)h.total);
    h.host = nullptr;
  }
  if (h.fd >= 0) {
    close(h.fd);
    h.fd = -1;
  }
  if (h.seq) {
    cudaFree(h.seq);
    h.seq = nullptr;
  }
  h.dev = nullptr;
  h.alive = false;
}

Handle& get(int64_t handle) {
  std::lock_guard<std::mutex> lk(g_mu);
  TORCH_CHECK(handle >= 0 && handle < (int64_t)g_handles.size() && g_handles[handle] && g_handles[handle]->alive,
              "p5_host_ar: invalid or destroyed handle ", handle);
  return *g_handles[handle];
}

std::string info() {
  return std::string("p5_host_ar v") + std::to_string(kVersion) + " two-shot host-staged all-reduce; worlds 2/4/8; " +
         "dtypes f32/bf16/f16; modes 0=native(fp32 accumulate) 1=int8(absmax/32 groups); threads " +
         std::to_string(kThreads) + "; max blocks " + std::to_string(kMaxBlocks) + "; built for sm_89";
}

int64_t init(int64_t rank, int64_t world, std::string name, int64_t max_bytes, int64_t device, int64_t timeout_ms) {
  TORCH_CHECK(world == 2 || world == 4 || world == 8, "p5_host_ar: world must be 2, 4 or 8, got ", world);
  TORCH_CHECK(rank >= 0 && rank < world, "p5_host_ar: bad rank ", rank);
  TORCH_CHECK(max_bytes > 0 && max_bytes <= (int64_t)1 << 30, "p5_host_ar: max_bytes must be in (0, 1 GiB]");
  TORCH_CHECK(timeout_ms >= 1 && timeout_ms <= 600000, "p5_host_ar: timeout_ms must be in [1, 600000]");
  TORCH_CHECK(name.size() >= 2 && name.size() < 200 && name[0] == '/' && name.find('/', 1) == std::string::npos,
              "p5_host_ar: shm name must look like /p5har_<unique> (one leading slash), got '", name, "'");
  int ndev = 0;
  cudaError_t ce = cudaGetDeviceCount(&ndev);
  TORCH_CHECK(ce == cudaSuccess && ndev > 0, "p5_host_ar: no CUDA device (", cudaGetErrorString(ce), ")");
  if (device < 0) {
    int cur = 0;
    P5_CUDA_CHECK(cudaGetDevice(&cur), "cudaGetDevice");
    device = cur;
  }
  TORCH_CHECK(device < ndev, "p5_host_ar: device ", device, " >= device count ", ndev);
  c10::cuda::CUDAGuard guard((c10::DeviceIndex)device);
  int can_map = 0, can_register = 0;
  P5_CUDA_CHECK(cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory, (int)device), "attr CanMapHostMemory");
  P5_CUDA_CHECK(cudaDeviceGetAttribute(&can_register, cudaDevAttrHostRegisterSupported, (int)device),
                "attr HostRegisterSupported");
  TORCH_CHECK(can_map && can_register, "p5_host_ar: device cannot map/register host memory (map=", can_map,
              ", register=", can_register, ")");

  auto h = std::make_unique<Handle>();
  h->rank = (int)rank;
  h->world = (int)world;
  h->device = (int)device;
  h->max_bytes = max_bytes;
  h->slot_bytes = slot_bytes_for(max_bytes);
  h->total = total_bytes_for((int)world, max_bytes);
  h->timeout_ns = (uint64_t)timeout_ms * 1000000ull;
  h->name = name;
  const double deadline = now_s() + (double)timeout_ms / 1000.0 + 30.0;

  try {
    if (rank == 0) {
      shm_unlink(name.c_str());  // a stale file of the same name (crashed run) must not be reused
      h->fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
      TORCH_CHECK(h->fd >= 0, "p5_host_ar: shm_open(create ", name, ") failed: ", strerror(errno));
      TORCH_CHECK(ftruncate(h->fd, (off_t)h->total) == 0, "p5_host_ar: ftruncate failed: ", strerror(errno));
      int fe = posix_fallocate(h->fd, 0, (off_t)h->total);
      TORCH_CHECK(fe == 0, "p5_host_ar: posix_fallocate(", h->total, " B) on /dev/shm failed: ", strerror(fe),
                  " (is /dev/shm large enough?)");
      void* m = mmap(nullptr, (size_t)h->total, PROT_READ | PROT_WRITE, MAP_SHARED, h->fd, 0);
      TORCH_CHECK(m != MAP_FAILED, "p5_host_ar: mmap failed: ", strerror(errno));
      h->host = (char*)m;
      std::memset(h->host, 0, (size_t)kDataOff);  // header, flags, signatures (the data area is zero-filled anyway)
      ShmHeader* hd = reinterpret_cast<ShmHeader*>(h->host + kHeaderOff);
      hd->world = (int32_t)world;
      hd->version = kVersion;
      hd->max_bytes = max_bytes;
      hd->slot_bytes = h->slot_bytes;
      hd->total_bytes = h->total;
      __atomic_store_n(&hd->magic, kMagic, __ATOMIC_RELEASE);
    } else {
      for (;;) {
        TORCH_CHECK(now_s() < deadline, "p5_host_ar: rank ", rank, " timed out waiting for shm ", name);
        int fd = shm_open(name.c_str(), O_RDWR, 0600);
        if (fd < 0) {
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
          continue;
        }
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size < h->total) {
          close(fd);
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
          continue;
        }
        h->fd = fd;
        break;
      }
      void* m = mmap(nullptr, (size_t)h->total, PROT_READ | PROT_WRITE, MAP_SHARED, h->fd, 0);
      TORCH_CHECK(m != MAP_FAILED, "p5_host_ar: mmap failed: ", strerror(errno));
      h->host = (char*)m;
      ShmHeader* hd = reinterpret_cast<ShmHeader*>(h->host + kHeaderOff);
      while (__atomic_load_n(&hd->magic, __ATOMIC_ACQUIRE) != kMagic) {
        TORCH_CHECK(now_s() < deadline, "p5_host_ar: rank ", rank, " timed out waiting for the shm header");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      TORCH_CHECK(hd->world == world && hd->max_bytes == max_bytes && hd->total_bytes == h->total &&
                      hd->version == kVersion,
                  "p5_host_ar: shm header mismatch (world/max_bytes/version differ between ranks)");
    }
    P5_CUDA_CHECK(cudaHostRegister(h->host, (size_t)h->total, cudaHostRegisterPortable | cudaHostRegisterMapped),
                  "cudaHostRegister(Portable|Mapped)");
    h->registered = true;
    void* d = nullptr;
    P5_CUDA_CHECK(cudaHostGetDevicePointer(&d, h->host, 0), "cudaHostGetDevicePointer");
    h->dev = (char*)d;
    P5_CUDA_CHECK(cudaMalloc((void**)&h->seq, 256), "cudaMalloc(seq)");
    P5_CUDA_CHECK(cudaMemset(h->seq, 0, 256), "cudaMemset(seq)");
    P5_CUDA_CHECK(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    h->max_blocks = p5_max_coresident_blocks((int)world, (int)device);
    TORCH_CHECK(h->max_blocks >= 1, "p5_host_ar: occupancy query failed");
    if (h->max_blocks > kMaxBlocks) h->max_blocks = kMaxBlocks;

    ShmHeader* hd = reinterpret_cast<ShmHeader*>(h->host + kHeaderOff);
    __atomic_fetch_add(&hd->attached, 1, __ATOMIC_SEQ_CST);
    while (__atomic_load_n(&hd->attached, __ATOMIC_SEQ_CST) < (int32_t)world) {
      TORCH_CHECK(now_s() < deadline, "p5_host_ar: rank ", rank, " timed out waiting for all ranks to attach (",
                  __atomic_load_n(&hd->attached, __ATOMIC_SEQ_CST), "/", world, ")");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    TORCH_CHECK(__atomic_load_n(&hd->attached, __ATOMIC_SEQ_CST) == (int32_t)world,
                "p5_host_ar: more ranks attached than world (stale or shared shm name?)");
    if (rank == 0) shm_unlink(name.c_str());  // every rank has it mapped; the name is no longer needed
  } catch (...) {
    if (rank == 0) shm_unlink(name.c_str());
    release(*h);
    throw;
  }
  h->alive = true;
  std::lock_guard<std::mutex> lk(g_mu);
  g_handles.push_back(std::move(h));
  return (int64_t)g_handles.size() - 1;
}

int dtype_code(const at::Tensor& t) {
  switch (t.scalar_type()) {
    case at::kFloat: return F32;
    case at::kBFloat16: return BF16;
    case at::kHalf: return F16;
    default: TORCH_CHECK(false, "p5_host_ar: dtype must be float32, bfloat16 or float16, got ", t.scalar_type());
  }
  return -1;
}

void all_reduce(int64_t handle, const at::Tensor& inp, const at::Tensor& out, int64_t mode, int64_t blocks) {
  Handle& h = get(handle);
  TORCH_CHECK(inp.is_cuda() && out.is_cuda(), "p5_host_ar: tensors must be CUDA tensors");
  TORCH_CHECK(inp.get_device() == h.device && out.get_device() == h.device,
              "p5_host_ar: tensors must live on the handle's device ", h.device);
  TORCH_CHECK(inp.is_contiguous() && out.is_contiguous(), "p5_host_ar: tensors must be contiguous");
  TORCH_CHECK(inp.scalar_type() == out.scalar_type(), "p5_host_ar: inp/out dtype differ");
  TORCH_CHECK(inp.numel() == out.numel(), "p5_host_ar: inp/out numel differ");
  TORCH_CHECK(mode == NATIVE || mode == INT8, "p5_host_ar: mode must be 0 (native) or 1 (int8)");
  const int dt = dtype_code(inp);
  const int64_t n = inp.numel();
  if (n == 0) return;  // no launch: the epoch stays in step on every rank (all ranks pass the same n)
  const int64_t nbytes = n * dtype_size(dt);
  TORCH_CHECK(nbytes <= h.max_bytes, "p5_host_ar: ", nbytes, " B > max_bytes ", h.max_bytes);
  TORCH_CHECK(slot_need(n, dt, (int)mode) <= h.slot_bytes, "p5_host_ar: staging slot too small");
  const uintptr_t pi = reinterpret_cast<uintptr_t>(inp.data_ptr());
  const uintptr_t po = reinterpret_cast<uintptr_t>(out.data_ptr());
  TORCH_CHECK(pi % 16 == 0 && po % 16 == 0, "p5_host_ar: data pointers must be 16-byte aligned");
  TORCH_CHECK(pi == po || pi + (uintptr_t)nbytes <= po || po + (uintptr_t)nbytes <= pi,
              "p5_host_ar: inp and out must be the same tensor or not overlap");
  int nb = blocks <= 0 ? auto_blocks(n, h.world) : (int)blocks;
  if (nb > h.max_blocks) {
    TORCH_CHECK(blocks <= 0, "p5_host_ar: blocks ", blocks, " > co-resident limit ", h.max_blocks);
    nb = h.max_blocks;
  }
  TORCH_CHECK(nb >= 1 && nb <= kMaxBlocks, "p5_host_ar: bad block count ", nb);
  c10::cuda::CUDAGuard guard((c10::DeviceIndex)h.device);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream((c10::DeviceIndex)h.device).stream();
  const uint64_t sig = call_sig(n, dt, (int)mode, nb, h.world);
  cudaError_t e = p5_launch(inp.data_ptr(), out.data_ptr(), h.dev, h.slot_bytes, h.seq,
                            reinterpret_cast<unsigned int*>(h.seq + 1), n, h.world, h.rank, dt, (int)mode, nb,
                            h.timeout_ns, sig, stream);
  TORCH_CHECK(e == cudaSuccess, "p5_host_ar: kernel launch failed: ", cudaGetErrorString(e));
}

std::vector<int64_t> status(int64_t handle) {
  Handle& h = get(handle);
  const volatile ShmHeader* hd = reinterpret_cast<const volatile ShmHeader*>(h.host + kHeaderOff);
  std::vector<int64_t> v;
  v.push_back((int64_t)hd->abort_flag);
  for (int r = 0; r < h.world; ++r) v.push_back((int64_t)hd->status[r]);
  v.push_back((int64_t)h.max_blocks);
  return v;
}

int64_t epoch(int64_t handle) {
  Handle& h = get(handle);
  c10::cuda::CUDAGuard guard((c10::DeviceIndex)h.device);
  unsigned long long v = 0;
  P5_CUDA_CHECK(cudaMemcpy(&v, h.seq, sizeof(v), cudaMemcpyDeviceToHost), "cudaMemcpy(seq)");
  return (int64_t)v;
}

void destroy(int64_t handle) {
  Handle& h = get(handle);
  c10::cuda::CUDAGuard guard((c10::DeviceIndex)h.device);
  // In-flight kernels (any stream) may still read/write the mapping or seq: finish them before unregister/munmap/free.
  // If the context is already dead (sticky error) nothing runs any more: release anyway, then report the error.
  const cudaError_t se = cudaDeviceSynchronize();
  release(h);
  TORCH_CHECK(se == cudaSuccess, "p5_host_ar: cudaDeviceSynchronize before destroy failed: ", cudaGetErrorString(se));
}

std::vector<int64_t> cover_check_op(int64_t n, int64_t world, int64_t dtype, int64_t mode, int64_t blocks) {
  return p5::cover_check(n, (int)world, (int)dtype, (int)mode, (int)blocks);
}

}  // namespace

TORCH_LIBRARY(p5_host_ar, m) {
  m.def("info() -> str", &info);
  m.def("init(int rank, int world, str name, int max_bytes, int device, int timeout_ms) -> int", &init);
  m.def("all_reduce(int handle, Tensor inp, Tensor(a!) out, int mode, int blocks) -> ()");
  m.def("status(int handle) -> int[]", &status);
  m.def("epoch(int handle) -> int", &epoch);
  m.def("destroy(int handle) -> ()", &destroy);
  m.def("cover_check(int n, int world, int dtype, int mode, int blocks) -> int[]", &cover_check_op);
}

// Tensor op: a CUDA-key kernel (like vLLM's custom ops), not a catch-all, so the dispatcher treats the mutable `out`
// as a normal in-place CUDA op. The schema already carries the alias annotation `Tensor(a!) out`.
TORCH_LIBRARY_IMPL(p5_host_ar, CUDA, m) { m.impl("all_reduce", &all_reduce); }

// Meta (fake-tensor) kernel for torch.compile / FakeTensor tracing: the op only mutates `out` in place and returns
// nothing, so shape propagation is a no-op. (vLLM integration notes: one handle per TP group per stream; the op is
// collective, so tracing must not reorder it relative to other collectives.)
namespace {
void all_reduce_meta(int64_t, const at::Tensor& inp, const at::Tensor& out, int64_t, int64_t) {
  TORCH_CHECK(inp.numel() == out.numel() && inp.scalar_type() == out.scalar_type(),
              "p5_host_ar: inp/out numel or dtype differ");
}
}  // namespace
TORCH_LIBRARY_IMPL(p5_host_ar, Meta, m) { m.impl("all_reduce", &all_reduce_meta); }
