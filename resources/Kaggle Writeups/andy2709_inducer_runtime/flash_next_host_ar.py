"""Host-memory all-reduce for Flash-Next's small TP4 decode reductions (SGLANG_FLASH_NEXT_HOST_AR=1).

Kaggle's 4xL4 G2 VMs have no usable peer path (0.46-0.9 GB/s per GPU when all four talk), so NCCL runs
its SHM transport, whose protocol costs dominate at decode sizes. Here every rank's kernel exchanges
its bf16 partial through one pinned /dev/shm buffer that all four scheduler processes map and register,
synchronized by system-scope flags (Kaggle probe kaggle_l4_hostmem_20261021_v1, us per op in a graph
chain against the shipped NCCL: 1x2560 9.5 vs 19.3, 8x2560 28.5 vs 35.6, 32x2560 74.5 vs 101.5).

- One-shot (at most ONE_SHOT_BYTES per rank): every rank writes its partial, then sums all four.
- Two-shot (at most MAX_BYTES): rank q owns quarter q; it sums that quarter and shares the result.
- Larger, non-bf16 or misaligned tensors stay on NCCL (it wins at prefill sizes: 2.5 MiB 831 vs 1005).

Every output element is the fp32 sum in rank order 0..3, rounded once to bf16, so all ranks hold the
same bits. Each communicator keeps an epoch per GPU: every CTA reads it at kernel start and the last
CTA to finish advances it, so it persists across CUDA-graph replays. Calls alternate between two halves
of the buffer by epoch parity; a rank starts call e + 2 only after every peer has entered call e + 1,
so no peer can still be reading the half it overwrites, whatever the shapes in between. Flags hold the
epoch at which a writer's CTA finished a phase. A wait longer than TIMEOUT_CYCLES (about nine minutes)
traps, so a lost peer crashes the server instead of corrupting its outputs.

Kernels compile with the runtime's NVRTC (no nvcc) and launch through the driver API on torch's
current stream. Setup steps are agreed across ranks: if any rank fails, all of them stay on NCCL.
"""
import ctypes
import glob
import logging
import mmap
import os
import sys
import uuid

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

WORLD = 4
ONE_SHOT_BYTES = 10 * 1024  # Two 2560-wide rows: one-shot measured 9.5 us at one row, two-shot 15.1.
MAX_BYTES = 320 * 1024  # 64 rows: two-shot 132.7 us against NCCL's 150.4 (NCCL wins at 512 rows).
THREADS = 256
MAX_CTAS = 48  # The measured grids: ceil(work / THREADS) CTAs, at most 48.
SLOTS = 64  # Flag slots per (phase, writer); at least MAX_CTAS.
TIMEOUT_CYCLES = 1 << 40
PAGE = 4096
CENSUS_LIMIT = 64

SOURCE = r"""
typedef unsigned int u32;
#define WORLD 4
#define SLOTS 64

struct Comm {
  char* data;          // two halves of `half` bytes, selected by epoch parity
  u32* flags;          // [phase 2][writer 4][SLOTS]
  u32* state;          // this rank's device memory: epoch, CTAs finished in the current call
  long long half;
  long long timeout;
  int rank;
  int pad;
};

__device__ __forceinline__ u32 ld_acquire(const u32* p) {
  u32 v; asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void st_release(u32* p, u32 v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ u32 ld_volatile(const u32* p) {
  u32 v; asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ uint4 ld_host(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}

// bf16 <-> fp32 by bit manipulation, rounding to nearest even like torch (NaN -> 0x7fc0).
__device__ __forceinline__ void set8(float* a, uint4 v) {
  const u32 w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; i++) { a[2 * i] = __uint_as_float(w[i] << 16); a[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u); }
}
__device__ __forceinline__ void add8(float* a, uint4 v) {
  const u32 w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; i++) { a[2 * i] += __uint_as_float(w[i] << 16); a[2 * i + 1] += __uint_as_float(w[i] & 0xffff0000u); }
}
__device__ __forceinline__ u32 rne(float f) {
  const u32 u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;
  return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}
__device__ __forceinline__ uint4 pack8(const float* a) {
  uint4 r;
  r.x = rne(a[0]) | (rne(a[1]) << 16); r.y = rne(a[2]) | (rne(a[3]) << 16);
  r.z = rne(a[4]) | (rne(a[5]) << 16); r.w = rne(a[6]) | (rne(a[7]) << 16);
  return r;
}

__device__ __forceinline__ void chunk(long long n, long long& lo, long long& hi) {
  const long long per = (n + gridDim.x - 1) / gridDim.x;
  lo = blockIdx.x * per;
  hi = lo + per < n ? lo + per : n;
}
__device__ __forceinline__ u32* flag(const Comm& cm, int phase, int writer) {
  return cm.flags + (phase * WORLD + writer) * SLOTS + blockIdx.x;
}

// After every thread has written and fenced its data: publish this CTA's phase; wait for the peers'.
__device__ __forceinline__ void exchange(const Comm& cm, int phase, u32 e) {
  const int t = threadIdx.x;
  __threadfence_system();
  __syncthreads();
  if (t == 0) st_release(flag(cm, phase, cm.rank), e);
  if (t < WORLD && t != cm.rank) {
    const u32* f = flag(cm, phase, t);
    if ((int)(ld_acquire(f) - e) < 0) {
      const long long t0 = clock64();
      while ((int)(ld_acquire(f) - e) < 0) {
        if (clock64() - t0 > cm.timeout) asm volatile("trap;");
      }
    }
  }
  __syncthreads();
}
// The last CTA to finish advances the epoch; every CTA read it before counting itself finished.
__device__ __forceinline__ void finish(const Comm& cm, u32 e) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(cm.state + 1, 1u) == gridDim.x - 1) {
      cm.state[1] = 0;
      cm.state[0] = e;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) one_shot(const uint4* __restrict__ in, uint4* __restrict__ out,
                                                           long long n, Comm cm) {
  __shared__ u32 s_epoch;
  const int t = threadIdx.x, r = cm.rank;
  if (t == 0) s_epoch = ld_volatile(cm.state) + 1;
  __syncthreads();
  const u32 e = s_epoch;
  char* half = cm.data + (e & 1) * cm.half;
  long long lo, hi; chunk(n, lo, hi);
  uint4* mine = (uint4*)half + r * n;
  for (long long i = lo + t; i < hi; i += blockDim.x) mine[i] = in[i];
  exchange(cm, 0, e);
  for (long long i = lo + t; i < hi; i += blockDim.x) {
    uint4 v[WORLD];
#pragma unroll
    for (int p = 0; p < WORLD; p++) v[p] = p == r ? in[i] : ld_host((const uint4*)half + p * n + i);
    float a[8]; set8(a, v[0]); add8(a, v[1]); add8(a, v[2]); add8(a, v[3]);
    out[i] = pack8(a);
  }
  finish(cm, e);
}

// Quarter q of writer w's partial goes to scatter slot w * WORLD + q; owner q's sum to result slot q.
extern "C" __global__ void __launch_bounds__(256) two_shot(const uint4* __restrict__ in, uint4* __restrict__ out,
                                                           long long n, Comm cm) {
  __shared__ u32 s_epoch;
  const int t = threadIdx.x, r = cm.rank;
  if (t == 0) s_epoch = ld_volatile(cm.state) + 1;
  __syncthreads();
  const u32 e = s_epoch;
  const long long qn = n / WORLD;
  uint4* half = (uint4*)(cm.data + (e & 1) * cm.half);
  uint4* result = half + WORLD * WORLD * qn;
  long long lo, hi; chunk(qn, lo, hi);
  for (int q = 0; q < WORLD; q++) {
    if (q == r) continue;
    uint4* d = half + (r * WORLD + q) * qn;
    const uint4* x = in + q * qn;
    for (long long i = lo + t; i < hi; i += blockDim.x) d[i] = x[i];
  }
  exchange(cm, 0, e);
  for (long long i = lo + t; i < hi; i += blockDim.x) {
    uint4 v[WORLD];
#pragma unroll
    for (int p = 0; p < WORLD; p++) v[p] = p == r ? in[r * qn + i] : ld_host(half + (p * WORLD + r) * qn + i);
    float a[8]; set8(a, v[0]); add8(a, v[1]); add8(a, v[2]); add8(a, v[3]);
    const uint4 y = pack8(a);
    out[r * qn + i] = y;
    result[r * qn + i] = y;
  }
  exchange(cm, 1, e);
  for (long long i = lo + t; i < hi; i += blockDim.x) {
#pragma unroll
    for (int p = 0; p < WORLD; p++) if (p != r) out[p * qn + i] = ld_host(result + p * qn + i);
  }
  finish(cm, e);
}
"""

c_void_p, c_int, c_uint, c_longlong, c_size_t = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_longlong, ctypes.c_size_t


class Comm(ctypes.Structure):
    _fields_ = [("data", c_void_p), ("flags", c_void_p), ("state", c_void_p), ("half", c_longlong),
                ("timeout", c_longlong), ("rank", c_int), ("pad", c_int)]


def load_nvrtc():
    """The NVRTC of torch's CUDA wheels (the runtime's nvidia/cu13), then the system toolkit."""
    major = (torch.version.cuda or "13").split(".")[0]
    candidates = []
    for root in sys.path:
        candidates += sorted(glob.glob(os.path.join(root, "nvidia", f"cu{major}", "lib", "libnvrtc.so*")))
        candidates += sorted(glob.glob(os.path.join(root, "nvidia", "cuda_nvrtc", "lib", "libnvrtc.so*")))
    candidates += [f"libnvrtc.so.{major}", "libnvrtc.so"]
    errors = []
    for path in candidates:
        if "builtins" in path or ".alt." in path:
            continue
        try:
            return ctypes.CDLL(path), path
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    raise RuntimeError("no NVRTC: " + "; ".join(errors[-3:]))


def compile_cubin(arch):
    lib, path = load_nvrtc()
    lib.nvrtcGetErrorString.restype = ctypes.c_char_p
    program = c_void_p()
    status = lib.nvrtcCreateProgram(ctypes.byref(program), SOURCE.encode(), b"flash_next_host_ar.cu", 0, None, None)
    if status:
        raise RuntimeError(f"nvrtcCreateProgram: {lib.nvrtcGetErrorString(status).decode()}")
    options = [f"--gpu-architecture=sm_{arch}".encode(), b"--std=c++17"]
    status = lib.nvrtcCompileProgram(program, len(options), (ctypes.c_char_p * len(options))(*options))
    size = c_size_t()
    lib.nvrtcGetProgramLogSize(program, ctypes.byref(size))
    log = ctypes.create_string_buffer(size.value)
    lib.nvrtcGetProgramLog(program, log)
    if status:
        raise RuntimeError(f"NVRTC ({path}) failed: {lib.nvrtcGetErrorString(status).decode()}\n{log.value.decode()}")
    lib.nvrtcGetCUBINSize(program, ctypes.byref(size))
    cubin = ctypes.create_string_buffer(size.value)
    lib.nvrtcGetCUBIN(program, cubin)
    lib.nvrtcDestroyProgram(ctypes.byref(program))
    return cubin, path


class Driver:
    """The CUDA driver API for one device's primary context (the context torch uses)."""

    def __init__(self, device):
        cu = self.cu = ctypes.CDLL("libcuda.so.1")
        cu.cuGetErrorName.argtypes = [c_int, ctypes.POINTER(ctypes.c_char_p)]
        cu.cuDeviceGet.argtypes = [ctypes.POINTER(c_int), c_int]
        cu.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(c_int), c_int, c_int]
        cu.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(c_void_p), c_int]
        cu.cuCtxGetCurrent.argtypes = [ctypes.POINTER(c_void_p)]
        cu.cuCtxSetCurrent.argtypes = [c_void_p]
        cu.cuModuleLoadData.argtypes = [ctypes.POINTER(c_void_p), c_void_p]
        cu.cuModuleGetFunction.argtypes = [ctypes.POINTER(c_void_p), c_void_p, ctypes.c_char_p]
        cu.cuMemHostRegister_v2.argtypes = [c_void_p, c_size_t, c_uint]
        cu.cuMemHostGetDevicePointer_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), c_void_p, c_uint]
        cu.cuLaunchKernel.argtypes = [c_void_p, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint,
                                      c_void_p, ctypes.POINTER(c_void_p), c_void_p]
        torch.zeros(1, device=device)  # The primary context exists before it is retained.
        dev, self.ctx = c_int(), c_void_p()
        self.check(cu.cuDeviceGet(ctypes.byref(dev), device.index), "cuDeviceGet")
        self.dev = dev.value
        self.check(cu.cuDevicePrimaryCtxRetain(ctypes.byref(self.ctx), self.dev), "cuDevicePrimaryCtxRetain")
        self.current = c_void_p()

    def check(self, status, what):
        if status:
            name = ctypes.c_char_p()
            self.cu.cuGetErrorName(status, ctypes.byref(name))
            raise RuntimeError(f"{what}: {name.value.decode() if name.value else status}")

    def use(self):
        self.cu.cuCtxGetCurrent(ctypes.byref(self.current))
        if self.current.value != self.ctx.value:
            self.check(self.cu.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")

    def attribute(self, attribute):
        value = c_int()
        self.check(self.cu.cuDeviceGetAttribute(ctypes.byref(value), attribute, self.dev), "cuDeviceGetAttribute")
        return value.value

    def functions(self, cubin, names):
        self.use()
        module = c_void_p()
        self.check(self.cu.cuModuleLoadData(ctypes.byref(module), ctypes.cast(cubin, c_void_p)), "cuModuleLoadData")
        self.module = module
        functions = {}
        for name in names:
            function = c_void_p()
            self.check(self.cu.cuModuleGetFunction(ctypes.byref(function), module, name.encode()), name)
            functions[name] = function
        return functions

    def register(self, address, size):
        """Pin and map host memory into this context; returns its device address."""
        self.use()
        self.check(self.cu.cuMemHostRegister_v2(c_void_p(address), size, 0x01 | 0x02), "cuMemHostRegister")
        pointer = ctypes.c_uint64()
        self.check(self.cu.cuMemHostGetDevicePointer_v2(ctypes.byref(pointer), c_void_p(address), 0),
                   "cuMemHostGetDevicePointer")
        return pointer.value


def agree(ok, group, what, error):
    """Whether every rank completed a setup step; logs the local failure."""
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    if not ok:
        logger.warning("flash_next_host_ar: %s failed on this rank: %s", what, error)
    return bool(flag.item())


class HostAllReduce:
    """Out-of-place bf16 all-reduce over pinned host memory for one four-rank group."""

    def __init__(self, group, device, rank, driver, functions, buffer, address):
        self.group, self.device, self.rank = group, device, rank
        self.driver, self.functions = driver, functions
        self.buffer, self.address = buffer, address  # The mapping must outlive the registration.
        self.state = torch.zeros(2, dtype=torch.int32, device=device)
        half = (5 * MAX_BYTES + PAGE - 1) // PAGE * PAGE  # Two-shot: 16 scatter and 4 result quarters.
        self.comm = Comm(data=address + PAGE, flags=address, state=self.state.data_ptr(), half=half,
                         timeout=TIMEOUT_CYCLES, rank=rank)
        self.census = {}

    @staticmethod
    def buffer_bytes():
        half = (5 * MAX_BYTES + PAGE - 1) // PAGE * PAGE
        return PAGE + 2 * half

    def should_all_reduce(self, tensor):
        nbytes = tensor.numel() * tensor.element_size()
        eligible = (tensor.dtype == torch.bfloat16 and tensor.device == self.device and tensor.is_contiguous()
                    and 0 < nbytes <= MAX_BYTES and tensor.data_ptr() % 16 == 0
                    and nbytes % (16 if nbytes <= ONE_SHOT_BYTES else 64) == 0)
        if self.rank == 0 and len(self.census) < CENSUS_LIMIT:
            # The epoch protocol needs every collective of the group on one stream per rank; log the stream.
            key = (tuple(tensor.shape), str(tensor.dtype), torch.cuda.is_current_stream_capturing(),
                   torch.cuda.current_stream(self.device).cuda_stream)
            if key not in self.census:
                self.census[key] = eligible
                method = "nccl" if not eligible else "one-shot" if nbytes <= ONE_SHOT_BYTES else "two-shot"
                logger.info("flash_next_host_ar: %s %s %d B %s stream %#x -> %s", list(key[0]), key[1], nbytes,
                            "graph" if key[2] else "eager", key[3], method)
        return eligible

    def all_reduce(self, tensor):
        out = torch.empty_like(tensor)
        nbytes = tensor.numel() * tensor.element_size()
        n = nbytes // 16
        one_shot = nbytes <= ONE_SHOT_BYTES
        work = n if one_shot else n // WORLD
        grid = max(1, min(MAX_CTAS, -(-work // THREADS)))
        source, target, count = c_void_p(tensor.data_ptr()), c_void_p(out.data_ptr()), c_longlong(n)
        params = (c_void_p * 4)(*[ctypes.addressof(a) for a in (source, target, count, self.comm)])
        self.driver.use()
        stream = torch.cuda.current_stream(self.device).cuda_stream
        self.driver.check(self.driver.cu.cuLaunchKernel(
            self.functions["one_shot" if one_shot else "two_shot"], grid, 1, 1, THREADS, 1, 1, 0,
            c_void_p(stream), params, None), "cuLaunchKernel")
        return out


def create_host_all_reduce(group, device, rank):
    """Build the communicator on every rank of a four-rank CPU group, or return None on all of them."""
    world = dist.get_world_size(group)
    if world != WORLD:
        return None
    driver = functions = error = None
    try:
        driver = Driver(device)
        if not driver.attribute(19):  # CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY
            raise RuntimeError("device cannot map host memory")
        major, minor = torch.cuda.get_device_capability(device)
        cubin, nvrtc = compile_cubin(f"{major}{minor}")
        functions = driver.functions(cubin, ("one_shot", "two_shot"))
    except Exception as exc:  # noqa: BLE001 - every rank must reach the agreement below.
        error = exc
    if not agree(error is None, group, "kernel build", error):
        return None
    # Rank 0 names and sizes a fresh /dev/shm file; every rank maps it, then rank 0 unlinks it.
    size = HostAllReduce.buffer_bytes()
    path = [f"/dev/shm/sglang-flash-next-host-ar-{os.getpid()}-{uuid.uuid4().hex}" if rank == 0 else None]
    error = None
    if rank == 0:
        try:
            descriptor = os.open(path[0], os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            os.ftruncate(descriptor, size)
            os.close(descriptor)
        except OSError as exc:
            error = exc
    dist.broadcast_object_list(path, src=dist.get_global_rank(group, 0), group=group)
    if not agree(error is None, group, "shared buffer creation", error):
        return None
    buffer = address = None
    try:
        descriptor = os.open(path[0], os.O_RDWR)
        try:
            buffer = mmap.mmap(descriptor, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(descriptor)
        address = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
        device_address = driver.register(address, size)
    except Exception as exc:  # noqa: BLE001
        error = exc
    mapped = agree(error is None, group, "shared buffer registration", error)
    if rank == 0:
        os.unlink(path[0])
    if not mapped:
        return None
    if device_address != address:
        logger.info("flash_next_host_ar: device address %#x differs from host address %#x", device_address, address)
    communicator = HostAllReduce(group, device, rank, driver, functions, buffer, device_address)
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    if rank == 0:
        logger.info("flash_next_host_ar: enabled for %d ranks, %d-byte shared buffer, one-shot <= %d B, "
                    "two-shot <= %d B, NVRTC %s", WORLD, size, ONE_SHOT_BYTES, MAX_BYTES, nvrtc)
    return communicator
