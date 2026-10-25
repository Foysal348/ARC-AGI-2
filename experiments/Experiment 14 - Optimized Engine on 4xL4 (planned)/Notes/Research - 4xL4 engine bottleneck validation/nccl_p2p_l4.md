# Multi-GPU communication on Kaggle 4x NVIDIA L4 (GCP G2, PCIe Gen4, no NVLink) for TP4 LLM decode

Labels used below: **[measured]** = a number someone measured and published; **[claimed]** = a vendor or author statement without data; **[inference]** = my own reasoning from the cited numbers. All dates are 2026 unless stated. Research done 2026-10-23.

Workload context (from the assignment): Qwen3.8-27B on SGLang 0.5.x, TP4, decode step about 115–130 ms for 12 requests × 8 DFlash2 tokens = 96 tokens per verify step. Our own measurement on Kaggle 4xL4: a 64 MB all-reduce takes 20.9 ms (algbw about 3.1 GB/s).

---

## Q1. Topology: PCIe layout of g2-standard-48, P2P support, virtualization effects

### Takeaway
No published `nvidia-smi topo -m` exists for a 4xL4 g2-standard-48 or for Kaggle's 4xL4. The closest evidence points two ways. A 2xL4 GKE node shows the GPUs as **PHB** (through a PCIe host bridge) with P2P read/write **OK**. Spare Cores' nvbandwidth run on g2-standard-48 reports **P2P supported, about 10 GB/s**. Kaggle's own 2xT4 VM, however, shows **P2P unavailable** (PHB inside a VM). Even when P2P is available, NCCL's default policy does **not** use P2P across a host bridge for 3+ GPUs, so the default transport on a PHB 4xL4 box is SHM either way.

### Cited Findings
- G2 / L4 interconnect is "PCIe Gen 4"; no NVLink is listed for L4. L4 = Ada Lovelace, 24 GB GDDR6 (300 GBps), 1/2/4/8 GPUs per VM **[claimed, vendor doc]** — [GCP GPU machine types](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines)
- g2-standard-48: 48 vCPUs, 192 GB memory by default (custom 192–216 GB), 4x L4 (96 GB), 50 Gbps network. "The G2 machine type is only available on the Cascade Lake platform." The G2 section says nothing about NUMA, PCIe layout, or P2P **[claimed]** — [GCP GPU machine types](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines)
- Spare Cores, g2-standard-48: 48 vCPUs on 24 physical cores, hypervisor flag set (VM), L3 39 MiB. "CPU and System Topology" section is empty (no NUMA data). nvbandwidth v0.5 (driver 535.183.01, CUDA 12.6): GPU→host 13.2 GB/s and host→GPU 12.4 GB/s per GPU, host-device latency 921 ns. **"P2P supported: yes", P2P bandwidth 10.1 GB/s, P2P duplex 8.2 GB/s, P2P gather 9.4 GB/s, P2P latency 921 ns** **[measured; the P2P latency equals the host-device latency exactly, which looks like a copy or labelling error, so treat it with caution]** — [Spare Cores g2-standard-48](https://sparecores.com/server/gcp/g2-standard-48)
- GKE 1.27 on **g2-standard-24 (2x L4)**, driver 535.161.07: `topo -m` shows GPU0↔GPU1 = **PHB**, CPU affinity 0-23, NUMA affinity 0. `topo -p2p r` = OK, `w` = OK, `p` = OK, `n` = NS, `a` (atomics) = NS **[measured]** — [gbrayut GKE gist](https://gist.github.com/gbrayut/71ef2d570765a8a07dd8a68574cd8c85)
- Kaggle 2x Tesla T4 VM: `topo -m` GPU0↔GPU1 = PHB on a flat virtual PCI bus (00:04, 00:05). NCCL reports `intraNodeP2pSupport 0`, so P2P is unavailable and NCCL uses `via SHM/direct/direct` **[measured, Kaggle, T4 not L4]** — [sxwangsxwang1/nccl-allreduce-bench](https://github.com/sxwangsxwang1/nccl-allreduce-bench)
- NCCL default P2P policy, read from source (NCCL 2.29.7, `ncclTopoCheckP2p` in `src/graph/paths.cc`): it starts from `p2pLevel = PATH_PXB`. Code comment: "By default don't use P2P across CPU Host Bridges and further apart". The only override is PATH_SYS for x86 AMD hosts with ≤2 GPUs. Result: **3+ GPUs on CPU root ports, any x86 host → SHM**; 2 GPUs on root ports, Intel host → SHM. On a KVM guest with 4 PHB GPUs, P2P OK by driver, transport was SHM by default and became `P2P/CUMEM` only with `NCCL_P2P_LEVEL=PHB` or `SYS` (`PXB` stayed SHM) **[measured, 4x RTX 5060 Ti]** — [HyperQwen issue #254](https://github.com/syv-ai/HyperQwen/issues/254)
- An NCCL maintainer says NCCL 2.23 "avoids P2P in more cases" and that `NCCL_P2P_LEVEL=PHB` does the same. Another maintainer: NCCL now avoids P2P where it would perform poorly, even when nominally supported **[claimed, maintainers]** — [NVIDIA/nccl #814](https://github.com/NVIDIA/nccl/issues/814)
- Virtualization: "VMs require ACS to function, so disabling ACS is not an option there." IO virtualization (VT-d/IOMMU) can redirect "all PCI point-to-point traffic to the CPU root complex", causing slowdowns or hangs. "A virtual PCI topology exposed through /sys can result in suboptimal performance." If P2P fails even though `topo -p2p` says OK, common causes are the IOMMU and ACS **[claimed, NVIDIA doc]** — [NCCL GPU troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)
- A driver that says P2P is OK is not proof that P2P works. One report: `topo -p2p w` OK for all 64 pairs, yet TP2 P2P/CUMEM hung under a translated IOMMU while 4- and 8-rank SHM runs completed **[measured]** — [vLLM RFC #51513](https://github.com/vllm-project/vllm/issues/51513)
- With ACS redirect on a Gen4 PCIe switch, one-way P2P latency was 1.79 µs vs 1.58 µs through the root complex and 1.25 µs with ACS cleared **[measured, V100 box]** — [squibloads/vllm-pcie-allreduce](https://github.com/squibloads/vllm-pcie-allreduce)
- GCP built a "software-defined PCIe fabric" with P2P only for its newer **G4** (RTX PRO 6000) VMs. Against "standard non-P2P offerings" it claims up to 2.2x on collectives and up to 41% lower ITL. The post does not say whether G2 has P2P **[claimed]** — [GCP blog, G4 P2P fabric](https://cloud.google.com/blog/products/compute/g4-vms-p2p-fabric-boosts-multi-gpu-workloads)

### Inferences
- [inference] On GCP g2 the L4s most likely appear as **PHB** pairs (GKE 2xL4 shows PHB; T4 Kaggle VM shows PHB). With 4 GPUs at PHB, **stock NCCL picks SHM even if P2P works**, unless `NCCL_P2P_LEVEL=PHB` or `SYS` is set.
- [inference] Whether P2P works on *Kaggle's* 4xL4 is unknown. Spare Cores (g2-standard-48) and GKE (g2-standard-24) say P2P works on GCP L4. Kaggle's T4 VM did not expose P2P, and the GCP G4 blog markets P2P as a new G4 feature against "non-P2P offerings". Kaggle may also use different VM images or GPU-passthrough settings from plain GCP. Only a run on the Kaggle box settles this (commands in Q7).
- [inference] Spare Cores' L3 = 39 MiB is about one Cascade Lake socket's L3, and GKE shows NUMA affinity 0. So the VM probably exposes one NUMA node, and the cross-socket (SYS) penalty is probably not in play. Unverified.

### Gaps
- No published `nvidia-smi topo -m` / `topo -p2p` for 4xL4 g2-standard-48 or for Kaggle 4xL4.
- No GCP statement on ACS/IOMMU mode for G2 passthrough.

---

## Q2. Transport without P2P (SHM) and expected latency/bandwidth; published numbers for L4 and similar PCIe GPUs

### Takeaway
Without P2P (or with stock NCCL on PHB at 4 GPUs), NCCL stages data through host shared memory (SHM). On PCIe Gen3/Gen4 GPUs this gives a **bus-bandwidth plateau of only about 2.4–4.6 GB/s**, about 12–25 µs per small all-reduce under CUDA graphs (about 40 µs eager), and **about 200–500 µs for a 1 MB all-reduce**. I found no published nccl-tests numbers for L4.

### Cited Findings
- SHM transport "stages data in host memory between devices when P2P isn't possible". If SHM is disabled, NCCL falls back to the network (IB or sockets) **[claimed]** — [NCCL env docs 2.32.3](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- **Kaggle 2xT4 (Gen3 x16), SHM, NCCL 2.25.1** **[measured]** — [nccl-allreduce-bench](https://github.com/sxwangsxwang1/nccl-allreduce-bench):
  - 8 B–1 KB: 14–16 µs over SHM vs 60 µs over sockets (`NCCL_SHM_DISABLE=1`).
  - busBW Ring / Tree / Socket: 16 KB 0.83 / 0.40 / 0.13; **1 MB 3.56 / 3.39 / 1.57**; 64 MB 3.97 / 4.58 / 2.08; 512 MB 3.98 / 4.53 / 2.13 GB/s.
  - α-β fits: LL α = 12.4 µs, β = 2.11 GB/s (8–64 KB); Simple α = 29.4 µs, β = 3.97 GB/s (≥512 KB). NCCL used LL up to 64 KB and Simple from 512 KB. **LL128 is disabled on PCIe.** Time jumps from 21 µs to 41 µs between 8 KB and 16 KB.
  - First run of a session was up to about 30% slower at 64–128 KB.
- **4x RTX 5060 Ti, Gen3 x8, KVM guest, all PHB, NCCL 2.29.7, CUDA graph, bf16 × 5120 hidden, median** **[measured]** — [HyperQwen #254](https://github.com/syv-ai/HyperQwen/issues/254):

  | tokens | bytes | NCCL default (SHM) | P2P forced (`NCCL_P2P_LEVEL=SYS`) |
  |---|---|---|---|
  | 1 | 10,240 | 43.88 µs | 29.08 µs |
  | 8 | 81,920 | 64.09 µs | 55.51 µs |
  | 16 | 163,840 | 98.36 µs | 85.16 µs |
  | 32 | 327,680 | 157.34 µs | 142.82 µs |
  | 64 | 655,360 | 267.61 µs | 248.50 µs |
  | 128 | 1,310,720 | 498.09 µs | 467.28 µs |
  | 512 | 5,242,880 | 1,863.94 µs | 1,794.32 µs |
  | 2048 | 20,971,520 | 7,338.91 µs | 7,136.45 µs |

- **6x RTX 4090, `NCCL_P2P_DISABLE=1`**: 1 MB 515.2 µs (busbw 3.39), 2 MB 1233 µs (2.83), 16 MB 10,892 µs (2.57), 64 MB 46,206 µs (algbw 1.45, busbw 2.42 GB/s). The maintainer blames AMD CPUs that "don't like having lots of flows at the same time" (the bidirectional traffic can cut throughput by more than 2x). p2pBandwidthLatencyTest on the same box: unidirectional with P2P disabled 20–22 GB/s, P2P writes 9.9–10.2 GB/s; latency 11–14 µs with P2P disabled vs about 1.2 µs for P2P writes **[measured]** — [NVIDIA/nccl #814](https://github.com/NVIDIA/nccl/issues/814)
- **8x A10, NCCL 2.18.1, transport not stated**: 32 KB 36.8 µs, 64 KB 38.5 µs, 128 KB 118.1 µs, 256 KB 127.0 µs, 512 KB 153.0 µs, **1 MB 199.6 µs (algbw 5.25, busbw 9.19 GB/s)** **[measured]** — [NVIDIA/nccl #922](https://github.com/NVIDIA/nccl/issues/922)
- **Dual RTX 5070 Ti, PCIe 4.0 x16, no P2P, NCCL SHM**: 20.97 MB in 10,859 µs (1.9 GB/s) "under every NCCL_PROTO"; NCCL takes 22–25 µs per call inside a CUDA graph and about 41 µs eager for 4096 bf16 (8 KB) **[measured]** — [SGLang PR #39605](https://github.com/sgl-project/sglang/pull/39605)
- On a V100 box without `NCCL_P2P_LEVEL=SYS`, "NCCL staged cross-board traffic through host shared memory at 3.5 GB/s" **[measured]** — [squibloads/vllm-pcie-allreduce](https://github.com/squibloads/vllm-pcie-allreduce)
- AWS g5.48xlarge (8x A10G): all pairs PHB, NUMA 0-1. All-reduce at 28 MB was about 2 GB/s on NCCL 2.18 and about 1 GB/s on NCCL 2.20 (regression); `NCCL_TOPO_FILE` brought it to about 1.6 GB/s **[measured, maintainer recap]** — [NVIDIA/nccl #1298](https://github.com/NVIDIA/nccl/issues/1298)
- Reference with P2P: 4x RTX 4090 with a patched driver that forces P2P reaches about 25 GB/s busbw at 16 MB on NCCL **[measured]** — [vLLM #36481 (search summary)](https://github.com/vllm-project/vllm/issues/36481)

### Inferences
- [inference] SHM small-message latency on 4 GPUs: about 12–25 µs per call under CUDA graphs (8–16 KB), about 40–65 µs at 10–80 KB on the KVM 4-GPU box. 1 MB costs about 200–500 µs depending on the host. PCIe Gen4 vs Gen3 matters little for SHM, because the host-memory staging and CPU-side flows set the limit, not the link (the 5070 Ti Gen4 x16 box was no faster than the T4 Gen3 box).

### Gaps
- No nccl-tests table for L4 (any count), for 4xA10G g5.12xlarge, or for 4x L40S without P2P was found.
- No published small-message latency for 4-rank SHM at exactly 8 KB–1 MB on Gen4 Intel hosts.

---

## Q3. Is our measurement (64 MB in 20.9 ms ≈ 3.1 GB/s algbw) consistent with SHM?

### Takeaway
Yes. 3.1 GB/s algbw at 4 ranks is **busbw ≈ 4.7 GB/s**. That sits in the SHM range published for PCIe GPUs (2.4–4.6 GB/s busbw) and far below what P2P rings reach on Gen4 (about 25 GB/s busbw on forced-P2P 4090s). Strong evidence that Kaggle's 4xL4 runs NCCL over SHM today. Confirm it with `NCCL_DEBUG=INFO` (look for `via SHM/direct/direct`).

### Cited Findings
- busbw for AllReduce = algbw × 2(n−1)/n; algbw = S/t **[definition]** — [nccl-tests PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
- SHM plateaus: Kaggle T4 3.97 (ring) / 4.58 (tree) GB/s busbw at 64 MB — [nccl-allreduce-bench](https://github.com/sxwangsxwang1/nccl-allreduce-bench). 6x4090 without P2P: 2.42 GB/s at 64 MB — [NVIDIA/nccl #814](https://github.com/NVIDIA/nccl/issues/814). V100 SHM 3.5 GB/s — [squibloads](https://github.com/squibloads/vllm-pcie-allreduce). 4x 5060 Ti SHM: 20.97 MB in 7.34 ms — [HyperQwen #254](https://github.com/syv-ai/HyperQwen/issues/254) **[all measured]**
- With P2P: 4x4090 forced P2P about 25 GB/s busbw at 16 MB — [vLLM #36481](https://github.com/vllm-project/vllm/issues/36481); g2-standard-48 nvbandwidth P2P 10.1 GB/s per pair — [Spare Cores](https://sparecores.com/server/gcp/g2-standard-48) **[measured]**

### Inferences
- [inference] 64 MiB / 20.9 ms = 3.21 GB/s algbw (3.06 if 64 MB is decimal) → busbw 4.6–4.8 GB/s for n = 4. The 4x 5060 Ti SHM point works out to 2.86 GB/s algbw / 4.3 GB/s busbw. Ours is about 10% better, consistent with SHM on a better host. A working P2P ring at about 10 GB/s per link would give roughly 6–7 GB/s algbw (64 MB in about 10 ms). If P2P were active, we would expect about 2x better than what we measured.
- [inference] Fitting our large-message rate to the ~1 MB decode messages: 983,040 B / 3.1 GB/s ≈ 317 µs, plus α of about 20–30 µs ≈ **340–350 µs per all-reduce**. This agrees with the interpolated 4x 5060 Ti SHM number (≈380 µs at 96 tokens × 5120 × 2 B).

### Gaps
- Our 64 MB number does not show the transport or the per-size curve. A full sweep is needed (Q7).

---

## Q4. NCCL tuning for small/medium messages on PCIe without (or with) P2P

### Takeaway
The levers that matter, in order of expected value for our ~1 MB decode messages:
1. **Turn on P2P if the Kaggle VM supports it** (`NCCL_P2P_LEVEL=SYS` or `PHB`), after checking correctness.
2. **Try `NCCL_SHM_USE_CUDA_MEMCPY=1`** (copy engines): it raises SHM bandwidth by up to 6x but adds latency; the crossover is near our message size, so measure it.
3. **Sweep channel/CTA counts** (`NCCL_MIN_CTAS` / `NCCL_MAX_CTAS`).
4. **Pin the NCCL version** and avoid known regressions.

`NCCL_PROTO` and `NCCL_ALGO` have small or negative effects in the published data (LL128 is unavailable on PCIe anyway).

### Cited Findings
- **NCCL_P2P_LEVEL**: values LOC, NVL, PIX, PXB, PHB (same NUMA, through CPU), SYS (across NUMA); default unset (NCCL picks). Integer values are legacy and discouraged **[doc]** — [NCCL env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- Forcing P2P on a 4-GPU PHB KVM box (Gen3 x8): all-reduce latency −33.9% at 10 KB, −13.3% at 80 KB, −9.7% at 328 KB, −6.1% at 1.31 MB. Decode tok/s +2% to +16% (C4 greedy +10–16%; C8 overlapping, not established). Measured step saving 0.74 ms at 8 tokens and 3.93 ms at 64 tokens per step **[measured]** — [HyperQwen #254](https://github.com/syv-ai/HyperQwen/issues/254)
- `NCCL_P2P_LEVEL=SYS` improved NCCL-only decode on 4x RTX PRO 6000 by about 3.6% at c=1 (67.3 → 69.7 tok/s) **[measured]** — [rtx6kpro pcie-oneshot-allreduce](https://github.com/local-inference-lab/rtx6kpro/blob/master/optimization/pcie-oneshot-allreduce.md)
- On 8x A40 PCIe, allowing P2P over PCIe made NCCL all-reduce slightly *slower* (about 23 GB/s busbw at 32 MB) **[measured, old issue]** — [nccl-tests #74 (search summary)](https://github.com/NVIDIA/nccl-tests/issues/74)
- Risk: P2P under a translated IOMMU/ACS can hang silently (TP2 hang; IOMMU `IO_PAGE_FAULT`s) **[measured]** — [vLLM RFC #51513](https://github.com/vllm-project/vllm/issues/51513); NCCL docs warn the same — [NCCL GPU troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)
- **NCCL_SHM_USE_CUDA_MEMCPY=1** on no-P2P dual 5070 Ti: SHM bandwidth 1.9 → **12.5 GB/s**, but 10 KB decode all-reduce latency 39 → 81 µs **[measured]** — [SGLang PR #39605](https://github.com/sgl-project/sglang/pull/39605). On 8x A10 (NCCL 2.18.1) it made 128 KB–1 MB about 2x slower (1 MB 199.6 → 414.4 µs); 32–64 KB unchanged. Maintainer: "CE engine imposes a higher launch overhead"; "designed for medium and large message sizes (e.g. tens of MB to GB)" **[measured + maintainer]** — [NVIDIA/nccl #922](https://github.com/NVIDIA/nccl/issues/922). A maintainer suggested it as "roughly 2x on G5/G6" (AWS A10G/L4 families); no outcome reported **[claimed]** — [NVIDIA/nccl #1298](https://github.com/NVIDIA/nccl/issues/1298). A hang with PyTorch was reported (old) — [NVIDIA/nccl #803 (search summary)](https://github.com/NVIDIA/nccl/issues/803)
- **NCCL_PROTO**: LL / LL128 / Simple; LL128 is enabled only on supported platforms, and forcing it elsewhere "can corrupt data". Users are "discouraged from setting it" **[doc]** — [NCCL env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html). On SHM, the 5070 Ti PR saw the same 1.9 GB/s "under every NCCL_PROTO" **[measured]** — [SGLang PR #39605](https://github.com/sgl-project/sglang/pull/39605)
- **NCCL_ALGO**: Ring, Tree, CollnetDirect, NVLS, NVLSTree, PAT **[doc]** — [NCCL env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html). On SHM T4, Tree is +14.5% at large sizes but **slower at 8 KB–1 MB** (16 KB 0.40 vs 0.83 GB/s; 1 MB 3.39 vs 3.56) **[measured, n=2]** — [nccl-allreduce-bench](https://github.com/sxwangsxwang1/nccl-allreduce-bench)
- **NCCL_NTHREADS** (64/128/256/512; default 512 on recent GPUs): 512 vs 256 gave +1.8% at the plateau on T4 SHM **[doc + measured]** — [NCCL env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html), [nccl-allreduce-bench](https://github.com/sxwangsxwang1/nccl-allreduce-bench)
- **NCCL_MIN_CTAS / NCCL_MAX_CTAS** (replace MIN/MAX_NCHANNELS since 2.17; up to 64): "Excessive CTAs can hurt small-message performance"; small messages may use fewer channels anyway **[doc]** — [NCCL env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html). On g5.24xlarge (4x A10G), `NCCL_MIN_CTAS=1 NCCL_MAX_CTAS=1` "recovered" performance to NCCL 2.18.3 levels (approximate, from nvtop) **[measured, weak]**; maintainer also suggested `NCCL_MAX_NCHANNELS=2` — [NVIDIA/nccl #1298](https://github.com/NVIDIA/nccl/issues/1298). On RTX PRO 6000 PCIe, `NCCL_MIN_NCHANNELS=8` is described as a "significant bandwidth improvement" (no numbers) **[claimed]** — [rtx6kpro sglang.md](https://github.com/local-inference-lab/rtx6kpro/blob/master/inference-engines/sglang.md)
- **NCCL_BUFFSIZE**: default 4 MiB per GPU pair; powers of 2 recommended **[doc]**; no measured effect found — [NCCL env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- **NCCL_CUMEM_ENABLE**: auto-on since 2.19. **NCCL_CUMEM_HOST_ENABLE**: default 1 since 2.24 with driver ≥12.6 / runtime ≥12.2 **[doc]** — [NCCL env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html). vLLM troubleshooting advice (via blog): try `NCCL_CUMEM_HOST_ENABLE=0` first for hangs, `NCCL_P2P_DISABLE=1` as a last resort **[claimed]** — [st-hakky vLLM TP troubleshooting](https://book.st-hakky.com/en/data-science/vllm-multi-gpu-tensor-parallel-troubleshooting)
- **NCCL_SHM_DISABLE=1** makes it worse: SHM → sockets, α about 4x higher (14–16 → 60 µs) and bandwidth −47% **[measured]** — [nccl-allreduce-bench](https://github.com/sxwangsxwang1/nccl-allreduce-bench)
- **NCCL_TOPO_FILE / NCCL_GRAPH_FILE**: on g5.48xlarge a topo file improved DDP time 7 → 4.5 h (torch2.0) — [NVIDIA/nccl #1298](https://github.com/NVIDIA/nccl/issues/1298); a graph XML gave +11% throughput on 2-NUMA Genoa — [rtx6kpro sglang.md](https://github.com/local-inference-lab/rtx6kpro/blob/master/inference-engines/sglang.md) **[measured, other setups]**
- **NCCL_GRAPH_REGISTER** only matters with CollNet and full P2P; **NCCL_RUNTIME_CONNECT** default 1 (lazy connect) **[doc]** — [NCCL env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- Version regressions exist: NCCL 2.20.5 halved all-reduce bandwidth vs 2.18.5 on 8x A10G PHB **[measured]** — [NVIDIA/nccl #1298](https://github.com/NVIDIA/nccl/issues/1298)

### Inferences
- [inference] Our per-call size (about 0.94 MiB at 96 tokens) is in the regime where SHM is **bandwidth-bound**, not latency-bound. Levers that raise bandwidth (P2P, CE memcpy, more channels) matter more than LL/LL128 tweaks. The A10 data (CE memcpy 2x slower at 1 MB) and the 5070 Ti data (12.5 vs 1.9 GB/s) conflict. The crossover depends on host and NCCL version, so an A/B at exactly 983,040 B under CUDA graphs is required.
- [inference] If P2P is available on Kaggle, `NCCL_P2P_LEVEL=SYS` is the cheapest test. On a Gen4 x16 box with about 10 GB/s P2P per pair (Spare Cores), the bandwidth-bound part of a 1 MB all-reduce should drop well below SHM. HyperQwen's small 6% gain at 1.3 MB was on Gen3 x8 links, where the link itself limits.

### Gaps
- No NCCL documentation text for `NCCL_SHM_USE_CUDA_MEMCPY` / `NCCL_SHM_MEMCPY_MODE` / `NCCL_P2P_USE_CUDA_MEMCPY` was retrieved (the page excerpts did not include them).
- No published effect of `NCCL_BUFFSIZE` or `NCCL_MIN_CTAS` on 4-rank SHM at 1 MB.

---

## Q5. Alternatives to NCCL in SGLang 0.5.x / vLLM on sm_89 PCIe

### Takeaway
At **TP4 on PCIe without NVLink, stock SGLang and vLLM fall back to NCCL**. Their custom all-reduce runs at TP2 or with full NVLink only. Torch symmetric memory is gated on capability tables and multicast (Hopper/Blackwell-oriented). The PCIe-specific kernels from 2026 (FlashInfer PCIe-IPC in SGLang 0.5.21, the lukealonso PCIe one-shot fork) were built and tested on SM120 RTX PRO 6000 and need working P2P. None is verified on sm_89. Nothing published accelerates TP4 all-reduce **without P2P** except NCCL SHM. The host-staged SGLang PR exists but is TP2-only and unmerged.

### Cited Findings
- **SGLang V1 CustomAllreduce**: `_SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]`, but `should_custom_ar` returns true only `if self.world_size == 2 or self.full_nvlink`. Comment: "for 4 or more non NVLink-capable GPUs, custom allreduce provides…" (no benefit). CUDA `_MAX_CAR_SIZE = 8 MiB` **[source code]** — [SGLang custom_all_reduce.py (main)](https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/distributed/device_communicators/custom_all_reduce.py)
- SGLang **CustomAllReduceV2** is admitted only when `full_nvlink is True`, so PCIe rigs run V1 (silent; success signal is "Registering N cuda graph addresses"). On 2x3090 PCIe with P2P, V1 on vs off gave +0.6% / +1.8% decode, which is boot noise **[measured]** — [club-3090 PR #1470](https://github.com/noonghunna/club-3090/pull/1470). An older SGLang log: "Custom allreduce is disabled because it's not supported on more than two PCIe-only GPUs" **[log, Feb 2025]** — (search summary of [club-3090 / sglang discussions](https://github.com/noonghunna/club-3090/pull/1470))
- **vLLM CustomAllreduce**: disabled with a warning when `world_size > 2` and not fully NVLink-connected; at ws = 2, only `_can_p2p` gates it **[source description]** — [vLLM RFC #51513](https://github.com/vllm-project/vllm/issues/51513)
- vLLM all-reduce dispatch order: NCCL symm-mem → QuickReduce (ROCm, MI300) → FlashInfer PCIe IPC (`VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC`) → FlashInfer AR → AITER (ROCm) → Custom → Torch symm-mem (`VLLM_ALLREDUCE_USE_SYMM_MEM`) → PyNCCL **[source]** — [vLLM cuda_communicator](https://docs.vllm.ai/en/stable/api/vllm/distributed/device_communicators/cuda_communicator.html). **QuickReduce = ROCm/MI300 only** (same source).
- vLLM **SymmMemCommunicator** disables itself if the capability string is missing from `SYMM_MEM_ALL_REDUCE_MAX_SIZES`, if the world size isn't listed, or if `handle.multicast_ptr == 0`. `_WORLD_SIZES_MULTIMEM = {"9.0": [4, 6, 8], "10.0": [6, 8]}` **[source, v0.16]** — [vLLM symm_mem docs](https://docs.vllm.ai/en/v0.16.0/api/vllm/distributed/device_communicators/symm_mem/). Users see "Device capability 8.6 not supported", "7.5 not supported", "10.3 not supported" **[logs]** — [vLLM #28910](https://github.com/vllm-project/vllm/issues/28910), [vLLM #29707](https://github.com/vllm-project/vllm/issues/29707), [vLLM #30630](https://github.com/vllm-project/vllm/issues/30630). "Device capability 12.0 not supported" on 5060 Ti — [HyperQwen #254](https://github.com/syv-ai/HyperQwen/issues/254); sm_120 rejected → PYNCCL on 5090 — [club-3090 FAQ](https://github.com/noonghunna/club-3090/blob/master/docs/FAQ.md)
- SGLang **TorchSymmMemCommunicator** (`--enable-torch-symm-mem`) disables itself without multicast. The two-shot fallback PR (only P2P mappings needed) was **closed unmerged** on 2026-10-07. Its data, on H200 NVLink: 8 KB two-shot 6.3 µs (ws2) / 7.8–14.6 µs (ws4) vs NCCL 23–27 µs under CUDA graph, and "all-reduce is about 12% of decode time" there **[measured]** — [SGLang PR #30364](https://github.com/sgl-project/sglang/pull/30364)
- **FlashInfer PCIe-IPC all-reduce in SGLang**: opt-in `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE` (default off), cap `SGLANG_PCIE_IPC_MAX_NUMEL`, bf16 only, TP group only, world sizes 8 and 4 (none for 2), tuning cannot run under graph capture, needs FlashInfer with flashinfer#4393 (merged 2026-08-20). Merged 2026-09-29 (commit c7be3e9). Tested on SM120 RTX PRO 6000 "switch-free" hosts. TP8: TPOT −36% (8k bs1); **TP4 DeepSeek-V4-Flash: TPOT 20.55 → 18.49 ms (−10%) bs1, 31.74 → 30.24 ms (−5%) bs4** **[measured]** — [SGLang PR #34528](https://github.com/sgl-project/sglang/pull/34528). Listed in **SGLang v0.5.21** (released 2 Oct) release notes as "[SM120] Add optional FlashInfer PCIe-IPC all-reduce for switch-free hosts (#34528)" — [SGLang v0.5.21 release](https://github.com/sgl-project/sglang/releases/tag/v0.5.21). FlashInfer docs: "Custom all-reduce for intra-node PCIe machines without NVLink". Above about 1 MB it can use copy engines (`COPY_ENGINE_RING`); kernels spin on peer flags with no timeout **[doc]** — [FlashInfer comm API](https://docs.flashinfer.ai/api/comm.html)
- **PCIe one-shot all-reduce (lukealonso/sglang fork, `--enable-pcie-oneshot-allreduce`)**: every GPU writes to all peers over P2P, then a system-scope barrier; optional fused AR+RMSNorm (`--enable-pcie-oneshot-allreduce-fusion`); auto crossover. **P2P is required**: "Without the config, custom AR runs about 15× slower than NCCL". 4x RTX PRO 6000 TP4 (P2P): 8 KB 6.4 vs 13.9 µs NCCL; 128 KB 17.6 vs 24.7; 512 KB 51.0 vs 71.7; **1 MB 95.6 vs 85.2 (NCCL wins)**. Decode +5–11% (Qwen3.5-397B TP4) **[measured]** — [rtx6kpro pcie-oneshot-allreduce](https://github.com/local-inference-lab/rtx6kpro/blob/master/optimization/pcie-oneshot-allreduce.md)
- V100 PCIe push kernel (1Cat-vLLM patch): 5 KiB 5.1 vs 16.0 µs NCCL; 80 KiB 30.7 vs 33.5 µs; 4-GPU decode 75.9 → 91.5 tok/s. Needs P2P with `NCCL_P2P_LEVEL=SYS`, `VLLM_CAR_P2P_MESH=1` **[measured]** — [squibloads/vllm-pcie-allreduce](https://github.com/squibloads/vllm-pcie-allreduce). On 8x 3090 (NVLink pairs + PCIe), the forced custom AR was "no faster on any model" (87.7 vs 91.75 t/s) **[measured]** — [loktar00/multigpu-p2p-tweaks](https://github.com/loktar00/multigpu-p2p-tweaks)
- **Host-staged all-reduce for GPUs without peer access** (SGLang PR #39605): `SGLANG_HOST_STAGED_ALLREDUCE=1` (≥1 MiB, outside graph capture) and `SGLANG_LL_ALLREDUCE=1` (bf16 ≤64 KiB through pinned host memory, graph-capturable). 20.97 MB: 1,494 µs vs NCCL SHM 10,859 µs; 8 KB LL: 6.5 µs vs NCCL 41.5 µs eager. **world_size == 2 only, open/unmerged, CI failing** **[measured]** — [SGLang PR #39605](https://github.com/sgl-project/sglang/pull/39605)
- PyTorch symmetric memory documents `one_shot_all_reduce` / `two_shot_all_reduce`; backends CUDA and NVSHMEM. The DMAConnectivityDetector replaced the NVLink detection **[doc/PR]** — [PyTorch 2.9 blog](https://pytorch.org/blog/pytorch-2-9/), [PyTorch symmetric memory docs](https://docs.pytorch.org/docs/main/symmetric_memory.html) (page body did not render in fetch)

### Inferences
- [inference] On Kaggle 4xL4 with SGLang 0.5.21 stock flags, TP all-reduce = **NCCL (PyNccl) inside CUDA graphs over SHM**. Custom AR V1 is skipped at ws = 4 without NVLink, V2 needs NVLink, and torch symm-mem needs a cc 8.9 table entry or multicast that PCIe Ada does not have.
- [inference] The only in-tree candidate is `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1` (v0.5.21). It uses CUDA IPC peer mappings, so it **cannot work if P2P is off** on the Kaggle VM, and SM120 is the only tested arch. If Kaggle exposes P2P, it is worth a correctness-gated try. Its copy-engine path above 1 MB fits our 0.94 MiB messages.
- [inference] Both P2P one-shot kernels lose to NCCL at about 1 MB on 4 GPUs. With 96-token batches (0.94 MiB per call), one-shot kernels give little; two-shot or copy-engine designs, or fewer bytes per call, matter more.
- [inference] FlashInfer `trtllm_allreduce_fusion` (AR + RMSNorm) targets NVLink/IPC systems; sm_89 support is not documented. MSCCL++ was not researched in depth (see Gaps).

### Gaps
- Whether FlashInfer `pcie_ipc` kernels compile or run on sm_89 (docs silent; PR tagged SM120).
- Whether vLLM's `SYMM_MEM_ALL_REDUCE_MAX_SIZES` lists "8.9" (table not shown in fetched page).
- MSCCL++ on sm_89 PCIe: no source found in this pass.
- lukealonso one-shot is a fork; its sm_89 build status is unknown.

---

## Q6. Overlap during decode with CUDA graphs and the expected communication share of the step; TP4 vs TP2 vs DP comparisons

### Takeaway
I found no published profile for 27–32B TP4 decode on PCIe without P2P. Our own numbers combined with published SHM latencies put communication at roughly **30–50% of our 115–130 ms step**: about 128–139 all-reduces per verify step × about 0.3–0.4 ms each = about 35–55 ms. This is an inference, not a measurement. Stock engines do not overlap TP all-reduce with compute in decode; the all-reduce sits on the critical path inside the CUDA graph.

### Cited Findings
- Model dims from a field report running a hidden-5120, **64-decoder-layer** model (attention or GatedDeltaNet + MLP) with a 5-layer DFlash2 drafter on TP4. "About **139 TP allreduces per verify step**" (counted from model code, not traced). Measured step savings from P2P: 0.74 ms (8 tokens) to 3.93 ms (64 tokens) **[measured + estimate]** — [HyperQwen #254](https://github.com/syv-ai/HyperQwen/issues/254)
- Same report: SHM all-reduce, CUDA graph, bf16: 64 tokens 267.6 µs, 128 tokens 498.1 µs **[measured]** — [HyperQwen #254](https://github.com/syv-ai/HyperQwen/issues/254)
- On H200 NVLink, all-reduce is "about 12% of decode time", so a 4x faster all-reduce kernel was within noise end to end **[measured]** — [SGLang PR #30364](https://github.com/sgl-project/sglang/pull/30364)
- The FlashInfer PCIe-IPC decode gain shrinks with batch: TP4 −10% TPOT at bs1, −5% at bs4; TP8 128k bs4 unchanged **[measured]** — [SGLang PR #34528](https://github.com/sgl-project/sglang/pull/34528)
- Tensor parallel inference "runs two all-reduces per layer per token in common implementations" **[claimed, secondary]** — (search summary; [theneuralbase TP inference](https://theneuralbase.com/deepspeed/learn/advanced/tensor-parallel-inference/))
- TP vs DP data on PCIe: forced TP = 1 on a 5090 beat 2x3090 TP2 by +46/+51% on Gemma 4 31B; dense Qwen3.6-27B on 2x3090 TP2 peaks around 104 tok/s at N = 2, falling to 54 by N = 8 (agent shape) **[measured, different GPUs]** — [club-3090 FAQ](https://github.com/noonghunna/club-3090/blob/master/docs/FAQ.md). A systematic study found single-GPU best for decode-dominated chat (22.2 tok/s vs TP2 20.5) **[measured, hardware not shown in excerpt]** — [arXiv 2512.01644 (search summary)](https://arxiv.org/pdf/2512.01644)
- CPU starvation can stall communication and kernel launches in multi-GPU inference, even with CUDA graphs **[measured, abstract]** — [arXiv 2603.22774](https://arxiv.org/abs/2603.22774)
- GCP claims P2P on G4 gives up to 168% higher TP throughput and 41% lower ITL vs non-P2P offerings **[claimed]** — [GCP G4 P2P blog](https://cloud.google.com/blog/products/compute/g4-vms-p2p-fabric-boosts-multi-gpu-workloads)

### Inferences
- [inference] Bytes per all-reduce for us: 96 tokens × 5120 × 2 B = 983,040 B (0.94 MiB), if the verify step all-reduces the full 96-token batch in bf16 and Qwen3.8-27B has hidden = 5120 (HyperQwen's dims match a 27B Qwen3.x; not confirmed for our exact checkpoint).
- [inference] Count: about 2 per layer × 64 = 128 for the target model, plus drafter all-reduces (smaller). Use ~128–139.
- [inference] Per-call cost under SHM: about 340–380 µs (from our 3.1 GB/s plus α, and the 5060 Ti interpolation). Total ≈ **44–53 ms per step ≈ 35–45% of 115–130 ms**. A loose range of 30–50% allows for message-size and drafter uncertainty. This is large enough that TP4 communication is **plausibly the top single bottleneck**. It must be confirmed with an nsys trace or a NCCL-only replay at the real shapes (Q7).
- [inference] The DFlash2 drafter's all-reduces (5 layers, for 96 tokens) add a smaller amount. If the drafter runs TP4 too, it adds about 10 more ~1 MB calls.
- [inference] Implications to test (not decisions):
  - (a) If P2P works on Kaggle: `NCCL_P2P_LEVEL=SYS` and possibly `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1` could remove a large share.
  - (b) `NCCL_SHM_USE_CUDA_MEMCPY=1` A/B at 1 MB.
  - (c) Configurations with fewer ranks per all-reduce (TP2 × DP2) or none (DP4, if the INT4 model plus KV fits 24 GB) remove or halve the per-step communication, at the cost of KV capacity. HyperQwen shows SHM latency grows almost linearly with bytes, so per-call bytes matter as much as the number of calls.

### Gaps
- No published TP4 vs TP2 vs DP4 comparison on 4x L4 or other 4x PCIe Ada GPUs for 27–35B models was found (the biMEMO card and DGX Spark study were not located in this pass).
- No source describes SGLang decode overlap of TP all-reduce (e.g., two-batch overlap applies to EP/MoE). The "no overlap in decode" statement above is an inference.

---

## Q7. How to measure this on an offline Kaggle notebook

### Takeaway
Run four things, in order: topology/P2P dumps (`nvidia-smi topo -m`, `topo -p2p r/w/p/a`, `torch.cuda.can_device_access_peer`); an NCCL INIT/GRAPH/P2P/SHM debug dump to read the chosen transport; a CUDA-graph PyTorch all-reduce sweep at the real decode sizes under an env-var matrix; and optionally nccl-tests built from source against the pip NCCL. Everything works offline if the nccl-tests source tarball is in the wheelhouse.

### Cited Findings
- `nvidia-smi topo -p2p <cap>` with r (read), w (write), n (NVLink), a (atomics), p (prop/PCIe) **[doc]** — [nvidia-smi manual](https://docs.nvidia.com/deploy/nvidia-smi/index.html), [GKE gist](https://gist.github.com/gbrayut/71ef2d570765a8a07dd8a68574cd8c85)
- A positive `topo -p2p p` is "not a complete system sanity check". Use the `simpleP2P` CUDA sample ("Test passed") or `nvbandwidth`, and compare runs with `NCCL_P2P_DISABLE` / `NCCL_P2P_LEVEL`. IOMMU check: `cat /proc/cmdline`, `dmesg | grep -i -E "iommu|dmar|default domain"`; ACS check: `lspci -vvv | grep ACSCtl` **[doc]** — [NCCL GPU troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)
- `NCCL_DEBUG=INFO` shows the transport, e.g. `Channel 00 : 0[0] -> 1[1] via SHM/direct/direct` vs `via P2P/CUMEM` **[measured logs]** — [nccl-allreduce-bench](https://github.com/sxwangsxwang1/nccl-allreduce-bench), [HyperQwen #254](https://github.com/syv-ai/HyperQwen/issues/254) (suggested boot check `NCCL_DEBUG_SUBSYS=INIT,P2P`)
- nccl-tests build: `make` (or `make -j`); `CUDA_HOME` if not /usr/local/cuda; `NCCL_HOME` (default /usr); MPI optional (`MPI=1 MPI_HOME=…`). Single node: `./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8`. Flags: `-b` min, `-e` max, `-f` factor, `-g` GPUs per thread, `-n` iters (20), `-w` warmup (1), `-c` check, **`-G` capture as CUDA graph and replay** **[doc]** — [NVIDIA/nccl-tests README](https://github.com/NVIDIA/nccl-tests)
- Spare Cores used **nvbandwidth v0.5** for the g2-standard-48 P2P numbers **[measured]** — [Spare Cores](https://sparecores.com/server/gcp/g2-standard-48)
- The first run of a session can be about 30% slower at 64–128 KB; take repeated runs **[measured]** — [nccl-allreduce-bench](https://github.com/sxwangsxwang1/nccl-allreduce-bench)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` crashes the PCIe one-shot kernel; IPC-based custom AR is incompatible with it **[measured]** — [rtx6kpro pcie-oneshot-allreduce](https://github.com/local-inference-lab/rtx6kpro/blob/master/optimization/pcie-oneshot-allreduce.md)

### Ready-to-use patterns (my code, untested on Kaggle — [inference])

**1. Topology and P2P dump (one cell, about 10 s):**
```bash
nvidia-smi -L
nvidia-smi topo -m
for c in r w p a n; do echo "== p2p $c"; nvidia-smi topo -p2p $c; done
nvidia-smi --query-gpu=index,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv
# link gen can read low at idle; query again under load
cat /proc/cmdline; (dmesg 2>/dev/null | grep -i -E "iommu|dmar|default domain") || echo "dmesg not permitted"
(lspci -tv 2>/dev/null; lspci -vvv 2>/dev/null | grep -i ACSCtl) || echo "lspci unavailable"
python - <<'EOF'
import torch
n=torch.cuda.device_count()
print([[int(torch.cuda.can_device_access_peer(i,j)) if i!=j else '-' for j in range(n)] for i in range(n)])
EOF
```

**2. Transport dump (NCCL picks the transport at first collective; the default is lazy connect):**
```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM,ENV,TUNING \
NCCL_TOPO_DUMP_FILE=/kaggle/working/nccl_topo.xml NCCL_GRAPH_DUMP_FILE=/kaggle/working/nccl_graph.xml \
torchrun --nproc_per_node=4 ar_bench.py --sizes 983040 --iters 5 2>&1 | tee /kaggle/working/nccl_init.log
grep -E " via |P2P level|intraNodeP2pSupport|NCCL version|Channel 00" /kaggle/working/nccl_init.log | head -40
```
(`NCCL_TOPO_DUMP_FILE` / `NCCL_GRAPH_DUMP_FILE` are NCCL env vars I know of but did not re-verify in the 2.32 doc excerpt fetched here.)

**3. CUDA-graph all-reduce sweep at decode sizes (`ar_bench.py`):**
```python
import os, sys, json, argparse, torch, torch.distributed as dist
p = argparse.ArgumentParser()
p.add_argument("--sizes", default="8192,16384,32768,65536,81920,131072,262144,524288,983040,1048576,2097152,67108864")
p.add_argument("--iters", type=int, default=200)
a = p.parse_args()
dist.init_process_group("nccl")
r = dist.get_rank(); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
res = {}
for nbytes in map(int, a.sizes.split(",")):
    x = torch.randn(nbytes // 2, dtype=torch.bfloat16, device="cuda")
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(10): dist.all_reduce(x)          # warmup + lazy connect outside capture
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    reps = 1 if nbytes > (8 << 20) else 20
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps): dist.all_reduce(x)
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); dist.barrier()
    t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
    n = max(5, a.iters // reps)
    t0.record()
    for _ in range(n): g.replay()
    t1.record(); torch.cuda.synchronize()
    us = t0.elapsed_time(t1) * 1000 / (n * reps)
    W = dist.get_world_size(); alg = nbytes / us / 1e3
    res[nbytes] = dict(us=round(us, 2), algbw=round(alg, 3), busbw=round(alg * 2 * (W - 1) / W, 3))
    # eager timing too (no graph), for comparison with engine eager paths
if r == 0:
    print(json.dumps({"env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}, "res": res}, indent=1))
dist.destroy_process_group()
```
Env-var matrix to run (each a separate `torchrun`, 3 repeats; write all to `/kaggle/working`):
- baseline (unset)
- `NCCL_P2P_LEVEL=SYS` (only if step 1 shows P2P OK; also check the result for correctness: `dist.all_reduce` of `torch.full(…, rank+1)` must equal 10)
- `NCCL_P2P_LEVEL=PHB`
- `NCCL_SHM_USE_CUDA_MEMCPY=1`
- `NCCL_MIN_CTAS=4 NCCL_MAX_CTAS=4`, then 8/8, then 1/1
- `NCCL_ALGO=Tree`; `NCCL_ALGO=Ring`
- `NCCL_PROTO=Simple`; `NCCL_PROTO=LL` (diagnostic only)
- `NCCL_NTHREADS=256`
- `NCCL_BUFFSIZE=1048576` and `8388608`

**4. nccl-tests from source, offline (source tarball shipped in the wheelhouse):**
```bash
NCCL_PY=$(python -c "import nvidia.nccl, os; print(os.path.dirname(nvidia.nccl.__file__))")   # pip NCCL used by torch
ls $NCCL_PY/include $NCCL_PY/lib                       # need nccl.h and libnccl.so.2
[ -e $NCCL_PY/lib/libnccl.so ] || ln -s libnccl.so.2 $NCCL_PY/lib/libnccl.so   # linker needs the unversioned name
cd /kaggle/working/nccl-tests && make -j MPI=0 CUDA_HOME=/path/to/cuda-13.0 NCCL_HOME=$NCCL_PY \
     NVCC_GENCODE="-gencode=arch=compute_89,code=sm_89"
export LD_LIBRARY_PATH=$NCCL_PY/lib:$LD_LIBRARY_PATH
./build/all_reduce_perf -b 8K -e 64M -f 2 -g 4 -n 200 -w 20 -G 50 -c 1 -d bfloat16
NCCL_DEBUG=INFO ./build/all_reduce_perf -b 1M -e 1M -g 4 2>&1 | grep " via "
```
(`NVCC_GENCODE` is a Makefile variable in nccl-tests' `src/Makefile` that I know of; the README fetched here did not mention it. Check that the pip package ships headers. If it does not, ship `nccl.h` from the matching NCCL release. The NCCL in the nccl-tests run must be the **same version** SGLang's torch loads, because NCCL versions have regressed before: [NVIDIA/nccl #1298](https://github.com/NVIDIA/nccl/issues/1298).)

**5. Attribute the share inside the real engine:** run one SGLang decode window under `nsys profile -t cuda,nvtx --cuda-graph-trace=node` (if nsys is in the image), or use the NCCL replay: number of all-reduces per step (≈128–139) × measured µs at 983,040 B, then compare with the 115–130 ms step. A cheaper A/B is to rerun the same EVAL slice with `NCCL_P2P_LEVEL=SYS` (if P2P OK) and compare the step time.

### Inferences
- [inference] Step 1 decides everything downstream. If `can_device_access_peer` is 0 or `topo -p2p r` says NS on Kaggle, P2P levers and every IPC-based kernel are off the table. The levers left are CE memcpy SHM, channels, the NCCL version, and reducing ranks/bytes per call.

### Gaps
- Not verified whether Kaggle's container allows `dmesg`, `lspci`, or `nsys`.
- Not verified that the `nvidia-nccl-cu13` wheel ships `include/nccl.h`.
