# Rev 2d: XL context band + FlashInfer PCIe-IPC all-reduce (user 2026-10-26: "১ ও ২ নাও" ("take 1 and 2"))

## 1. XL band (both notebooks)
- **Why:** the windows reached ~48k context at most. Production thinks to 60k on top of a 5-10k prompt, so decode spends most of its time at 60-100k, where KV reads grow and DFlash / MTP accept and step time can change. R2 never measured width 6 even at 40k.
- **What:** a shared ~85k-token block of other eval tasks + the task (`E14_XL_TOKENS`, default 85,000), run as a third 75-s window on every single-server / TP4 arm.
  - Streams = `kv_tokens // 100k`, capped by the arm's level. About 8 on SGLang INT4 (~0.9M KV) and about 6 on vLLM FP8.
  - EMS flushes the radix cache before the XL window.
  - Not run on 2 x TP2 / DP2 (half the KV per engine).
  - vLLM `max_model_len` is 106,496, so 85k + 8k generated fits.
- **Report:** XL tok/s, streams, ctx per request, x reference, accept, step ms.

## 2. FlashInfer PCIe-IPC all-reduce
- **Why:** NCCL all-reduce is 50.9 ms of a 116-ms step (44 %, R2 profiler). TP4 custom all-reduce is off on 4 PCIe GPUs.
- **SGLang 0.5.21** has `srt/distributed/device_communicators/pcie_ipc_ar.py`:
  - switch `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE` (environ.py:1354);
  - dispatched after custom all-reduce and before NCCL (parallel_state.py:973-978);
  - bf16 decode reductions only, with the workspace sized `max_bs x draft tokens x hidden`;
  - tuning skipped under `--disable-flashinfer-autotune` (seed policy).
  - It needs `flashinfer.comm.PcieIpcAllReduceWorkspace`. Checked in the wheels: **not in FlashInfer 0.6.18 (our SGLang venv), present in 0.7.0.post1** (`flashinfer/comm/pcie_ipc_ar.py`, JIT-built CUDA-IPC kernels, world sizes 2/4/8, no SM restriction).
  - Its dependencies vs the SGLang 0.5.21 lock:
    - apache-tvm-ffi 0.1.11 ok;
    - nccl4py 0.6.0 ok;
    - cudnn-frontend 1.31 ok;
    - nccl-extensions 0.1.0 new (taken from the vLLM wheelhouse).
- **EMS:**
  - At EMS start, flashinfer-python 0.7.0.post1 + flashinfer-jit-cache 0.7.0.post1+cu130 (the '+' restored) + nccl-extensions 0.1.0 are installed with `pip --no-deps --target /tmp/ems-fi07` from **the vLLM 0.32.0 wheelhouse dataset, which must be attached to the EMS notebook**. The import is verified with the venv python.
  - The PCIe arms get `PYTHONPATH=/tmp/ems-fi07`, `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1` and `FLASHINFER_DISABLE_VERSION_CHECK=1`. Every other arm keeps the shipped 0.6.18.
  - Arms:
    - `dflash_w6_pcie`, first after the oracle, 30-min launch;
    - `mtp_3_1_4_pcie`.
  - The record keeps the log proof: "FlashInfer PCIe-IPC all-reduce enabled" / "workspace built" / "has no pcie_ipc_comm".
  - Risk: SGLang 0.5.21 was built against FlashInfer 0.6.18. An API break makes the arm fail at startup (recorded, the run goes on).
- **VMS:** vLLM 0.32.0 has it natively, `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC` (cuda_communicator.py:73-75, 140-148, 389-391; FlashInfer 0.7.0.post1 is already in the wheelhouse).
  - Arm: `fp8_mtp3_pcie`, first after the oracle.
  - Proof: "Initialized FlashInfer PCIe IPC all-reduce" in the log.

## 3. Budget
200 → 260 min for both notebooks: wall ≈ 4.5 h, quota ≈ 9 h each.

## 4. Correction found while building rev 2d (rev 2d-b)
The rev 2b EMS report lines and the try / except around the EMS body never applied: a silent `str.replace` miss. Now applied with asserted replacements and checked by the harness. See the A1-F13 row of the rev 2b disposition.
