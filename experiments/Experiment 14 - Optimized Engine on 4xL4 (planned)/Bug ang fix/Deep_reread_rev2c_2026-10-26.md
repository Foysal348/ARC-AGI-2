# Deep re-read of the GPT pre-GPU audit (rev 2c), 2026-10-26

**Sources re-read:**
- `ARC_AGI2_LLM_Serving_PreGPU_Audit_BN.pdf`, all 20 pages. Its extracted text is the same document as `ARC_AGI2_LLM_Serving_PreGPU_Audit_README.md`; it has no extra claims.
- `README_QWEN38_SERVING_PRE_GPU_AUDIT_2026-10-25.md`.

Every item below is mapped to the current notebooks: EMS (SGLang) and VMS (vLLM).

## The audit's FINAL three priorities

| Priority | Status |
|---|---|
| 1a. #45151 / #44145 backport status and pool sizing | **#44145 applies to our stack.** Proof in our SGLang 0.5.21 source: `scheduler.py:1049-1075` sizes the target KV pool (`init_target_memory_pool`) before the draft's `alloc_memory_pool`, and `eagle_worker_v2.py:226-241` binds the target embed / lm_head only there. The MTP draft's duplicate BF16 embed + lm_head (248,320 x 5,120 each, ~4.74 GiB, ~1.19 GiB/GPU at TP4) is resident while KV is sized. I read PR #45151 (open, unmerged) in full (`_scratch/upstream_prs/pr45151.diff`): 2 hunks in `EagleDraftWorker`. **rev 2c:** an env-gated patch of the installed file (`EMS_PR45151=1`; inert otherwise; applied only when both anchors match exactly once, else recorded) plus the measured arm `mtp_3_1_4_pr45151` right after `mtp_3_1_4`. The best-MTP picks for 20 streams and TP2 carry the arm's env. DFlash is not affected (it is not `EagleDraftWorker`) |
| 1b. #43029 status | Merged 2026-10-10, after 0.5.21. It fixes the mamba track-boundary check when a **grammar** truncates the accepted speculative run. EMS and production use no grammar-constrained decoding, so **not applicable**. The oracle gate would show any state divergence |
| 2. DFlash cross-request / thinking-state correctness (#36548, #38009) | rev 2 / 2b: target-only oracle; serial + all-at-once greedy at 256 tokens; a 1,024-token probe at 12 concurrent with thinking enabled; a verdict vs the oracle and vs the DFlash reference rows |
| 3. Mamba effective admission (#36889) | `capped_to` parsed from the log; the windows run at the real level; a CAPPED flag (8k and 40k bands); slots-aware sizing (4 lazy / 5 extra_buffer, checked in source); NTN arms sized 85 |

## Gate A (CPU / static)

| Item | Status |
|---|---|
| Package versions / wheel hashes | The wheelhouses carry `requirements.lock` + `SHA256SUMS`; the E14 facts cell records the environment |
| `--help` flag check | Done from source instead (`--help` fails on CPU for vLLM: "Failed to infer device type"); agents 2 and 4 rebuilt every argv against the 0.5.21 / v0.32.0 arg definitions |
| MTP BF16 vs packed (vLLM) | `vms_int4_mtp_static()`, vLLM's exact ignore matching: RedHat passes (`re:^mtp.*`, 15 dense keys); the bad-ignore case skips every INT4 arm |
| #43029 / #45151 applied commit | Checked in our installed-version source: neither is present; #45151 is now an opt-in arm; #43029 is N/A |
| `VLLM_TEST_FORCE_FP8_MARLIN` exists? | Not in v0.32.0 `envs.py`; arm dropped; `--linear-backend triton` is the only real W8A8 arm (auto = Marlin W8A16 on SM 8.9) |
| `--no-enable-prefix-caching` spelling | Verified in v0.32.0 `arg_utils.py:1369-1375` |

## Gate B (safety acceptance)

| Test | Status |
|---|---|
| Same-checkpoint temp=0 target vs spec | Both notebooks: the oracle gate (token ids + top-1/top-2 margins; large-margin flips vs the oracle's A/A) |
| Serial and concurrent | Yes (6 serial + 12 all at once) |
| Prefix cache hot / cold | Pass b of each A/A repeats pass a's prompts, so it is a warm-cache run against a cold one. The VMS oracle has prefix caching off; the 40k band shares a 36k prefix under load; there is a degenerate-output count |
| Effective concurrency | Real running level, cap flags (above) |
| Server restart x3 | **Not done**: about 3 extra launches per arm, out of budget. The A/A covers same-server stability; each arm is a fresh launch compared with the oracle |
| Thinking on / off | **Thinking on only.** Production thinks; the off control is out of scope for a speed matrix |
| Context 64k-155k | **Partly**: the bands are ~6-8k and ~40k prompts plus up to 8k generated. The >64k regime is covered by production runs (E15B / E15P), not here |
| Streaming vs assembled | The VMS windows stream; the EMS windows use the E14 generators; the production shim already streams (E15P: 0 dropped tokens) |
| Spec partial rejection | Implicit in every greedy pass; no forced-reject harness |

## Gate D (kernel / communication)

- `nvidia-smi topo -m` and `-p2p r` are recorded in EMS (TP2 pair choice) and VMS.
- `NCCL_DEBUG=INFO` (INIT) is set on every TP2 server.
- The nccl-tests microbenchmark is not run here. E14A already measured it: P2P/IPC 1,880 us vs NCCL 344 us. That is why custom all-reduce is off for TP2.

## High-value alternative arms (audit list) not added

These are not added because of the 200-minute budget; they are candidates for the next matrix:
- target-only at 12 / 20 / 24 streams;
- extra_buffer_lazy vs extra_buffer vs no radix cache;
- chunked prefill 2048 vs 4096;
- `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE` (agent 2);
- `--language-model-only` (agent 4);
- `--kv-cache-dtype auto` in vLLM.
