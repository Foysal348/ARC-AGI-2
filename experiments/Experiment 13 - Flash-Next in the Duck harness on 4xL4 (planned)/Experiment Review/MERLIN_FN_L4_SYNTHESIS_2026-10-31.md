# Merlin Flash-Next RC2 on Kaggle 4xL4: synthesis report

Date: 2026-10-31. Scope: read-only synthesis of four investigations (offline env, engine source, capacity and checkpoint, GPT-claim audit) and their adversarial verifiers. Nothing was pushed, run or edited.

Rules used in this report:
- Where a verifier disputes a finding with primary evidence, the verifier's corrected statement is used.
- Where neither side has primary evidence, the item is marked UNRESOLVED.
- GUESS marks an estimate that has not been measured.

Path legend (all local, Windows):
- **[MS]**: `D:\ARC AGI 2\_scratch\e23\p2\merlin-src`, the Merlin v3.0.0 checkout (HEAD ad42e37f, tag merlin-v3.0.0).
- **[FNP]**: `D:\ARC AGI 2\_scratch\fn_public\merlin_flash-next.json`, byte-identical (sha ef0d63cf) to `[MS]\tools\merlin\profiles\flash-next.json`.
- **[E23]**: `D:\ARC AGI 2\experiments\Experiment 23 - Six levers vLLM 0.32 (planned)`.
- **[E13]**: `D:\ARC AGI 2\experiments\Experiment 13 - Flash-Next in the Duck harness on 4xL4 (planned)`.
- **[E14R]**: `D:\ARC AGI 2\experiments\Experiment 14 - Optimized Engine on 4xL4 (planned)\Experiment Results\Kaggle Runs\E14A SLIM R2 run1 (hosen42, 2026-10-23)`.
- **[TREE]**: `D:\ARC AGI 2\_scratch\hf_tree_halt95_Qwen3.8-Flash-Next-W4A16-Merlin.json`.

---

## 1. DECISION: GO-WITH-GATES

Build the Merlin Flash-Next diagnostic notebook RC2. Pushing it still needs the user's explicit go. The environment side has been run on the same hardware:
- The Merlin v3.0.0 offline payload, the glibc 2.35 workaround (one byte set weak plus an LD_PRELOAD shim for the four `__isoc23_strto*` symbols), CPython 3.13, GCC 13, driver R580 and CUDA 13 were all installed and served correct text on Kaggle 4xL4 in E23 SMOKE_A (`[E23]\Experiment Results\smoke_rev3g\SMOKE_A\logs\install_merlin.log:256,265,271`; `SMOKE_REV3G_RESULTS_2026-10-30.md:11-15,25-28`).
- The Flash-Next code path (qwen4_exp, PLE offload, QSA, MTP, Marlin MoE) ships in the same wheel and tree.
- The PP fixes stock vLLM 0.32.0 lacks (#54709, #54793/#63617, #54795) are in Merlin.
- The checkpoint config passes Merlin's FP8-PLE check.

What remains is Flash-Next-specific and never measured on Ada:
- native E4M3 QSA KV on sm_89 (profile `fn-quad-24g-sm89` is "untested");
- TP2xPP2+EP without P2P;
- the PLE transport (cuStreamWaitValue32) inside the Kaggle container;
- MTP K3 plus RecoverSSM on generations of 60k tokens or more;
- the real KV pool at the L4-lowered pin;
- throughput.

These are gates, not blockers. RC2 is built to answer each one with a pass/fail arm.

Three things are mandatory in the build. Leaving any one out turns a working engine into a failed run:
1. Export `KV_BYTES` lowered for L4. `launch.build()` keeps the 4.1e9 pin otherwise.
2. Override `reasoning_effort` to xhigh. Merlin serves "medium" whatever the template.
3. Run Merlin's `host_phase` checks with the venv bin on PATH.

The checkpoint upload (29 Merlin-only files, 72.55 GB, plus a manifest) is a prerequisite and also needs the user's go. The zero-upload Swift1.5 route works as an alternative arm.

---

## 2. RISK REGISTER (after verification)

Ratings: BLOCKER = the run fails if this is not handled. HIGH = likely to cost the run or the result. MED = could cost a re-run or degrade results. LOW = minor.

### 2.1 Explicitly requested items

| # | Risk | Final rating | Final status |
|---|---|---|---|
| R1 | glibc 2.38 requirement vs Kaggle glibc 2.35 | **LOW** | SOLVED (verified on the Kaggle GPU) |
| R2 | sm_89 native E4M3 QSA KV correctness | **HIGH** | OPEN: never run on any Ada card; GPU gate |
| R3 | No-P2P TP2xPP2+EP performance (and whether P2P actually works) | **MED** | OPEN; P2P availability UNRESOLVED |
| R4 | PP + PLE + MTP state bugs | **MED** | Main upstream fixes included; the Merlin pre-copy design and the MTP-layer scale need a soak |
| R5 | MoE NaN recurrence (SGLang layer-2 NaN) | **MED** | UNKNOWN; probably does not carry over; GPU gate |
| R6 | Conversion accuracy for long reasoning | **HIGH** | OPEN: no BF16-matched run, no Merlin output above 768 tokens |
| R7 | Host RAM | **LOW-MED** | Likely fits; plan against the cgroup, not the host total |
| R8 | Startup / compile time | **MED** | UNMEASURED; GUESS 15-25 min cold |
| R9 | Checkpoint upload | **MED** | Plan ready (29 files); needs the user's go; Kaggle processing time unknown |

**R1 glibc 2.38: LOW, SOLVED.**
- Evidence: the Merlin docs ask for glibc 2.38 (`[MS]\README.md:226`, `[MS]\docs\merlin\ENGINE-README.md:24`, `[MS]\docs\merlin\hardware-matrix.md:229`). Only `_C_stable_libtorch.abi3.so` needs GLIBC_2.38, for `__isoc23_strtol/ll` (`D:\ARC AGI 2\_scratch\e23\p2\glibc_scan_wheel.json`). `vllm-rs` needs 2.39 but is off by default.
- The fix: `[E23]\prep\P2_merlin_env\glibc235_fix.py:1-13,99-103` (sets VER_FLG_WEAK and refuses any other symbol above 2.35), `glibc238_shim.c:1-4,36-39`, `install_merlin_offline.sh:62-66`.
- Controls: on the CPU kit, import fails without the shim and works with it (`D:\ARC AGI 2\_scratch\e23\p2_out7\p2_report.json`, steps import_noshim rc 1 / import_shim rc 0). On the Kaggle GPU: `install_merlin.log:256` ("weak flag set"), `:265` (ENVIRONMENT VERIFIED), `:271` (INSTALL OK).
- Small caveat from the verifier: the patch scan covers only `$TREE/vllm` .so files, not site-packages. The dependency wheels seen are manylinux_2_27/2_28. The cutlass-dsl and tilelang imports Flash-Next needs were not exercised by the 27B smoke (GUESS: low risk).
- Mitigation:
  - Reuse `foysalemonshanto/e23-merlin-env` (`merlin_offline_tar.bin`, 4,290,693,120 B, sha d854bc14) unchanged, with `LD_PRELOAD=` the shim and `VLLM_USE_RUST_FRONTEND=0`.
  - In the A0 preflight, import `cutlass` (nvidia-cutlass-dsl) and `tilelang` in the venv.
  - Pin the payload sha.

**R2 sm_89 native E4M3 QSA KV: HIGH, OPEN.**
- Evidence: `[FNP]` fn-quad-24g-sm89 has status "untested", evidence "None: no A30 run report exists" (a copy-paste stub), notes "path has not run on any Ada card", and acceptance "native E4M3 KV and kernel parity on Ada (needs hardware access)" (`[FNP]:830-945`).
- Code path on sm_89: `qsa_e4m3_uses_native_cast()` is true at capability 89 or above (`[MS]\vllm\models\qwen4_exp\nvidia\ops\qsa.py:27-36`). The fused Triton write path does bf16 round, then `/scale`, then `.to(fp8e4nv)` (`ops\qsa_prepare.py:184-190`). QK runs in bf16 and PV in fp16 (`ops\qsa.py:213-276`).
- Saturation of Triton fp32 to fp8e4nv on sm_89 is UNVERIFIED (GUESS: `cvt.rn.satfinite`).
- Scales are loaded strictly (fail closed) for 12 layers, 3, 7, ..., 47 (`nvidia\model.py:236-315,1145-1146,1365-1366`).
- GUESS: the MTP draft QSA layer gets no sidecar scale and runs at 1.0 (`mtp.py:277-283`; log "DEFAULT 1.0" at `qsa.py:502-513`). That would only affect draft acceptance.
- The clip counter (`COUNTERS=1`) turns off the fused writer (`qsa.py:706-712`), so it does not test the production path.
- Fallback cost: BF16 KV gives a pool of about 292k tokens instead of about 558k.
- Mitigation: arm A1 boots BF16 KV as the control. Arm A3 runs an FP8 vs BF16 A/B on identical prompts (teacher-forced and greedy). Check all logprobs are finite, detect constant-token loops, check the scales log lines, and grep for "DEFAULT 1.0". Optional bisection: `--kv-cache-dtype-skip-layers` (`qsa.py:559`, vllm#60023).

**R3 No-P2P TP2xPP2 performance: MED, OPEN; P2P availability UNRESOLVED.**
- The layout is not the main risk. TP2xPP2+EP 25/23 is Merlin's tested release-gate layout on 4x3090 with P2P (`[FNP]` fn-quad-3090 status "tested"). A no-P2P PP4 profile is tested on sm_80 (fn-sm80-64g-x4-pp4).
- The E23 SMOKE_B PP4 crash (illegal memory access at 8k) was stock vLLM plus RedHat INT4 in eager mode (`[E23]\Experiment Results\smoke_rev3g\SMOKE_REV3G_RESULTS_2026-10-30.md:62-66`). It is a weak predictor for the Merlin fork.
- The no-P2P TP2xPP2 rows (fn-quad-3090-nop2p) are untested even on 3090. On the v2.2.0 no-P2P 3090 run, decode was 127-141 tok/s against 168-184 with P2P (`[FNP]` fn-quad-3090-nop2p evidence), about −20 to −25 %.
- Kaggle L4 is PCIe gen 3 max x16 (`[E23]\...\SMOKE_A\p5_bench.json:49`).
- UNRESOLVED: the project context says "no P2P", but the E14 probe shows `can_access_peer` True for all pairs and NCCL "via P2P/IPC" under NCCL_P2P_LEVEL=SYS (`[E14R]\bundle\e14\e14_nccl_work\_probe\probe.json`). Whether P2P gives real bandwidth on GCP L4 is not established.
- In off mode the profile unsets NCCL_ALGO/PROTO and sets `disable_custom_all_reduce` (`[FNP]:903-915`; `[MS]\tools\merlin\launch.py:366-370`).
- Mitigation:
  - Correctness arms run P2P=off (the profile default).
  - Optional arm A7 reboots with P2P=on and repeats the throughput ladder.
  - Every PP2 arm gates on long prefill and decode (8k/32k/60k).
  - If a PP boot fails with StopIteration / "Worker failed", see R4.

**R4 PP + PLE + MTP state bugs: MED.**
- Included in Merlin:
  - #54709: placement-based PLE PP gate and the MTP `intermediate_tensors is None` check (`[MS]\vllm\model_executor\models\config.py:1027-1044`; `nvidia\mtp.py:354`). Stock 0.32.0 refuses PP>1 (`D:\ARC AGI 2\_scratch\vllm032_qwen4exp_src\vllm\models\qwen4_exp\nvidia\model_state.py:35-38`), so stock vLLM cannot run this layout.
  - #54793 and #63617: the planner filter at `[MS]\vllm\v1\core\kv_cache_utils.py:1659-1663`. I checked this myself: `_project_kv_cache_groups_to_worker` (`:2520-2560`) keeps empty projected groups, and the planner skips them through `if layer_name in group.layer_names` / `elif group.layer_names`. The GPT-audit remark "still has the unfiltered projection" is therefore consistent with the fix being present: RESOLVED.
  - #54795 (`[MS]\vllm\v1\attention\backends\utils.py:287-296`).
  - #60210, replaced by Merlin's persistent per-request-slot `ALIGN_BT_ROW_IS_REQ` (`[MS]\vllm\v1\worker\mamba_utils.py:38,480-487,555-563,1200-1224,1400-1457`).
  - The RecoverSSM align-mode boundary fix in 3.0.0 (`[MS]\docs\merlin\ENGINE-CHANGELOG.md:122`).
- Corrected framing (verifier, primary evidence via gh): upstream v0.32.0 already indexes both mamba align-copy kernels by request slot (via #60210), so #55506 has no remaining upstream content. Merlin departs from upstream: its pre-copy uses `batch_idx` against the current-step gather (`mamba_utils.py:680-701`; `gpu\model_states\mamba_hybrid.py:213-217`), and only its deferred post-process uses `req_idx`. GUESS: safe, because the gather and the pre-copy happen in the same step.
- Our layout (PP2 + MTP K3 + prefix caching + up to 8 seqs) matches the #55506 trigger conditions: 14-33 % constant-token loops from all-NaN logits (https://github.com/vllm-project/vllm/pull/55506).
- Mitigation: multi-stream long soak (arms A4/A5) with constant-loop detection, finite logprobs and MTP acceptance tracked over time. Never disable prefix caching while MTP is on (E19 rule; Merlin always enables it, `launch.py:461`).

**R5 MoE NaN recurrence: MED, UNKNOWN (probably does not carry over).**
- The SGLang NaN is deterministic on every forward, first at `model.layers.2.mlp.experts` (GPTQMarlinMoEMethod) with finite inputs (`[E13]\Experiment Results\E13_NANPROBE_rev2_GPU_run1_2026-10-30.md:19-27`).
- Correction (verifier, primary): the card's "layer 2" is 1-based `ple_layer_ids=[2]`, so the PLE block is in `model.layers.1`. The NaN is in the next layer's MoE, with finite inputs (`E13_NANPROBE_rev2...md:23-27,34-36`; `D:\ARC AGI 2\_scratch\vllm032_qwen4exp_src\vllm\models\qwen4_exp\nvidia\model.py:218-219`). A bad PLE input is already argued against.
- Merlin differs in nearly everything:
  - compressed-tensors checkpoint, CompressedTensorsWNA16MoEMethod, then the Marlin MoE backend (`fused_moe\oracle\int_wna16.py:125-143`);
  - INT8 GDN on dense Marlin W8A16;
  - EP2 inside each TP2 stage;
  - a PLE offload process with in-graph stream memory ops (`[MS]\vllm\model_executor\layers\ple_offload_layer.py:1-26,88-200`).
- Shared with SGLang: the experts derive from Intel AutoRound (HF card).
- Merlin does not carry sglang #47188's one-block-per-SM rule (`[MS]\csrc\moe\marlin_moe_wna16\ops.cu:262-277`). It hard-codes `MERLIN_FULL_K` (`ops.cu:40-62`; `marlin_template.h:382-383,546-553`), so `slice_count` is 1 and the split-K lock is skipped (source inference, verified by the verifier; not tested on hardware).
- Merlin scores GSM8K-200 199/200 on 3090 (`[MS]\models\flash-next\benchmarks\2026-10-20\BENCH-CARD.md:114`).
- Mitigation:
  - A1 smoke: finite logits, count, 2+2.
  - Grep the boot log for "Using 'MARLIN' WNA16 MoE backend".
  - Determinism probe: the same prompt at T=0 sent 8x alone and 8x inside a mixed batch.

**R6 Conversion accuracy for long reasoning: HIGH, OPEN.**
- Checkpoint (HF card; config at the pinned revision https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin/resolve/cec53e9f90e5995b81691604ec0b32191debf797/config.json):
  - INT4 g128 experts (Intel AutoRound) and INT4 MTP experts;
  - INT8 per-channel GDN in_proj/out_proj;
  - FP8 PLE;
  - FP8 E4M3 KV with static scales;
  - "no BF16 run of the same checks was made".
- Quality evidence is GSM8K (thinking off) and needles at 262K, thinking off, greedy, on a pre-release build (`BENCH-CARD.md 2026-10-20:94,114-116`). Published decode outputs are 128-512 tokens (Merlin) and 768 tokens (Jon-Nielsen, https://huggingface.co/Jon-Nielsen/Qwen3.8-Flash-Next-Merlin-3090-Serving-Kit).
- Jon-Nielsen reports FP8 KV MTP acceptance of about 3.0 vs 3.5 with BF16 KV. Merlin reports xhigh acceptance of 2.29 (prose) vs 2.66 at medium (`ENGINE-CHANGELOG.md:13-14`).
- Mitigation: A3 FP8/BF16 A/B; A4 60k-thinking stability; A6 ARC mini-eval at xhigh on tasks with known Qwen3.8 solve history (from the Puzzle Capability Ledger). Save the full thinking traces.

**R7 Host RAM: LOW-MED.**
- Requirement: one shared pinned FP8 PLE table of about 47.7 GiB, a 69 GiB measured floor and a 96 GB "qualified" allocation (`[FNP]:465,941`; `[MS]\tools\merlin\launch.py:67-70`; `[MS]\tools\merlin\preflight.py:54,476`). The checks are warn-only (`launch.py:1105-1162`). /dev/shm must be at least 1 GiB.
- Kaggle 4xL4 (from `[E14R]\bundle\e14\e14_results.json`):
  - MemTotal 197.9 GB;
  - cgroup limit 185.8 GB;
  - /dev/shm 91.8 GB;
  - nproc 48.
- Verifier correction (primary): vLLM's cgroup-aware "available RAM" read only 92.04 GiB (`[E23]\...\SMOKE_A\logs\server_merlin_tp4.log:69`), 99.13 GiB (`SMOKE_B\logs\server_fp8_mtp0.log:60`) and 78 GiB with 4 replicas running (`[MS]\vllm\model_executor\model_loader\weight_utils.py:933-951`). The margin over the 69 GiB floor is about 25-45 GiB, not about 130 GB.
- Counter-evidence that it fits: SGLang pinned about 95 GiB of BF16 PLE (23.84 GiB × 4) on Kaggle without failure (`E13_NANPROBE_rev2...md:36`).
- Mitigation:
  - Never stage the checkpoint into /dev/shm (it counts against the cgroup).
  - Read the checkpoint from /kaggle/input through symlinks.
  - Log `/sys/fs/cgroup/memory.max`, `memory.current`, MemAvailable, /dev/shm use and peak RSS every 10 s.

**R8 Startup / compile time: MED, UNMEASURED.**
- Measured for 27B on Kaggle (SMOKE_A):
  - payload extract 24.5 s plus install 175.6 s;
  - checkpoint prefetch 193.6 MB/s;
  - cold TP1 ready 384 s (torch.compile 54.6 s, FlashInfer JIT about 150 s, `server_merlin_tp1.log:71,88,113-114,129`);
  - TP4 204 s;
  - warm replicas 135 s.
- Flash-Next facts:
  - WARM=0, because the persisted Triton autotune is sm_86-only (`[FNP]` notes), so GDN kernels autotune live on the first boot and may pick slow configs (`[MS]\docs\merlin\hardware-matrix.md:216`);
  - PLE ready timeout 1800 s (`launch.py:383`);
  - reference cold start on 4x3090 is 268 s (`[MS]\models\flash-next\benchmarks\2026-10-09\BENCH-CARD.md:87`);
  - SGLang Flash-Next on Kaggle took 1,132 s (`[E13]\Experiment Results\E13_LOADPROBE_GPU_run1_2026-10-27.md:12`).
- GUESS: 15-25 min cold. That is the 123.8 GB read at 190-540 MB/s (4-11 min), plus 3-8 min of compile and JIT, plus about 1 min of PLE pinning. A second boot in the same session takes about 8-12 min.
- Mitigation:
  - Use shared `VLLM_CACHE_ROOT`, `FLASHINFER_WORKSPACE_BASE` and `TRITON_CACHE_DIR` across boots. E23's per-port TRITON_CACHE_DIR (`[E23]\Experiment Notebooks\Build\build_e23_notebook.py:277`) defeats reuse.
  - Raise `VLLM_PLE_OFFLOAD_READY_TIMEOUT` to 3600.
  - Use a 45-min ready timeout.
  - Tar the compile caches to /kaggle/working for a possible warm-cache dataset (GUESS: validate the cache keying first).

**R9 Checkpoint upload: MED.**
- Plan in §4.
- Risks: Kaggle processing time after a 72 GB upload is unknown, and 3-worker streaming speed is unmeasured (single file 32-34 MB/s: `[E11]\Experiment Results\Stage 1 runs (2026-10-15)\E11-01 run (both models created)\first_run_quota_error.log:45-46`, where E11 = `D:\ARC AGI 2\experiments\Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)`).
- Mitigation: the E3 streaming uploader (idempotent, sha256 while streaming, instance created only when all files match), with HF revision cec53e9f pinned. Fallback: 2 instances. Needs the user's go.

### 2.2 Other material risks

| # | Risk | Rating | Evidence | Mitigation |
|---|---|---|---|---|
| R10 | `KV_BYTES` not exported, so `launch.build()` pins 4.1e9 B on a 22.49 GiB card | **BLOCKER if omitted** (mitigated by design) | `launch.py:303` (default `srv['kv_cache_memory']`); `preflight.py:920-931,1225-1236,1262` ("the plain launcher needs that export too"); `serve_cli.py:714-715`; L4 = 23034 MiB (`preflight.py:178`; `[E23]\...\SMOKE_A\e23_results.json:9-28`) | Always export `KV_BYTES=2479838700` (FP8) or `2464415744` (BF16, 79 × 31,195,136). E23 precedent: `build_e23_notebook.py:270,620` |
| R11 | Reasoning effort silently "medium" | **HIGH** (mitigated by design) | `registry.py:409`; `launch.py:470-472` always passes `--default-chat-template-kwargs` with medium; froggeric v22.5 `:17,26,81-88`; checkpoint template accepts medium, so its xhigh default is never used (`[MS]\models\flash-next\templates\chat_template.jinja:47-49`). Verifier correction: `CHAT_TEMPLATE=` alone still serves medium | Trailing `--default-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"xhigh"}'` (last wins, `launch.py:110-111`; E23 rev 3h `build_e23_notebook.py:255-258`), plus top-level `"reasoning_effort":"xhigh"` in every request, plus an assertion that the rendered prompt contains the xhigh instruction |
| R12 | `max_num_seqs` 8 above the resident fit, so preemption at long contexts | **HIGH** | 149 usable blocks, 25 per 70k request: 5 fit (§3). 8 admitted allows an average of only about 47k | Client-side admission at 5 (FP8) / 3 (BF16) for no-preemption runs; ladder measures 6-8 with preemption monitoring. Raising seqs above 8 also needs capture sizes beyond 32 (`[FNP]:846-868`) |
| R13 | Merlin `host_phase` checks skipped by `build()` | **MED** | `launch.py:1194-1240` (products, config FP8 PLE, shadow, `_toolchain(full)`, topology, RAM/shm); E23 uses `build()` only | Put `VENV/bin` and the GCC13 bin on the base PATH **before** `ML.build`, then call `ML.host_phase(plan)`, then apply E23's env overrides; cwd must not contain `vllm/` (`launch.py:840-853,878-896`) |
| R14 | Wrong KV scales for the weights (base vs Swift) on the `--profile` path | **MED** | Scales are picked by profile/env, not weights (`launch.py:52,395-403`); config.json identical (64f990cd) in both checkpoints; base profile has no marker; Swift marker is config.json | Base = fn-quad-24g-sm89 + `SCALES=<tree>/models/flash-next/scales/qsa_kv_scales_262k.json` (sha 5cbe6ae8); Swift = fn-swift1.5-quad-24g-sm89 + `kv_scales-swift-e4m3.json` (sha d211be1f). Assert the sha; never create `qsa_kv_scales.json` in the merged dir |
| R15 | PLE transport (cuStreamWaitValue32/WriteValue32 on host-mapped memory) inside the Kaggle container | **MED, UNKNOWN** | `ple_offload_layer.py:88-279`; E23 27B has no PLE, so it is untested on Kaggle | Caught at the A1 boot; save the PLE child-process log; `VLLM_PLE_OFFLOAD_READY_TIMEOUT` 3600 |
| R16 | Coverage: throughput may not cover the hidden set at ≥ 60k thinking | **HIGH** (production, not RC2) | §3: GUESS 200-320 tok/s at 5 streams vs about 260-350 tok/s needed for 240 tasks at a 45-60k average decode | RC2 measures it; levers are P2P on, BLOCK_SIZE 2048, DRAFT_HEAD=int8 (−6.5-7.2 % ms/token, `[MS]\models\flash-next\CHANGELOG.md:10`), a larger KV pin, Swift1.5 (31-58 % fewer thinking tokens on BF16 benchmarks) |
| R17 | Live Triton autotune on sm_89 picks slow configs | **MED** | `hardware-matrix.md:216`; `qsa_tuning_capability()` is None on sm_89 (`ops\qsa.py:39-51`) | Ladder at steady state after warmup requests; keep the Triton cache for later boots |
| R18 | /kaggle/input read throughput for 123.8 GB (GCSFUSE/NFS) | **MED** | 27B prefetch 193.6 MB/s (`[E23]\...\SMOKE_A\e23_results.json` setup.prefetch_merlin); SGLang ready about 1,200 s | Parallel prefetch (`cat` to /dev/null, 8 readers) during the Merlin install; record MB/s |
| R19 | Toolchain env mistakes (CUDA_HOME 12.8, Rust frontend, NCCL leaks) | **LOW** | `launch.py:865-875` refuses a non-13 CUDA_HOME; `envs.py:180` | E23 merlin_server env: CUDA_HOME = venv nvidia/cu13, `VLLM_USE_RUST_FRONTEND=0`, start from a clean `os.environ` |
| R20 | sm_90-only ops (cooperative_topk, cute_dsl hc_down_silu) | **LOW** | `qsa_indexer.py:638-648` gate; `hyperconnection.py:40,112-116` runtime gate; `hc_down_silu.py:198-217` warmup gate; cutlass-dsl still imported | A0 imports cutlass-dsl; boot log check |
| R21 | Commit notebook size above 400 kB | **LOW** | Project memory (Kaggle platform limits) | Keep modules in the payload dataset; notebook < 400 kB |
| R22 | Swift1.5 licence (Swift Open License v1.0 §5 Commercial Use, US$1M revenue Legal Entity threshold) | **MED** (Swift route only) | https://huggingface.co/halt95/Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin LICENSE §5; base Merlin = Qwen Community License 1.0 (no concern, GUESS, not legal advice) | User to check the competition's external-model rules before a prize submission on Swift |
| R23 | Third-party dataset drift (yasserothman Swift copy) | **MED** (Swift route only) | 31 files, 72.55 GB, no local file list | Pin the version; hash all files against `[MS]\models\swift1.5\checkpoint.sha256` (40 entries) |
| R24 | thinking_token_budget misuse below the 60k floor | **LOW** | `protocol.py:265,780`; `config\reasoning.py:60-62` (ceiling only) | Use only as a ceiling of at least 120,000, or not at all; never below 60,000 |

---

## 3. CAPACITY & SPEED

**KV pin on L4** (Merlin's own rule, verified by both investigators):
- The L4 has 23,034 MiB (22.49 GiB), against the 24 GiB class.
- `kv_bytes_for_vram` gives 4.1e9 − 1,616,904,192 = 2,483,095,808.
- That rounds down to 150 blocks × 16,532,258 = **2,479,838,700 B (2.31 GiB)** (`preflight.py:920-933,1276-1279`; `[MS]\docs\merlin\hardware-matrix.md:28`).
- The same method ran with no OOM for 27B on Kaggle (E23 KV_TP4 16,301,686,784 → 1,829 blocks; logged pool 1,789,034).

**Pools** (formula floor((blocks−1) × 262,144 / 70), which reproduces 924,993 and 486,838 exactly):

| KV dtype | Blocks | Pool tokens | 70k req (25 blk) | 100k (32 blk) | 131k (39 blk) | 262k |
|---|---|---|---|---|---|---|
| FP8 E4M3 (2.31 GiB) | 150 (149 usable) | **557,992** (Merlin linear 559,471) | **5** (6 marginal) | 4 | 3 | 1-2 (UNRESOLVED: Merlin ~1, block rule 2) |
| BF16 fallback (2,464,415,744 B) | 79 (78 usable) | **292,103** | 3 | 2 | 2 | – |
| SGLang 0.5.21, measured | – | 161,984 | – | – | – | – |

Notes on the table:
- Per-request blocks = ceil(L/4096) + 7 (Merlin fit rule "71 per 262K, 39 per 131K", `[FNP]:1385`).
- A 10k prompt + 60k thinking + 2k answer ≈ 72k is still 25 blocks.
- Concurrency dispute resolved: the verifier's "6-7 at 70k" is a linear-ratio GUESS. The block rule is Merlin's own primary rule and gives 5. Use **5 resident at the ≥ 60k floor** (6 only with staggered admission). At 8 admitted, the average context must stay at or below about 47k.
- MTP draft KV is already inside the per-block bytes (binding block 16,515,072 = 7 × 2,359,296). Do not subtract it again.
- The cause of Merlin's larger pool vs SGLang is unverified. The shared host PLE saves host RAM, not VRAM (verifier correction).
- Sensitivity: each extra 0.25 GiB of pin adds about 16 blocks, about 0.6 more streams at 70k. At 2.50 GiB: 6 streams at 70k / 5 at 100k. At 2.75 GiB: 7 / 5 (computed).
- GUESS levers to measure: BLOCK_SIZE 2048 (`launch.py:334`), a lower image cap, DRAFT_HEAD=int8. Keep the 0.9 GiB minimum free on stage 0 (`[MS]\models\flash-next\docs\reference.md:178`).

**Speed anchors:**
- Kaggle 4xL4, Merlin 27B TP4 MTP4: 421.5 tok/s at 40k@8 and 329.0 at 120k@8, 0 preemptions, accept length 4.227 (`SMOKE_REV3G_RESULTS_2026-10-30.md:12-15`).
- 4x3090 Flash-Next: single stream 168-184, 638 at 8 agents (16K + 512 follow-ups, thinking off, 5.97 s window, `BENCH-CARD.md 2026-10-20:104-112`). Six 126,976-token sessions gave 574.5 aggregate (`reference.md:170`).
- The L4/3090 decode ratio is 0.43 (single stream, 120k vs 131K), the cleanest ratio, up to 0.56 (cross-method, use loosely). There is no valid prefill ratio yet (verifier correction).

**Projection (all GUESS):**
- Single stream: about 72-100 tok/s.
- 5-6 long streams: about **200-320 tok/s aggregate** (247-322 from the ratio, minus up to about 20 % if the P2P-off penalty is not already inside the ratio).
- 8 streams at moderate context: perhaps 300-400.
- Over 11.5 h (41,400 s): 8.3-13.2M decode tokens, about 140-220 requests if each decodes the full 60k (upper bound; many finish earlier). Prefill time for the prompts is extra.

**Production coverage gate** (from project memory: hidden = 240 tasks, 12 h limit):
- Required aggregate decode ≈ tasks × average decode / 41,400 s.
- At a 60k average that is 348 tok/s; at a 45k average, 261 tok/s.
- RC2 must report the measured tok/s and average thinking length against this line.

---

## 4. CHECKPOINT PLAN

**Revision:**
- Pin HF `halt95/Qwen3.8-Flash-Next-W4A16-Merlin` at `cec53e9f90e5995b81691604ec0b32191debf797`, as Merlin's card and `[MS]\models\flash-next\checkpoint.sha256` (38 entries) do.
- Current main d9d99c3f matches all 38 entries (0 mismatches; only README.md and .gitattributes changed).
- config.json at cec53e9f has `text_config.ple_embedding_dtype = "float8_e4m3fn"`, `ple_layer_ids [2]` and compressed-tensors W4A16 g128, so `host_phase` passes.

**Upload, option B (recommended, self-contained except the PLE): 29 files, 72,553,802,055 B (67.57 GiB)**, from [TREE]:
- 14 main shards (69,751,531,528 B):
  - `model-00001-of-00017.safetensors` 2,764,456,336
  - `00003` 5,162,391,536
  - `00004` 5,162,332,544
  - `00005` 5,218,067,368
  - `00006` 5,161,608,528
  - `00007` 5,162,349,440
  - `00008` 5,192,811,704
  - `00009` 5,161,510,288
  - `00010` 5,162,349,440
  - `00011` 5,115,155,880
  - `00012` 5,161,592,568
  - `00013` 5,162,349,168
  - `00014` 5,175,581,416
  - `00015` 4,988,975,312
- `model-00017-of-00017.safetensors` 1,284,526,608 (identical to Intel/overlay, oid fdbe0ca6; included to remove any dependency on where it lives on Kaggle).
- `model_extra_tensors.safetensors` 181,140,320
- `mtp-routed-experts-int4.safetensors` 1,298,193,176
- `model.safetensors.index.json` 25,460,994
- `config.json` 5,918
- `quantization_config.json` 94,026
- `tokenizer.json` 12,809,320
- `processor_config.json` 1,220
- `qsa_kv_scales_262k.json` 1,482
- `LICENSE` 3,220
- `README.md` 23,469
- `chat_template.jinja` 8,952
- `generation_config.json` 214
- `preprocessor_config.json` 443
- `tokenizer_config.json` 1,165
- Plus `KAGGLE_MIRROR_MANIFEST.json` (per-file sha256), so 30 files, under the 50-file cap.
- There is no model-00002 or 00016 (the PLE shards replace them); assets/ is README images only.
- Option A (24 files, 71,269,264,673 B) also reuses 00017 and 4 identical small files from the overlay. That saves 1.28 GB but depends on which Kaggle instance holds 00017. UNRESOLVED from local records, so B is preferred.

**Reused from our overlay** `foysalemonshanto/qwen38-flash-next-fp8ple-overlay`:
- Only the 10 `model-plefp8-0000{0..9}` shards, 51,200,267,901 B.
- Same LFS oid (dc2e845b..61de98b8) in the base, Swift and albucino repos. Full sha256 already verified on Kaggle in E3 (`D:\ARC AGI 2\experiments\Experiment 3 - Flash-Next Feasibility (FN)\Experiment Results\Kaggle Runs\FN_SMOKE_v3_run\fn_smoke\fn_smoke_report.json`, full_hash true, from_overlay 10).
- **Never** take config.json (266,471 vs 5,918), the index (23,657,880 vs 25,460,994), tokenizer.json (19,989,325 vs 12,809,320) or LICENSE from the overlay. Do not reuse `E13_OVERLAY_SHARDS` main-shard sizes (`[E13]\Experiment Notebooks\Build\build_e13_duck.py` around line 1016): they are Intel sizes.

**Method:**
- A CPU Kaggle notebook with internet ON, using the E3 streaming uploader (`...\FN_FP8PLE_build_R1\notebooks\FN-PLEFP8-UPLOAD.ipynb`): HF to Kaggle resumable blob upload, RAM only, 3 workers, 256 MiB chunks, sha256 while streaming, instance created only when every file matches.
- No .bin wrapping is needed (no .zip/.tar, no `[ ] = , & @ +` in names).
- One model instance version (suggested GUESS name `foysalemonshanto/qwen38-flash-next-w4a16-merlin`, instance `merlin-base-nople`, v1).
- Fallback if resume fails: 2 instances, about 33.8 GB (00001, 00003-00008) and about 38.7 GB (00009-00015 plus the rest).
- Estimated transfer: 13-36 min (upper end measured at 34 MB/s; lower end GUESS), plus unknown Kaggle processing (GUESS 30-90 min).
- **Needs the user's explicit go.**

**Runtime merge (RC2 A0):**
1. Symlink the 29 Merlin files plus the 10 overlay PLE shards into `/tmp/fn_merlin` (about 2 s, E3 merge_sec 1.9).
2. Asserts:
   - the file set ⊇ every file in the index weight_map;
   - every size equals [TREE];
   - small files match `checkpoint.sha256`;
   - config FP8-PLE field is set;
   - `SCALES` sha = 5cbe6ae8;
   - no `qsa_kv_scales.json` present;
   - no `vllm/` in cwd.
3. Full sha256 of all 39 entries (124 GB at 0.29-0.65 GB/s = 190-430 s, measured in E3), run in parallel with the Merlin install (175 s) and the page-cache prefetch. Merlin itself never hashes shards (`launch.py:1373`).

**Zero-upload alternative, Swift1.5:**
- Sources:
  - Kaggle `yasserothman/swift15-flash-next-merlin` (31 files, 72.55 GB);
  - our overlay PLE;
  - the Swift scales file from the engine tree (`[MS]\models\swift1.5\scales\kv_scales-swift-e4m3.json`, d211be1f);
  - profile `fn-swift1.5-quad-24g-sm89`.
- On HF the non-PLE Swift files are 33 files, 72,553,829,482 B, which matches the Kaggle size. So the 2 missing files total at most about 8.8 MB, and the index and tokenizer are present (GUESS: the missing ones are .gitattributes and README.md or SHA256SUMS).
- Hash everything against `[MS]\models\swift1.5\checkpoint.sha256` (112-250 s).
- Trade-offs:
  - (+) no upload and no wait;
  - (+) 31-58 % fewer thinking tokens with near-equal hard-benchmark scores at BF16 (UkisAI, Swift card), a possible coverage lever;
  - (−) different weights: attention q/k/v/o, GDN and shared experts replaced, so the ARC effect is unmeasured;
  - (−) FP8 KV only, with no BF16 profile, so the R2 fallback is a GUESS-level override;
  - (−) evidence carried over from v2.2.0;
  - (−) licence §5 (R22);
  - (−) third-party dataset dependency (R23);
  - (−) file list unknown until the user's Kaggle API listing.
- A hybrid for base (only the 14 main shards uploaded, 69.75 GB, the rest from yasserothman) is possible but adds the same third-party dependency.

---

## 5. RC2 NOTEBOOK DESIGN

**Modes:**
- `RC2_SHORT`: arms A0-A3 plus a 1/4-stream ladder; hard stop 100 min. A 30-min smoke cannot fit, since the cold boot alone is GUESS 15-25 min.
- `RC2_FULL`: A0-A6 (+ A7 optional); hard stop 4 h.
- Every arm has its own timeout, and a failed gate skips dependent arms but still writes all results (fail closed, keep going where independent).

**Launch path (reuse E23):**
- `merlin_install`: extract `merlin_offline_tar.bin`, run `install_merlin_offline.sh` (glibc fix, rehash, verify, GCC13, uv CPython 3.13.13, wheelhouse), and assert the payload sha d854bc14.
- `merlin_server`:
  1. Base env from a clean `os.environ`: `PATH=$VENV/bin:/tmp/merlin_gcc13/bin:...`, `CUDA_HOME=$VENV/.../nvidia/cu13`, `CC/CXX/CUDAHOSTCXX`=GCC13, its libstdc++ first on `LD_LIBRARY_PATH`, `LD_PRELOAD=libglibc238_shim.so`, `VLLM_USE_RUST_FRONTEND=0`, PYTHONPATH stripped.
  2. Merlin knobs: `KV_BYTES` (FP8 2479838700 / BF16 2464415744); `SCALES=<tree>/models/flash-next/scales/qsa_kv_scales_262k.json` (FP8 only; unset for BF16); `CHAT_TEMPLATE=` empty (checkpoint template); `WARM=0`; P2P mode = profile default off.
  3. `plan = ML.build(profile="fn-quad-24g-sm89" [BF16: --kv bf16 variant], ckpt="/tmp/fn_merlin", ...)`, then `ML.host_phase(plan)`.
  4. Apply E23 overrides: shared `VLLM_CACHE_ROOT`, `FLASHINFER_WORKSPACE_BASE`, `TRITON_CACHE_DIR` (not per port), `VLLM_PLE_OFFLOAD_READY_TIMEOUT=3600`.
  5. Extra args, appended last (last wins):
     - `--default-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"xhigh"}'`
     - `--max-model-len 131072`
     - `--max-num-seqs 8` (keep the captured graph range; concurrency is enforced client-side)
     - keep MTP K3, prefix caching and `--use-replayssm` as the profile sets them.
  6. Start from a cwd without `vllm/`; ready timeout 45 min (cold), 25 min (warm).
- Every request: top-level `"reasoning_effort":"xhigh"`, `logprobs` / `top_logprobs` 5 where needed. Never cap thinking below 60,000; ARC arms use `max_tokens` 120,000 (memory rule: 120k band, max-model-len 131072).

**Outputs to save** (per arm, under `/kaggle/working/rc2/`):
- full request/response JSON with the **full thinking trace of every request**;
- rendered prompts (sample per arm);
- server log and per-rank worker logs plus the PLE child-process log;
- boot-phase timings (extract, install, prefetch MB/s, hash, load, compile, JIT, graph capture, PLE pin, ready);
- `/metrics` scrape every 5 s (running, waiting, preemptions, KV usage, spec-decode acceptance);
- `nvidia-smi` dmon (power, memory, util);
- host memory every 10 s (cgroup memory.max/current, MemAvailable, /dev/shm, RSS);
- the logged "GPU KV cache size" / pool per boot;
- logprob arrays for A/B;
- `rc2_results.json` with every gate verdict;
- a tarball of the Triton, FlashInfer and vLLM compile caches.

**Arms (ordered):**

**A0 Env preflight and checkpoint merge** (no server; budget 10 min, sha256 overlapped).
- Record: uname, glibc version, `nvidia-smi` (driver at least R580, CUDA Version 13.x, 4 × 23034 MiB, PCIe gen), torch `can_access_peer` matrix, cgroup memory.max, MemAvailable, /dev/shm (≥ 1 GiB), /tmp free, nproc.
- Run the Merlin install.
- Venv imports: vllm (with the shim), `cutlass` (cutlass-dsl), `tilelang`, `flashinfer`, `triton`, `transformers` 5.18.0 with qwen4_exp.
- Run the merge and the asserts from §4.
- PASS: all checks; install OK; import OK; sha256 all match (commit mode); host_phase dry run raises no refusal. FAIL stops the run.

**A1 BF16-KV control boot and smoke** (cold boot ≤ 35 min; smoke 3 min).
- Boot log must show:
  - no "requires every PLE layer" error;
  - "Using 'MARLIN' WNA16 MoE backend";
  - the Marlin mixed-precision kernel for INT8 GDN;
  - PLE offload ready;
  - pool ≈ 292k (accept 250-330k);
  - no OOM;
  - cgroup memory below 90 % of the limit.
- Probes at T=0: count "1..10"; "2+2=" → 4; "7*6=" → 42; "9-3=" → 6. Rendered prompt contains the xhigh sentence.
- One 2k-prompt / 512-token generation with logprobs.
- PASS:
  - all probes correct;
  - every logprob finite;
  - no constant-token loop (no single token ≥ 64× in a row, no 8-gram repeating ≥ 16×);
  - the text is not "!!!!" garbage.
- FAIL here means the R5 path (MoE/PLE). Save everything, skip to A3 only if the failure is BF16-specific.

**A2 BF16 references and long prefill** (15 min).
- Fixed prompt set: 4k, 8k, 32k and 60k-token prompts (ARC task text plus a needle), greedy, 1,024 tokens out, `top_logprobs` 5.
- Plus 2 ARC prompts, greedy, 4,096 out.
- Store the token ids for A3 teacher forcing.
- PASS: finite logprobs, needle retrieved at 32k and 60k, no loop, PP2 prefill stable (no illegal memory access, the E23 SMOKE_B symptom).

**A3 FP8-KV boot and A/B on identical prompts** (warm boot ≤ 25 min; A/B 15 min).
- Boot log must show:
  - "applied static K/V scales ... N of 12" with N summing to 12 across the 2 PP stages;
  - "DEFAULT 1.0" only, if anywhere, on the MTP draft layer (record it);
  - pool ≈ 558k (accept 500-600k).
- Re-run the A1 probes.
- Re-run the A2 prompts greedy and compare. Teacher-force the BF16 continuations through `prompt_logprobs`.
- Determinism probe: one prompt ×8 alone vs ×8 in a mixed batch.
- PASS (thresholds GUESS, E23 precedent rank-1 agreement 0.9883):
  - teacher-forced rank-1 agreement ≥ 0.97 per prompt;
  - mean |Δlogprob| ≤ 0.05 nat;
  - greedy shared prefix ≥ 256 tokens, or divergence at a near-tie (top-2 margin < 0.1);
  - needle retrieved;
  - all finite;
  - determinism divergences only at near-ties.
- FAIL means production uses BF16 KV (pool 292k, 3 streams). The remaining arms then run on the BF16 boot (one more reboot).

**A4 Long-generation stability, 8k/32k/60k** (on the winning KV dtype; 30 min).
- (a) 2 concurrent hard ARC prompts at xhigh with natural stopping, `max_tokens` 120k.
- (b) 2 concurrent synthetic stress requests with `min_tokens` 60,000 (labelled synthetic), with checkpoints at 8k, 32k and 60k generated tokens.
- Record MTP acceptance per 1k-token window, finish_reason, logprob finiteness over the whole trace, and loop detection.
- PASS:
  - every request reaches 60k (stress) or a natural stop;
  - all logprobs finite;
  - no constant loop;
  - MTP acceptance stays ≥ 2.0 (xhigh reference 2.29) with no collapse after 8k block boundaries (RecoverSSM);
  - no preemption at 2 streams;
  - no NCCL watchdog.

**A5 MTP-on throughput ladder at 1/2/4/8 streams with real long generations** (40 min; runs while A4 traces finish if time is short).
- Each level: ARC prompts at xhigh, the 120k ceiling, warmup, then a 5-min steady window.
- Assert the real running count equals the level (memory rule: verify real concurrency).
- Record aggregate and per-stream tok/s, accept length, KV usage and preemptions.
- For level 8, record the context at which preemption begins.
- PASS: real concurrency matches; 0 preemptions at ≤ 5 (FP8) / ≤ 3 (BF16); numbers recorded.
- GO line for a production build (the §3 coverage gate): measured aggregate at 5 streams vs 261-348 tok/s. If it falls short, the report names the next lever (A7 P2P on, a larger pin, BLOCK_SIZE 2048, Swift).

**A6 ARC mini-eval** (60-75 min, FULL only).
- 8 EVAL tasks (about 12 test outputs), half with known Qwen3.8 solves in the Puzzle Capability Ledger and half hard.
- xhigh, ≥ 60k thinking allowed, `max_tokens` 120k, 5 resident streams, the production prompt and parser, 2 attempts as in production.
- PASS (informative):
  - ≥ 90 % of answers parse to valid grids;
  - no garbage or loops;
  - average thinking length and tokens per task recorded;
  - solves compared with the ledger (no hard threshold; this is first quality evidence).

**A7 (optional) P2P-on reboot** (25 min).
- Profile interconnect `on` (unset NCCL_P2P_DISABLE / VLLM_SKIP_P2P_CHECK per the profile mode).
- Re-run the A1 probes plus the 4 and 8 stream ladder levels.
- PASS: correct probes, and tok/s delta reported against A5.

**Time budget (FULL):**

| Arm | Budget |
|---|---|
| A0 | 10 min |
| A1 | 38 min |
| A2 | 15 min |
| A3 | 40 min |
| A4 | 30 min |
| A5 | 40 min |
| A6 | 70 min |
| A7 | 25 min |
| Total | about 4 h 30 min worst case |

The hard stop is 4 h; A7 is dropped first, then the second A6 wave. SHORT: A0 + A1 + A2 (reduced) + A3 + a 1/4-stream ladder ≈ 100 min.

---

## 6. OPEN QUESTIONS ONLY A GPU RUN CAN ANSWER

1. Does native E4M3 QSA KV (Triton `fp8e4nv` cast, KV_DTYPE 3) give token and logprob parity with BF16 KV on sm_89? Does the cast saturate or produce NaN for |x| > 448·scale?
2. The real KV pool at `KV_BYTES` 2,479,838,700 (FP8) and 2,464,415,744 (BF16), and whether the 2.75 GiB overhead assumption holds on L4 with graphs captured up to 32. Minimum free memory per PP stage.
3. Does the Merlin PLE transport (cuStreamWaitValue32/WriteValue32 on host-mapped memory, offload child process) work inside the Kaggle container, and within the ready timeout on the slow /kaggle/input mount?
4. Does TP2xPP2+EP on Merlin stay free of the PP illegal-memory-access crash (#63505 class) on long prefill at 8k/32k/60k?
5. Does Merlin's Marlin MoE on the same Intel AutoRound INT4 experts give finite layer-2 output on L4? That would localize the SGLang NaN to the SGLang runtime.
6. Is the Merlin pre-copy design (batch_idx with the current gather) safe under PP2 + MTP + prefix caching at 5-8 streams (no #55506-style constant loops)?
7. MTP K3 acceptance and RecoverSSM behaviour across generations of 60k tokens or more at xhigh, FP8 vs BF16 KV. Does the MTP draft QSA layer run at the default scale 1.0?
8. Real aggregate decode tok/s at 1/2/4/5/8 streams with long thinking, P2P off vs on. Does P2P actually work and help on Kaggle L4 (`can_access_peer` True in E14)?
9. Cold and warm startup time: /kaggle/input read MB/s for 124 GB, torch.compile for 2 PP stages, FlashInfer/Triton JIT and live autotune with WARM=0, and whether a cache tarball can make reruns warm.
10. Host memory accounting during load and serving: PLE pin + page cache + /dev/shm transport against the 185.8 GB cgroup, plus peak RSS.
11. Real thinking length and quality on ARC tasks at xhigh for this quantized checkpoint, against the coverage gate (261-348 tok/s for 240 tasks).
12. Whether the capacity levers (larger pin within 0.9 GiB free, BLOCK_SIZE 2048, a smaller image cap, DRAFT_HEAD=int8) free real KV and speed on Flash-Next.
13. Swift1.5 only: the exact yasserothman file list and version (needs a Kaggle API listing), and whether FP8-only KV with Swift scales passes the same A/B.