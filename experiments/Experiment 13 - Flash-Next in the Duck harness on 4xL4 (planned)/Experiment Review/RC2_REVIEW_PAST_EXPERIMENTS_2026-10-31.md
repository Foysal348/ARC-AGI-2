# E13 RC2 review: what to add or change, from past runs (2026-10-31)

**What this is:** an internal, read-only review of all past experiments, done before the RC2 build. It is kept here as the input to RC2. `RC2_BUILD_2026-10-31.md` records how RC2 handled each item.

**What was read:**
- the project memory;
- E13 (the synthesis; the RC1/RC1b, NANPROBE, LOADPROBE and SMOKE1 reports; the RC1 raw outputs);
- E23 (`e23_lab.py`, `build_e23_notebook.py`, the SMOKE rev3g results and fix list, the rev3h audit brief, DESIGN.md, the P2 install log);
- the E14 NCCL/P2P probe;
- the E18/E19/E21 results;
- E22 ENGINE_FINDINGS;
- the E3 FN_SMOKE_v3 report;
- the Merlin v3.0.0 source (`launch.py`, `registry.py`, the PLE offload code, the flash-next profile);
- the Merlin FN checkpoint part-1 output.

**Path shorthand:**

| Tag | File |
|---|---|
| [RC1LOG] | the RC1 vLLM server log |
| [LAB] | `e23_lab.py` |
| [B23] | `build_e23_notebook.py` |
| [ML] | Merlin `launch.py` |
| [FNP] | Merlin's `flash-next.json` |

**The six most important fixes:**
1. **Fix the host-RAM guard** (A4). It counts page cache as used memory, and that is what skipped the third RC1 arm.
2. **Require near-empty GPUs before each boot, and clean up between boots** (A5, A6). Merlin leaves under 1 GiB of slack.
3. **Send one sampled request right after the first boot** (A7). The `-lcuda` sampler failure is what killed the RC1 stream test.
4. **Run teacher-forcing last** (A8). It is likely to OOM the server.
5. **Calibrate FP8-vs-BF16 against an A/A baseline** (A10). An absolute 0.05 nat is below the drift a layout change alone produces.
6. **Fix the 8-stream ladder point** (A11). As planned, it will preempt.

## A. MUST ADD (otherwise a crash, a wasted run or a wrong verdict)

**A1. Inputs.** E23 CELL1/CELL2 have must-find inputs:
- the 27B FP8 model ([B23]:104);
- KITS and WHEELHOUSE (:113-114);
- `KERNEL_KIT_HASH_FAILED` (:129-137);
- the 27B Merlin checkpoint (:124-128).

Change:
- write RC2-specific input discovery;
- keep the wheelhouse only for the stock fallback;
- render prompts from `/tmp/fn_merlin/tokenizer.json` (sha256 0997f410…, byte-identical to the 27B Merlin tokenizer) and `chat_template.jinja` (sha256 c3cf9e34…);
- assert both hashes at runtime.

**A2. Call `ML.build` for Flash-Next.**
- E23 calls `ML.build("27b", …)` ([B23]:271). RC2 needs `build("flash-next", "fn-quad-24g-sm89", "/tmp/fn_merlin", base, extra=…, kv=None|"bf16", status="untested")`. The BF16 variant is selected with `kv="bf16"` (`KV_TAG`, `variant_profile`) and carries its own seqs (3).
- `build()` resets three values that RC2 must re-apply AFTER build:
  - `VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800` ([ML]:383), which RC2 raises to 3600;
  - `TRITON_CACHE_DIR`, set only with WARM=1 (:413);
  - `PYTORCH_CUDA_ALLOC_CONF`, removed (FN_UNSET :55-61). Do not put RC1's expandable_segments back.
- Keep `--served-model-name q27` in `extra`. The profile serves "flash-next-v2 flash-next …", while e23_lab defaults to `model="q27"` ([LAB]:316, 404). Without it every request returns 404.

**A3. Boot-OOM retry on KV_BYTES.**
- Merlin's profile note: "KV_BYTES is a per-card pin estimate … adjust in 0.25 GiB steps (lower it if the boot OOMs)".
- The margin is thin:
  - CUDA reports 22.03 GiB in total ([RC1LOG]:136; E14 `probe.json`), against nvidia-smi's 23,034 MiB;
  - RC1's load hit "expandable_segments … OOM … free: 3211264" on all ranks ([RC1LOG]:56-62, 89-108).
- Change: on an OOM at startup, retry once with KV_BYTES − 268,435,456, and record the pool for each attempt.

**A4. Host-RAM guard.**
- RC1 skipped arm 3 with cgroup free 8.9 GB while MemAvailable was 55 GB.
- vLLM saw "Available RAM: 0.24 GiB" and "177152 MiB current usage … under 177152 MiB limit" ([RC1LOG]:48-49, 67, 121). The cgroup was full of page cache.
- Change: use `memory.max − (memory.current − inactive_file)` ([ML]:1123-1131) plus MemAvailable. Log anon, file, shmem and inactive_file. Never block an arm on `max − current`.

**A5. Near-empty GPUs before every Merlin boot.**
- `wait_gpu_release` accepts up to 1024 MiB ([LAB]:191).
- After RC1's dead vLLM server the GPUs held 561/515/515/515 MiB.
- The Merlin pin leaves about 0.9 GiB free.
- Change: use about 100-150 MiB for Merlin boots.

**A6. Graceful shutdown, then cleanup.**
- e23 `Server.kill` waits 40 s, then SIGKILLs ([LAB]:113-125), while the profile passes `--shutdown-timeout 60` ([ML]:473).
- The PLE transport unlinks its `/dev/shm` storages only on a graceful teardown (`vllm/v1/ple_offload/connector.py`:39-99, 362-363).
- RC1 ended with "1 leaked shared_memory objects", and `/dev/shm` counts against the cgroup.
- Change: wait ≥ 75 s after SIGTERM. Between boots, list `/dev/shm`, delete stale `torch_*` / `psm_*` / vllm files, and record the cgroup memory before and after.

**A7. Sampled path right after ready.**
- RC1 died at the first temperature-0.7 request: `ld: cannot find -lcuda` in the FlashInfer sampling JIT ([RC1LOG]:198, 274). That JIT takes about 100 s.
- Change: in A1, send one short sampled request (T 0.6, top_k 20) and check the engine is alive afterwards. Keep the E23 SHIM ([B23]:168-173, 275) on both Merlin and stock.

**A8. Teacher-forcing last.**
- prompt_logprobs materialises chunk × vocab logits: 1024 × 248,320 ≈ 1.0 GB in fp32. That lands on the last PP-stage rank, which also holds MTP and lm_head, with about 0.9 GiB slack.
- E23 recorded "prompt logprobs = chunk x 248k-vocab logits: OOM at 0.90" ([B23]:569).
- Change: run TF after the probes, greedy rows and ladder; log free memory per rank before it; label a death during TF as `TF_OOM`.
- GUESS: `CHUNK=256` on a boot if TF is required there. It would have to be the same for BF16 and FP8.

**A9. Explicit sampling.** Merlin passes `--override-generation-config` with temperature 1.0 ([ML]:51, 467). Every request must carry temperature, top_p and top_k (production: 0.6 / 0.95 / 20).

**A10. A/A-calibrated FP8-vs-BF16.**
- The planned thresholds (0.05 nat, rank-1 0.97) are too strict:
  - E23 TP4 vs TP1 alone gave mean 0.0772, p99 1.0, rank-1 0.9883;
  - NVFP4 gave 0.3787 / 0.9648.
- Change:
  - run TF twice on the BF16 boot, once on FP8;
  - report FP8 relative to the A/A and the E23 scale;
  - use the rev3h `compare_tf` ([LAB]:533-557) and `compare()`'s NEAR_TIE margin of 0.5 ([LAB]:37).

**A11. The 8-stream point at 40k preempts.**
- With 149 FP8 blocks, 8 streams get 18 blocks each = (18 − 7) × 4096 = 45,056 tokens.
- Change: run 8 streams on 8k-16k prompts, or label the point CAPACITY_LIMITED.

**A12. Measured prefill.** `steady()` assumes 700 tok/s ([LAB]:576). Time the first token of the 40k row in A1 and pass `prefill_tok_s`.

**A13. Natural vs forced long rows.**
- `Stream` defaults to `ignore_eos=True` ([LAB]:316, 320). Text past the EOS is out of distribution, which can produce false garbage or false MTP-acceptance FAILs.
- Change: one stream with `ignore_eos=False` on a real single-task xhigh ARC prompt, max_tokens 120,000. Label the forced rows synthetic.

**A14. Save every request.**
- `steady()` keeps only a 200-character tail ([LAB]:723-725).
- Change: append every `Stream.row()` (text, ids, finish_reason, margins, sampling, prompt sha) to a JSONL. For chat requests, read `message["reasoning"]`.

**A15. Stock fallback.**
- RC1's 30,208-token BF16 pool cannot take `--max-model-len 131072`.
- Change: use fp8_e4m3 KV and a max-model-len the pool supports, labelled diagnostic-only. Keep the SHIM and `--language-model-only`. If MTP is added, prefix caching must stay ON (E19).

## B. SHOULD ADD

**B1. Census at 90k and 116k:**
- FP8 90k: 149/30 → 4 streams;
- 120k: 37 blocks → 4 (Merlin's 131k figure says 3);
- measure @3/@4 and check whether @5 comes out CAPACITY_LIMITED.

**B2. Real running counts.** Keep the STEADY rule ([LAB]:707-713) and read the server caps. GUESS: PP2 bubbles at 1-2 streams.

**B3. Startup phase timings.** Parse "Loading weights took", "Model loading took X GiB and Y s", "torch.compile took", "Graph capturing finished", "init engine … took", the PLE-ready lines and "GPU KV cache size". Also time extract, install, prefetch and hashing. vLLM skips its own auto-prefetch above 90 % of RAM ([RC1LOG]:48-49).

**B4. Compile and JIT cache tarball**, saved after the first good boot:
- write it as `.bin`;
- `/kaggle/working` is only 20 GB;
- use one shared TRITON_CACHE_DIR.

GUESS: the FP8 boot may not reuse the BF16 boot's compile cache.

**B5. Host-RAM telemetry every 10 s:** memory.current/max, memory.stat, MemAvailable, `/dev/shm`, RSS, and `ulimit -l` (E3: 8,388,608 KB). Merlin's floor is about 69 GiB ([ML]:67-70) of 173 GiB.

**B6. Proof of the xhigh template:** the render assert ([B23]:398-406), `template_probe` ([LAB]:485-499), and one chat request with no per-request kwargs. Merlin's served default is medium ([ML]:470-472).

**B7. MTP acceptance over long rows:** the ACC/DRF/DTOK counters every 10 s, mapped to 1k windows, with marks every 4096 tokens. Reference values: the Merlin changelog gives xhigh 2.29 / medium 2.66; E23 Merlin 27B MTP4 measured 4.2.

**B8. P2P: low priority.** E22: P2P is 5.5× slower than NCCL SHM, and the PCIe IPC all-reduce is 0.68×. Record only the peer matrix, topo, PCIe generation and the NCCL transport.

**B9. Prefix-cache hits in align mode.** The E19 continuation protocol: `cached_tokens` and the counters. E18 got 7.7 % hits, E21 54-68 %.

**B10. Traces.** Covered by A14; also the per-arm prompts, worker and PLE child logs, and `nvidia-smi dmon`.

**B11. FP8 boot first?** GUESS: FP8 is the production candidate, and the comparison only needs both rows at the end.

**B12. Multimodal memory.** GUESS: `--limit-mm-per-prompt '{"image":0,"video":0}'` or `--language-model-only` may free encoder memory. Make it its own arm only.

**B13. Boot-log checks:**
- "Using 'MARLIN' WNA16 MoE backend";
- "applied static K/V scales … N of 12";
- "DEFAULT 1.0" (MTP);
- PLE ready;
- the cudagraph mode actually used (on Merlin 27B, FULL_AND_PIECEWISE was downgraded to PIECEWISE under spec decode).

**B14. CPU threads for the PLE lookups.** GUESS: record OMP_NUM_THREADS and the PLE child's CPU %.

## C. Already covered by the planned design

- The glibc 2.35 shim, `VLLM_USE_RUST_FRONTEND=0`, GCC 13 and cu13 CUDA_HOME from E23.
- The L4 KV_BYTES pins.
- The trailing xhigh kwargs, where the last flag wins (rev3h).
- `--max-model-len 131072`, appended last.
- MTP K3 + prefix caching always on (the E19 rule holds).
- The sha256-verified merge and SCALES. Check that no `vllm/` directory is in the cwd.
- The -lcuda SHIM.
- `compare_tf` with the top-1-id metric.
- Greedy oracle rows and the arithmetic probes.
- The STEADY windows, GPU-release gate, hard-stop watchdog, fail-closed step records and `save()`.
- The 60k rows; the ready timeouts and PLE timeout.
- The rerun fallback submission.

## D. Conflicts with project rules

- **D1. Size limit < 400 kB.** RC1 was at 398 kB. Assert the size in the builder.
- **D2. MTP with prefix caching OFF is forbidden.** RC1's stock flags had `--no-enable-prefix-caching`, which is fine only with MTP off.
- **D3. Thinking 120k.** Measure at 120k, not only 90k: add the B1 census.
- **D4. Time budget.** GUESS, about 105-135 min in total:

  | Phase | Estimate |
  |---|---|
  | A0 | 8-10 min |
  | A1 cold boot | 20-30 min |
  | A1 probes | 10 min |
  | Cleanup | 3 min |
  | A3 boot | 12-20 min |
  | A3 probes and compare | 10 min |
  | Ladder | 20-25 min |
  | Long rows | 12-17 min |
  | Census | 10 min |

  Set realistic est_min values.
- **D5. Pre-run capacity check.** Show the blocks per arm, the prefill tokens, the long-row tokens and the admission caps.
- **D6. Process.** Use a new notebook name; RC1 e702c098 stays in PUSHED; a Kaggle CPU kit after one local pass; push only on the user's go; report shas.
- **D7. No token cuts below 60k.** No conflict: oracle rows and probes are not solver budgets.
