# E28 K100 notebook: how it is built and run, and where a direct solver can plug in

**Scope.** I read the whole code export (`e28k100_code_export.py`, 5,070 lines) and the whole builder. I also read these in full: `e28k100_lib.py`, `build_e28_k100_cpu_kit.py`, `E28_SCHED_API.md`, `e28k100_build_facts.json` and `e28k100_kit_facts.json`.
- Supporting code read in part:
  - Duck's request path: `tool_agent.py` sha f7454201, `openai_compat.py` sha fea0a750, `arc2_duck.py` sha b9f67010. Both vendor shas match the manifest.
  - `nb5_lib.py` sha 18efcbc1: the functions the notebook calls.
  - `k100load_lab.py` sha ef393fbb: Server, watchdog and Stream.
- **Not read in full:** `e28_sched_patch.py` (I used its API doc) and `e28k100_unit_tests.py`.
- From the run output I read only config, engine and aggregate fields. I opened no thinking text.
- File paths:
  - Export (E): `D:\ARC AGI 2\experiments\Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)\Experiment Notebooks\Build\e28k100_code_export.py`
  - Builder (B): `...\Build\build_e28_k100.py`
- Line numbers below are E:line or B:line.

**Top finding for the 1-h design (capacity).** In the real run the server averaged 23.4 running at 419.5 tok/s. That is about 18 tok/s per stream.
- A 1-h commit leaves about 36 min of solve window, because boot takes about 15 min (see item 2).
- At 24 streams that is about 39k tokens per stream, so no request reaches the 60k thinking floor.
- Earlier K3/RC2 measurements give about 29–33 tok/s per stream at 8–16 streams. At that rate one stream can do about 63–72k tokens in 36 min. A 90k request needs about 45–52 min and a 120k request about 61–69 min.
- So the 1-h profile needs both fewer streams (about 8) and a longer window, or separate band phases, to cover the 90k / 120k bands.

---

## 1. Cell layout
The notebook has 9 cells (index 0 is markdown). The build asserts the kinds at B:83 and that cell 1 is E18 R2's CELL 0 at B:593.

| nb idx | Export header | Export lines | What it does |
|---|---|---|---|
| 0 | markdown MD0 | B:569-586 | Header. |
| 1 | **CELL 0**, mode + fallback submission (stdlib only) | E:1-46 | Verbatim E18 R2 CELL 0. Sets `NOTEBOOK_T0` (anchored to the kernel process start, at most 600 s back). `RERUN` = `KAGGLE_IS_COMPETITION_RERUN`. `ARC_ROOT` = `/kaggle/input/competitions/arc-prize-2026-arc-agi-2` (fallback `/kaggle/input/arc-prize-2026-arc-agi-2`). `OUT=/kaggle/working/q27x`. Loads `arc-agi_test_challenges.json` and writes the identity submission (below). |
| 2 | **CELL 1**, NB5 configuration | E:49-535 | Commit-only deadman, profiles, deadlines, engine and Duck parameters, payload loaders, `nb5_lib`, `k100q_report`, the E28 packed modules, Q-defer lib, CONFIG. |
| 3 | **CELL 3**, Duck package + ARC-2 adapter | E:538-986 | Duck zip from the payload dataset, sha manifest, E28 scheduler apply (K6), `import arc2_duck as E12A`, E21 finish_reason restore, sandbox memory limit and self-test. Then the "NB5 commit-turn variant C, request trace, admission" block (E:715-986): per-request /tokenize guard, tracer, `_nb5_session_run` reservation, scheduler edits N1/N2/N4. |
| 4 | (no CELL number) "E10: original task-order and evaluation profiles" | E:989-1006 | Loads the task set (item 6). |
| 5 | **CELL 5**, NB5 engine: Merlin Flash-Next | E:1009-3658 | All engine code as functions: Merlin install, klee100 merge, patches, server wrapper, gates, probes, capacity, `nb5_phase`, `nb5_boot_try`, `nb5_boot`. Mostly definitions; module-level effects: telemetry thread (E:1176), libcuda shim (E:1102), lab load from the dataset (E:1052-1061). |
| 6 | **CELL 6/7**, the Duck run and its report as functions | E:3660-4813 | `def e21_cell6()` (Duck solve) and `def e21_cell7()` (score and bundle). In the notebook, `e21_cell7` is a sha-pinned lzma+b85 literal that is exec'd (B:455-469; 33,048 B source, sha 3be03c1b). The export holds the readable text. |
| 7 | **CELL 8**, boots and phases | E:4816-4894 | Runs everything: inputs, merge, install, sha gate, #60490, `k100_setup`, I8 static check, fixed window, `nb5_boot` per (MTP, KV) group, final kill, `nb5_report.json`. |
| 8 | **CELL 9**, NB5 report | E:4897-5070 | Throughput, gpu-only gates, quality_ab, `e28_k100_report.json`, `Q27X_NB5_BUNDLE.zip`. |

**CELL 0 code (E:33-43):**
```python
def write_json_atomic(path, obj):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)
def identity_submission(challenges):
    return {tid: [{"attempt_1": q["input"], "attempt_2": q["input"]} for q in task["test"]]
            for tid, task in challenges.items()}
write_json_atomic(SUBMISSION_PATH, identity_submission(test_challenges))
```

**Commit-only vs rerun behaviour:**
- **Commit only:**
  - Deadman thread (E:110-117).
  - sha256 gate (E:4844-4848; rerun skips it, "E18 R2 / E15B SUBMIT have none").
  - These are fatal in the commit and only recorded in the rerun: wrong content-probe answer (E:2493-2496), contract probe (E:3022), C-11 config mismatch (E:3485), k100 gates (E:3493), Q-defer unverified tree (E:1760), NO_GO_E28_SCHED (E:614).
  - Window shift after the preflight (E:4155).
  - Cache diagnostic (disabled: `E12_R6_CACHE_DIAG: 0`).
  - `k100_finish`, flags and markdown in CELL 9 (E:5041).
- **Rerun only:**
  - `PROFILE="RERUN_PROD"` (E:120). Source is test; the E18 ladder applies (at most 3 tries, ready limit 30/15/15 min, E:3600).
  - `with_servers` uses `kill(fast=True)` after the solve (E:2208-2212).
  - `e12_write_submission` overlays the model's answers (E:3718-3723).
  - The E28 report status is `SKIPPED_RERUN`.
  - The probe prompts use the test file (E:1996).

---

## 2. Engine boot path, end to end

**Inputs (kernel-metadata, build facts):**
- Datasets: `foysalemonshanto/q27-e25-nb0-tf-ref`, `foysalemonshanto/q27-e15b-payload`.
- Competition: `arc-prize-2026-arc-agi-2`.
- Models:
  - `foysalemonshanto/qwen38-flash-next-klee100-3bpw-mtp/PyTorch/autoround-3bpw-mtp-nople/1`
  - `foysalemonshanto/qwen38-flash-next-fp8ple-overlay/PyTorch/albucino-fp8ple/1`
  - `foysalemonshanto/e23-merlin-env/Other/v1/1`
- Docker image `gcr.io/kaggle-private-byod/python@sha256:37c64f7d…`, machine `NvidiaL4`, no internet, private, id `esfersami50/q27-xhigh-e28-k100-eval120-6h`.

**Mount paths (nb5_rc2_inputs, E:1971-2021):**
- K100_DIR = `INP/models/foysalemonshanto/qwen38-flash-next-klee100-3bpw-mtp/pytorch/autoround-3bpw-mtp-nople/1` (else a glob for `*/*/*/KAGGLE_MIRROR_MANIFEST.json`).
- PLE_DIR = `…/qwen38-flash-next-fp8ple-overlay/pytorch/albucino-fp8ple/1`.
- MERLIN_PAY = `…/e23-merlin-env/other/v1/1/merlin_offline_tar.bin`.
- REF_DIR = `INP/datasets/foysalemonshanto/q27-e25-nb0-tf-ref` (needs `nb0_reference.json`).
- INTEL and WHEELHOUSE were not attached ("None" in the run).
- GPU preflight: exactly 4 GPUs, else `GPU_PREFLIGHT_FAILED` (E:2015).
- `L.start_watchdog(HARD_STOP_EPOCH)`, where `HARD_STOP_EPOCH = T_START + TOTAL_BUDGET_SEC` (E:1018). At that time every live server is killed.

**Payloads (CELL 1):**
- `k100q_payload()` (E:456-469): `k100q_payload_emb.json`, sha fd92a814. Holds the blobs `k100kit_core.py`, `k100load_lib.py`, `klee100_shard_digests.json`, `nb1_worker_ext.py`, `nb5_lib.py`, `tables/klee100.json`, `tables/merlin.json`.
- `k100q_ds_text()` (E:472-480) loads `k100q_report.py` (sha 359629ec) and `nb4_ppack.py` (sha 8dadf076) from the same dataset dir.
- The lab: `k100load_lab.py`, sha ef393fbb, loaded as `e23_lab` (E:1052-1061).
- The Duck zip: `q27x_e15b_duck_9e4b89285136.bin` (E:546).

**Merge (`nb5_rc2_prompts` → `merge_ckpt`, klee100 version E:1602-1643):**
- `k100_overlay()` (E:1506-1547) regenerates the fixkeys overlay with the kit core and requires byte equality with the v2 artifact. It also checks the config and index shas, the tables, and the PLE headers.
- `KL.build_model_dir(CKPT=/tmp/fn_klee100, …)` builds a dir of symlinks.
- `KL.k6_check` is the K6 check.
- A background thread streams sha256 over 23 files / 101.83 GB with 8 threads (136.2 s in the run).
- The tokenizer/template come from CKPT. Then it asserts the pinned TOK_SHA `06b95093…` and TPL_SHA `c3cf9e34…`, and that the xhigh render differs from medium (E:2105-2112).

**`merlin_install` (E:1359-1396):**
- Untar the payload into `/tmp/merlin_pay`, then run `bash install_merlin_offline.sh` into `/tmp/merlin_off` (timeout 5,400 s).
- Parse `MERLIN_TREE / VENV / SHIM / CC / CXX`, then `import launch as ML` from `TREE/tools/merlin`.
- Run: extract 36.1 s, install 171.7 s, rc 0.

**Patches and setup (CELL 8, E:4842-4866):**
1. sha gate (commit only).
2. `nb5_apply_60490()` (E:2687-2729): a strict unified-diff applier on `vllm/models/qwen4_exp/nvidia/qsa.py`, pre a23d5f90 → post 88fe0a8f, with marker, py_compile and vllm-origin checks. A failure sets the pin start to 0 (E:3576).
3. `k100_setup()` (E:1684-1711), in this order:
   - OVERLAY_MERGE_K6 gate (raises);
   - `k100_worker_ext()` copies `nb1_worker_ext.py` into the tree (py_compile, origin);
   - Q-defer `k100_qdefer_apply()` with nb4_ppack rev 2 on `pp_utils.py`, `model_runner.py`, `gpu_worker.py` (snapshot, restore on failure, env `NB4_PPDEFER=1`);
   - `k100_qd_gate()`;
   - `k100_scales()`: the QSA static scales sidecar (sha 5cbe6ae8), from the tree first, then the dataset, copied into CKPT (raises if missing).
4. I8: `e28_i8_static()` (E:1934-1950). An `UNSUPPORTED_STATIC` verdict drops I8 before the first boot.

**Server construction:** `merlin_fn_server` (RC2 version E:1409-1465, wrapped by the K100 version E:1865-1886).
- Before every launch: K6 check, `VLLM_QSA_KV_SCALES`, `NB4_PPDEFER` only on a verified patch, and a `serve --help=all` flag check (`FAIL_CLI_FLAGS`).
- Code (E:1413-1431):
  ```python
  extra = ["--served-model-name", "q27", "--max-model-len", str(MAX_LEN), "--port", str(port),
           "--default-chat-template-kwargs", XHIGH] + list(NB5_BOOT_EXTRA)
  base.update(GPUS="0,1,2,3", PORT=str(port), KV_BYTES=str(kv_bytes), P2P="off", VENV=..., TREE=..., CHAT_TEMPLATE="", WARM="0", CACHE_ROOT=f"{TMP}/merlin-cache", PATH=..., CUDA_HOME=...)
  base.update(NB5_BASE_ENV); base.update(NB5_BOOT_LEVERS)
  if kv == "fp8": base["SCALES"] = str(CKPT / "qsa_kv_scales_262k.json")
  plan = eng["ML"].build("flash-next", MERLIN_PROFILE, str(CKPT), base, extra=extra, kv=kv, checkpoint="flash-next", status="untested")
  eng["ML"].host_phase(plan)
  ```
  - `XHIGH = json.dumps({"enable_thinking": True, "reasoning_effort": "xhigh"})` (E:1405); `MERLIN_PROFILE = "fn-quad-24g-sm89"`.
  - `NB5_BOOT_EXTRA = base_extra + ["--max-num-seqs", str(seqs)]` (E:3594).
  - `base_extra = NB5_EXTRA_ARGS` (MTP on), made of:
    - `--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,3,4,6,8,12,16,24,32,40,48,64,80,84]}` (E:290-292);
    - `--language-model-only --worker-extension-cls nb1_worker_ext.NB1WorkerExt` (E:176, 293).
- **The real argv tail, last 60 elements (nb5_report `argv_tail`; vLLM argparse: the last value wins):**
  ```
  --enable-expert-parallel --disable-custom-all-reduce
  --speculative-config {"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic","draft_lm_head_quant":"int8"}
  --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 1024 --gpu-memory-utilization 0.96
  --kv-cache-memory-bytes 7933526016 --kv-cache-dtype fp8_e4m3 --limit-mm-per-prompt {"image":42,"video":0} --mm-processor-cache-gb 0
  --mm-processor-kwargs {...} --compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1..32]}
  --enable-prefix-caching --mamba-cache-mode align --enable-prompt-tokens-details --no-enable-flashinfer-autotune
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
  --override-generation-config {"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}
  --default-chat-template-kwargs {"enable_thinking": true, "reasoning_effort": "medium"} --shutdown-timeout 60 --block-size 4096
  --use-replayssm --prefix-match-unit 64 --served-model-name q27 --max-model-len 131072 --port 8001
  --default-chat-template-kwargs {"enable_thinking": true, "reasoning_effort": "xhigh"}
  --compilation-config {..."cudagraph_capture_sizes":[1,2,3,4,6,8,12,16,24,32,40,48,64,80,84]}
  --language-model-only --worker-extension-cls nb1_worker_ext.NB1WorkerExt --max-num-seqs 24
  ```
  - Effective values: max-model-len 131072, max-num-seqs 24, chat kwargs xhigh, the K100 captures.
  - TP2 × PP2 is in the argv head. `NB5_RC2_FP8_ARGV_TAIL` (E:2616) begins `"127.0.0.1","--port","8001","--tensor-parallel-size","2","--pipeline-parallel-size","2"`. The PP partition env is `VLLM_PP_LAYER_PARTITION=25,23`.
  - Because only the last 60 elements are recorded, `diff_vs_rc2` shows TP/PP as "removed". That is an artifact of the cut.
  - Merlin's `plan.argv()` head (the binary, `serve`, the model path) is not in our code.
- **Server env:**
  - Code (E:1440-1458):
    ```python
    env.update(LD_PRELOAD=kvv["MERLIN_SHIM"], CUDA_HOME=..., VLLM_USE_RUST_FRONTEND="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               LIBRARY_PATH=f"{SHIM}:...", PATH=..., VLLM_CACHE_ROOT=f"{TMP}/merlin-cache", TRITON_CACHE_DIR=..., FLASHINFER_WORKSPACE_BASE=...,
               CUDA_DEVICE_ORDER="PCI_BUS_ID", VLLM_PLE_OFFLOAD_READY_TIMEOUT="3600")
    # gcc13 CC/CXX/CUDAHOSTCXX + LD_LIBRARY_PATH; MASTER_PORT/VLLM_PORT from ML.distributed_ports; then NB5_BASE_ENV, NB5_BOOT_LEVERS
    ```
  - Recorded env in the run:
    - `NCCL_P2P_DISABLE=1`, `P2P=off`, `KV_BYTES=7933526016`, `VLLM_PP_LAYER_PARTITION=25,23`;
    - `VLLM_QSA_KV_SCALES=/tmp/fn_klee100/qsa_kv_scales_262k.json`, `VLLM_SERVER_DEV_MODE=1`, `VLLM_USE_V2_MODEL_RUNNER=1`, `VLLM_SKIP_P2P_CHECK=1`;
    - `VLLM_HOST_EMBED_TABLE=1`, `VLLM_PLE_OFFLOAD_HOME_DEVICE=0`, `VLLM_E2_*` (several), `VLLM_PORT=30100`;
    - levers `DRAFT_HEAD=int8`, `MERLIN_SHARE_DRAFT_WORKSPACE=1`, and `NB4_PPDEFER=1`.
- **Q-defer and I8 mechanics:** Q-defer is the env `NB4_PPDEFER=1` on the patched tree. I8 is `DRAFT_HEAD=int8` passed into `build()`, which turns into `"draft_lm_head_quant":"int8"` in `--speculative-config`.

**Readiness (`MServer.wait_ready`, E:1220-1240):**
- Ready means `/health` answers AND the server log contains `"GPU KV cache size"`.
- A log silent for `E18_SILENCE_SEC=600` s counts as a stall.
- Timeout `NB6_READY_SEC`: 30 min for every try in E28EVAL120; in RERUN_PROD 30 min for the first try, then 15/15 (E:3600).
- `with_servers` (E:2173-2222): start, wait, run the body, always kill. Then a GPU-release gate; if the GPUs stay held, ABORT.

**Boot ladder and fallbacks (`nb5_boot`, E:3562-3649):**
- KV pin ladder, highest first: `NB5L.pin_tries(2,479,838,700, start, steps=K100_PIN_STEPS_GIB)`. Steps in GiB per card: 5.08 / 4.08 / 2.60 / 1.0 / 0.0, i.e. 7,933,526,016 / 6,859,784,192 / 5,270,142,976 / 3,553,580,524 / 2,479,838,700 bytes.
- Launch seqs: 24 on the top rung (E:3592-3593), then `K100_SEQS_TABLE` 18 / 14 / 9 / 6.
- I8 drop: a try that fails with INT8 (or reads back as `INCONCLUSIVE_CONFIG_MISMATCH` / `I8_READBACK_UNVERIFIED`) is retried at the same pin with the BF16 head (E:3610-3621).
- Lever drop: when an error line names a lever (E:3624-3628), and the "no lever" fallback (E:3629-3633).
- No further try when less than 25 min is left (E:3587).
- There is no stock or Intel fallback for klee100: any setup miss raises, nothing boots, and CELL 0's identity submission stays.

**Gates in a boot try (`nb5_boot_try`, E:3446-3559), in order:**
1. `boot_gate`: GPUs below 150 MiB used, waits up to 300 s.
2. `nb5_host_gate`: page-cache drop, then a RAM floor of 69 + 4 GiB.
3. Launch.
4. C-11 `config_readback`: dtype, seqs, kv bytes, spec. Run: PASS.
5. I8 read-back plus `e28_i8_gate`. Run: `I8_ENGAGED`, INT8 head 2 × 307.9 MiB on PP1_TP0/TP1.
6. Q-defer read-back, recorded. Run: `ENGAGED_FALLBACK` 0.8333 at boot, `ENGAGED` 0.9944 at the end.
7. `k100_boot_gates`: method map via `/collective_rpc k100_method_map` → `GPU_METHOD_MAP_PASS`; static scales `STATIC_SCALES_APPLIED`; partition [25, 23]. Run: both PASS.
8. `nb5_probes` (E:3002-3033), all passed in the run:
   - `kv_census`;
   - `L.chat_probes`;
   - `sampled_check` (64 tokens at T 0.6);
   - `e21_content_probes`: 3 greedy prompts, thinking off;
   - `e21_contract_probe`: tools + reasoning low, turn-2 history growth, cache probe;
   - `nb5_commit_probe` (c_usable True, render parity 558 = 558);
   - prefill probe: 40k prompt, TTFT 8.8 s = 4,534 tok/s.
9. `nb5_capacity`.
10. `nb5_warmup`: 1×2k, 24×8k, 2×40k, 24×2k; 78.7 s.
11. No pure-decode window (`no_pure_decode`).
12. Phases.

**Boot time in the real run (nb5_report / engine json):**
- Boot try started at notebook +252 s.
- Server "ready in 485s" from launch; ready at notebook +769 s.
- Phase started at +877 s; solve window started at +879 s (14.65 min).
- Window end +22,479 s (actual 21,599.8 s); solver returned +22,481 s.
- Notebook at CELL 9: 374.9 min (22,492.7 s).
- One try only (`merlin_fp8_mtp_pin508_i8`), no lever drops. The deadman (415 min) did not fire.

---

## 3. How Duck talks to the server

**Endpoint:** `POST {base_url}/chat/completions` with `base_url = SGL_URL + "/v1"` = `http://127.0.0.1:8001/v1`. The model id comes from `/v1/models` (E:4108), fallback `"q27"`. Requests are non-streaming.

**Duck request code (`tool_agent.py:5226-5348`):**
```python
payload = build_chat_payload(provider=self._model.provider, model=self._model.model_id, messages=messages,
    max_tokens=self._max_output_tokens if max_tokens is None else max_tokens,
    temperature=_LOCAL_ANALYZER_TEMPERATURE, top_p=_LOCAL_ANALYZER_TOP_P, top_k=_LOCAL_ANALYZER_TOP_K,
    thinking=(bool(_LOCAL_ANALYZER_ENABLE_THINKING) if enable_thinking is None else enable_thinking),
    tools=tools, tool_choice=_request_tool_choice(tools), seed=_LOCAL_ANALYZER_SEED)
template_kwargs = dict(payload.get("chat_template_kwargs") or {}); template_kwargs.update(self._harness_template_kwargs())
...
requests.post(f"{base_url}/chat/completions", json=payload, timeout=(_CONNECT_TIMEOUT_SECONDS, request_timeout_seconds or self._timeout), stream=bool(payload.get("stream")))
# retries ARC3_HTTP_RETRIES=3, base 5 s; first-request grace ARC3_HTTP_RETRY_INITIAL_SECONDS=900 (DUCK_ENV)
choice = choices[0]; return _ChatCompletionResult(message=_extract_leaked_tool_calls(choice["message"]), finish_reason=str(choice.get("finish_reason","")), usage=payload.get("usage"), ...)
```

**`build_chat_payload` for the vllm provider (`openai_compat.py:38-98`):**
```python
payload = {"model": model, "messages": messages, "stream": bool(stream), "temperature": temperature, "top_p": top_p}
payload["max_tokens"] = max_tokens; payload["tools"] = tools; payload["tool_choice"] = tool_choice
if normalized == "vllm":
    if top_k > 0: payload["top_k"] = top_k
    payload["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
    if seed is not None and seed >= 0: payload["seed"] = seed
```

**Reasoning effort xhigh for Flash-Next:** it is a chat-template kwarg, applied in three places:
- On the server: `--default-chat-template-kwargs {"enable_thinking": true, "reasoning_effort": "xhigh"}`.
- Per request: `arc2_duck._harness_template_kwargs` (`arc2_duck.py:1559-1567`) adds `reasoning_effort` from `E12_REASONING_EFFORT="xhigh"` (E:332-335). Duck also adds `preserve_thinking: True` (`ARC3_PRESERVE_THINKING=1`).
- So each request carries `chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "xhigh"}`.
- The tokenizer's own template (sha c3cf9e34) puts an "xhigh" sentence in the head. The notebook asserts it at E:2112. Head is 61 tokens; the generation tail is 7 tokens and ends `<think>\n` (`NB5_TAILS` = `[[248068,198],[248068,271,248069,271]]`).

**Sampling:**
- T = `E12_DUCK_TEMPERATURE` 1.0 (E:331 → `LOCAL_ANALYZER_TEMPERATURE`), top_p 0.95, top_k 20 (DUCK_ENV).
- No seed: `LOCAL_ANALYZER_SEED` is unset, so -1 and not sent.
- The server's `--override-generation-config` holds the same values, with min_p 0.

**max_tokens and budget:**
- Per attempt `E12_B_ATTEMPT_TOKENS` = 104,000 (T1), reserve 4,096. The first-request cap `E12_DUCK_MAX_OUTPUT = NB5_CAP = 99,904` (E:282, 328).
- Every request then passes `_nb5_guard` (E:741-807). It renders the request on the server's `/tokenize` (2 retries; local verified tokenizer as fallback, else no request). Then `NB5L.context_guard` sets `cap = min(requested, peak_len - prompt - 2048)` with `peak_len = min(131072, 8192 + budget + 8192)` (E:733-738; T1 120,384, B2 131,072).
- An attempt that can no longer reach the 60k floor becomes BUDGET_LIMITED. No room → `CONTEXT_EXHAUSTED`, and no request is sent.
- Run: 2,781 requests tokenized, 0 estimated, 265 clamps.

**Stop handling:** there are no stop strings. Turns end at a tool call or finish. The commit turn uses mode "A" in this phase, so it goes through the normal chat path. Variant C (`/v1/completions` with token ids, `NB5L.commit_c_call`) exists but was not used (spec cmode "A").

**Timeouts:**
- `E12_DUCK_REQUEST_TIMEOUT` = 7,200 s (E:375) becomes `analyzer_timeout` in `solve_all_r6`. B2 uses `E28_B2_REQUEST_TIMEOUT` = 9,000 s (E:598).
- Connect timeout 10 s.
- `solve_all_r6` docstring: "every request is clamped to the deadline (R6-S6)".

**Reading finish_reason and reasoning:**
- E21 wrapper (E:637-657): vLLM 0.32 reports `tool_calls` even when a reply is cut at max_tokens. The wrapper restores `"length"` when `completion_tokens >= min(cap, 131072 - prompt_tokens)`.
- Reasoning is read via `_extract_reasoning_text`: `message["reasoning"]`, then `reasoning_content` (`tool_agent.py:3439-3443`).
- History sends reasoning back under `ARC3_REASONING_HISTORY_KEY="reasoning"` (E:626; vLLM 0.32 reads only that key). `E12_RAW_REASONING_HISTORY=1`.

**Run request stats:**
- 2,765 traced requests, 22 errors. finish: tool_calls 2,712, length 30, stop 1.
- TTFT mean 6.86 s; e2e mean 184.4 s; latency p50 66.6 s, p90 479 s, max 4,207 s.

**Other client helpers in the notebook:**
- `nb5_chat(body, timeout=600)` (E:2830): urllib POST to `/v1/chat/completions`.
- `nb5_complete(ids, max_tokens, timeout=600, **extra)` (E:2834): `/v1/completions` with prompt token ids, model `"q27"`, T 0 by default.
- `_e12v_request_json` (E:2332).
- `NB5L.tokenize_chat(base, payload)`: `/tokenize`.
- `L.Stream(port, ids, max_new, temperature, seed)`: streaming `/v1/completions`, `ignore_eos=True`, read timeout 7,200 s.

---

## 4. Concurrency and KV

**Settings:**
- `--max-num-seqs 24` on the top rung only (E:3592-3593). The rule `NB5L.workers_rule = max(1, floor(pool/(ceil(65536/4096)+F)))` gives 21 → `RULE_BELOW_LAUNCH`; the top-rung override sets workers = 24 (E:2806-2807).
- Admission is `"slots"`: `NB5_RES = NB5L.Reservation(24)` (E:3248), one slot per attempt; the engine schedules KV and preemptions are allowed.
- `ARC3_MAX_ACTIVE_STREAMS=24`; the isolation preflight passed, so 24 workers.
- `NB5_GATE` = 1.0.

**Measured:**

| Measure | Value |
|---|---|
| KV tokens (`GPU KV cache size`) | 1,652,197 |
| `num_gpu_blocks` / usable / block size | 480 / 479 / 4096 |
| Max concurrency at 131,072 | 12.61× |
| F_log / F | 5.986 / 6 |
| T1 projected | 36 blocks → 13 streams |
| B2 projected | 38 blocks → 12 streams |
| Preemptions | 438 (119 sampler intervals); 180 of 189 attempts overlapped one (upper bound) |
| Running mean / max | 23.41 / 24 |
| running_check | FAIL_RUNNING_NE_INFLIGHT (server below client in 401 of 1,439 samples; server waiting in 378 samples = 5,670 s) |
| Solver tok/s | 419.5 (window), 403.6 (whole notebook) |
| Steady-mixed | p50 506.5, mean 473.7 |
| Generated tokens | 9,062,803 |
| Prompt tokens / prefix hits | 101.4 M / 80.9 M (79.79 %) |
| MTP mean acceptance length | 2.815 |
| GPU memory after ready (per card, MiB) | ≈20,307 / 20,311 / 21,877 / 21,877 of 23,034 |
| Host RSS | 70.7 GB |

**Per-band capacity facts:**
- This run, `streams_fit_by_context` at gate 1.0, F 6: 40k 29, 60k 22, 70k 19, 90k 17, 120k 13, 131,072 12.
- K3_ONLY v2 GPU (build facts `k3_v2_gpu`): resident bound {40,000: 29, 64,096: 21, 90,000: 17, 120,000: 13}. Sweep at the 0.95 limit:
  - 20k@16: 460.4 tok/s;
  - 20k@24: 562.0 tok/s, 0 preemptions;
  - 40k@16: 444.7 tok/s;
  - 40k@24, 60k@16, 100k@16: KV-bound, skipped.
- **Not in the E28 code:** no 8k-band or 116k-band capacity table exists. Only the 8k warm-up at 24 streams, and the "116k" mention at E:125, which is RC2 FP8 answer history.

---

## 5. Time budget machinery

**CELL 1 (E:243-252):**
- `TOTAL_BUDGET_SEC` = `P["time_budget_sec"]`, overridable by `Q27X_TIME_BUDGET`.
- `HARD_STOP = T0 + TOTAL - hard_margin_sec`.
- `VERIFY = T0 + TOTAL - 60`.
- `DRAIN_DEADLINE = HARD_STOP - 8 min` and `ADMIT = DRAIN - 2 min`. Both are only asserted; nothing uses them.

**Profiles:**
- **E28EVAL120** (E:217-219): TOTAL 415 min (24,900 s), margin 5 → HARD_STOP 410 min.
  - Solve end cap = `HARD_STOP - E12_STOP_BEFORE_HARD_SEC` (180 s) = 407 min (E:395, 4114).
  - Fixed window `E28_WINDOW_SEC` = 21,600, set in CELL 8 (E:4867-4869).
  - `nb5_phase` (E:3191-3209): `window = min(fixed, HARD_STOP-180 - now - 60)` (flagged unequal when shortened); skipped if below 5 min.
  - `e21_cell6` (E:4114-4167): `_end = min(HARD_STOP-180, now + window)`. In the commit it is shifted after the rev6 isolation preflight.
- **Deadman:** `K100_DEADMAN_SEC = 24900` is a literal, not derived from the profile (E:54). At T0 + 415 min it writes a marker to stdout, stderr and `/kaggle/working/k100q_deadman.txt`, SIGKILLs the process tree, then `os._exit(0)` (E:58-117). Commit only.
- **e23 watchdog:** kills every server at `T_START + TOTAL` (E:1018, 2019).
- **E15B watchdog** (E:4185-4269): relaunches a dead or stalled server (no progress for 600 s, or /health failing for 180 s), at most 2 relaunches, needs at least 20 min left. `e21_relaunch` (E:3395) needs at least 30 min before the solve end and reuses the booted config.
- **Drain:** no separate drain step. Branches stop at the deadline; `join_grace_sec` = 90; writes after the solve end are skipped (E:3747).

**Competition rerun path:**
- `RERUN_PROD` (E:213, 221): `_NB5_T` with 12 h − 3 min = 43,020 s, hard margin 2 min → HARD_STOP 715 min, solve end 712 min.
- Source test, `n_tasks=None` (all tasks); NB6 levers plus E28 levers; phase `["EVAL","all",None,"on","A","fp8"]`.
- E18 ladder (≤3 tries); no window (it solves until HARD_STOP − 180 s); no deadman, no sha gate.

**Submission writes:**
- CELL 0 writes the identity submission of the test file.
- `e12_write_submission` (E:3715-3724) runs after every branch callback (until the solve end), at the end of `e21_cell6`, and in `e21_cell7`:
```python
sub = identity_submission(test_challenges)
if RERUN:
    recs = {t: list(v) for t, v in E12_RECORDS.items() if v}
    for t, rows in E12A.build_submission({t: CHALLENGES[t] for t in recs}, recs).items():
        if t in sub and len(sub[t]) == len(rows): sub[t] = rows
_e12_write_json_atomic(SUBMISSION_PATH, sub)
```
  In the commit, `submission.json` stays the test identity. The eval answers go to `phase_*/e12_generated_attempts.json`.
- CELL 8 writes no submission; it writes `nb5_report.json`.
- CELL 9 writes `e28_k100_report.json` and `Q27X_NB5_BUNDLE.zip`.
- The Duck bundle `Q27X_E21_NB5_HALF<phase>_DUCK_BUNDLE.zip` comes from `e21_cell7`.

**What a ~1-h commit profile needs:**
- **Existing small profiles:** this notebook keeps only `RERUN_ALL`, `RERUN_PROD`, `E28EVAL120` (and `NB5HARNESS` for the harness). The builder cut the smaller rows for size (B:248-254): NB5EVAL120 / NB5EVAL30 / NB5EVAL30MTP / NB5QUALITY / NB5SMOKEDIAG / NB5BUDGET, NB5QUALITY_R2, NB6COMMIT. **There is no EVAL20 in this lineage.**
- Nearest proven rows:
  - NB6's `NB6COMMIT` (`build_e25_nb6.py:88`): 50 min, 3 tasks.
  - K100Q's `K100QUALITY` (`build_e26_k100_quality.py:366-372`): 150 min, 30 tasks, 6,600 s fixed window, deadman 9,000 s.
- To add a 1-h profile:
  - a new `PROFILE_TABLE` row: `time_budget_sec ≈ 3600`, `hard_margin_sec 300`, source "eval", `task_list` or `n_tasks`, `fixed_window_sec`, the K100 boot levers and phases;
  - add it to `K100_PROFILES` (E:180) and set `NB5_DEFAULT_PROFILE`;
  - change the `K100_DEADMAN_SEC` literal.
- Timeline with this run's overhead (window start 879 s) and a 60-min budget: HARD_STOP 55 min, solve end ≤ 52 min, window room ≈ 2,180 s (about 36 min).
- A failed boot try costs about 13 min. A second try is still allowed because left() stays above 25 min, but it leaves about 8 min of window.
- Per-stream throughput limits are in the top finding.

---

## 6. Per-task eval correctness and task loading

**Loading (E:989-1006):**
```python
if P['source']=='eval':
    _ch=_load_json(ARC_ROOT/'arc-agi_evaluation_challenges.json'); _sol=_load_json(ARC_ROOT/'arc-agi_evaluation_solutions.json')
    TASK_IDS=sorted(P['task_list']) if P.get('task_list') else sorted(random.Random(TASK_SAMPLE_SEED).sample(sorted(_ch),P['n_tasks']))
    CHALLENGES={t:_ch[t] for t in TASK_IDS}; SOLUTIONS={t:_sol[t] for t in TASK_IDS}
else: CHALLENGES=dict(test_challenges); TASK_IDS=sorted(CHALLENGES); SOLUTIONS=None
```
- `TASK_SAMPLE_SEED` = 20261005. With `n_tasks`=120 every eval task is taken. A smaller `n_tasks` gives a seeded random subset.
- CELL 8 snapshots `NB5_ALL_IDS / CH / SOL` (E:4820) and the smoke eval files (E:4821).
- `nb5_phase` rebinds `TASK_IDS / CHALLENGES / SOLUTIONS` and `OUT = BASE_OUT/phase_<name>` (E:3215-3220).

**Scoring:**
- `e21_cell7` (E:4694-4712):
  ```python
  for t in TASK_IDS:
      for qi, att in enumerate(_gen[t]):
          gold = SOLUTIONS[t][qi]; a = att["attempt_1"] == gold; b = att["attempt_2"] == gold
          hits += a or b; pass1 += a; pass2 += b; total += 1; th += a or b
          recs_eval.append({"task","query","hit","pass1","pass2","accepted_programs","accepted_programs_correct","harvested_programs","harvested_programs_correct"})
      task_norm += th / max(1, len(_gen[t]))
  report.update(eval_query_hits, eval_query_total, eval_pass1_hits, eval_pass2_hits, eval_task_norm_pct, eval_unit_records)
  ```
  - `_gen = E12A.build_submission(CHALLENGES, records)` is written to `OUT/e12_generated_attempts.json`. `OUT/e12_run_report.json` also holds `eval_b_attempts` (E:4713-4760).
- Other per-task records:
  - `nb5_phase` copies `score` and `bands / task_solve / quality`.
  - `nb5_qab_phase_record` writes `NB5L.qab_tasks` (per task: queries, query hits, solved, per-attempt outcome / tokens / exact) to `nb5_report.json → quality_ab.per_config[phase].tasks`.
- CELL 9 `E28L.build_e28_report`:
  - `score_from_generated` recomputes the score from the generated attempts against gold and must agree;
  - `per_task` {queries, hits, solved};
  - per-attempt `attempts` (phase, tokens, outcome, train k/n, accepted, exact);
  - status VALID.
- Run: 107/172 queries (pass1 100, pass2 7), task-norm 65.0 %, 74/120 solved, agree True.

---

## 7. Cleanest hook point for a direct solver

**Call chain:** `nb5_phase` (E:3179) → `e21_cell6()` → `E12A.solve_all_r6(...)` (E:4274-4291) → `e21_cell7()`.

**Cleanest swap:** keep cells 0, 3, 4, 5, 8, 9 and the `e21_cell6` / `e21_cell7` text unchanged, and rebind `E12A.solve_all_r6` to a direct solver with the same signature, in one new cell between CELL 6/7 and CELL 8.
- `e21_cell6` looks the attribute up at call time, so the rebind takes effect.
- The swap keeps unchanged:
  - the model-id lookup and the isolation preflight / worker count;
  - the window and deadline math, and the E15B watchdog and relaunch;
  - `_e12_branch_done_r6` → `e12_write_submission` (the proven E15B writer and the rerun overlay);
  - the CELL 7 scoring and bundle, the nb5_phase record, and the E28 report.
- CELL 1 cannot stay byte-identical: the new profile, guard and deadman literal live there. Everything else can.

**Call contract (E:4274):**
```python
E12A.solve_all_r6({t: CHALLENGES[t] for t in E12_ORDER}, E12_ORDER, out_dir=E12_OUT,
    deadline=time.monotonic() + max(0.0, _end - time.perf_counter()), base_url=SGL_URL + "/v1",
    model_id=E12_RUN["model_id"], branches=E12_BRANCHES, max_generated_tokens=E12_MAX_GEN_PER_BRANCH,
    save_request_logs=..., on_branch_done=_e12_branch_done_r6, workers=_workers, ..., capacity_probe=_e12_capacity_probe,
    max_generated_tokens_b1=..., ...)
```
- It returns `{task: [records]}`. Afterwards `E12_RUN["rev6_scheduler"] = dict(E12A.SCHED_LAST)` is read; `gauges` are optional.
- Each published record must carry `task_id`, `branch` (0 = attempt 1, 1 = attempt 2), `status="done"`, `publish_seq`, `accepted`, `generated_tokens`, `t0`/`t1` (epoch), `note`, `nb5_verdict`, and `submissions=[{"test_outputs":[grid per test], "accepted": bool, "train_exact": [...], "t": epoch, "branch": b}]`.
- All downstream readers use `.get()`: CELL 7 (E:4422-4441), `NB5L.qab_tasks`, `E28L.attempt_rows`.

**What to set in the profile for a direct notebook:**
- Leave out `e28_sched`. CELL 3's apply block is gated by `P.get("e28_sched")` (E:585-586), so Duck stays original. Otherwise `build_submission` switches to `select_attempts_e28`.
- `select_attempts_r5` treats `accepted=False` submissions as partial candidates. Under `E12_B_MODE=1` it also appends train-exact CPU rule outputs after the model candidates (`arc2_duck.py:2790-2827`). A pure direct measurement should record or avoid that.
- `attempt_end` reports a non-accepted record without a verdict as `FAILED_OR_INTERRUPTED/no_terminal_evidence`. Direct runs need their own outcome labels, or must fill the existing verdict fields.

**What the direct client should call:**
- `NB5L.tokenize_chat(SGL_URL + "/v1", payload)` → prompt length.
- `NB5L.context_guard` → max_tokens ≤ 131,072 − prompt − 2,048, floor 60k.
- A chat request:
  ```python
  {"model":"q27","messages":[{"role":"user","content":prompt}],"max_tokens":cap,"temperature":1.0,"top_p":0.95,"top_k":20,"chat_template_kwargs":{"enable_thinking":True,"reasoning_effort":"xhigh"}}
  ```
  Send it with `_e12v_request_json(SGL_URL + "/v1/chat/completions", payload, timeout=min(7200, deadline_left))`. `nb5_chat`'s default timeout of 600 s is too short.
- Read `choices[0].message.content` (the grid text), `.reasoning`, `finish_reason` ("length" means cut, with no content), and `usage.prompt_tokens / completion_tokens / prompt_tokens_details.cached_tokens`.
- Run about `workers` threads.

**Bypassed by the direct path, so the solver must count these itself:**
- Duck's `_nb5_cc` tracer and guard (E:849-890): the phase `tracer` will be empty.
- The `_nb5_session_run` reservation (E:894-937): reservation grants 0.
- The 15-s `nb5_sampler` still runs: server running, preemptions and tok/s are still recorded.

**Must still load:**
- CELL 3's Duck package: CELL 8 aborts if the sandbox self-test fails (E:4827).
- `nb5_probes` uses `E12A.ARC2_TOOL_DESCRIPTION`, `_E12_TA` and `NB5_TAILS` (E:2518, 2877).

---

## 8. Notebook size, payload handling, CPU kit

**Size:**
- The notebook is 394,891 B (sha d3d132da). The build asserts `len(data) < 400_000` (B:626); about 5.1 kB is left.
- It is written as compact JSON (separators `,` `:`; B:612-617), and `e21_cell7` is packed as an lzma + b85 literal exec'd at import (B:455-469).
- Cell source sizes: CELL 1 79.7 kB, CELL 3 30.8 kB, CELL 5 187.2 kB, CELL 6/7 67.2 kB (readable `e21_cell6` ≈ 51.4 kB), CELL 8 6.1 kB, CELL 9 17.8 kB.
- The E28 module literal (E:494) is 34.2 kB: lib 11.8 kB + scheduler 23.1 kB compressed. Dropping the scheduler frees about 23 kB, but that edits CELL 1.

**Payload pattern:** large files live in sha-pinned dataset files.
- `q27x_payload(name, sha)` (E:428-444): searches the payload dirs and `/kaggle/input` up to 3 levels deep.
- `k100q_payload()` / `k100q_ds_text(name, sha)` (E:456-480): same tf-ref dir, raise on a mismatch.
- Inline literals are sha-checked lzma+b85 (`NB5_LIB_SRC` from the payload blob E:481; `E28_SRC` E:494).
- A direct solver can ride the same pattern: either a new file in a new tf-ref dataset version (the kits then check the version as a prerequisite), or one packed literal.

**CPU kit (`build_e28_k100_cpu_kit.py`):**
- It execs `build_e26_k100q_cpu_kit.py` with exact-once edits.
- It reads `e28k100_build_facts.json` and the notebook. `E28_READABLE` rebuilds `code[5]` from the packed literal using the marker `"# ---- E28 K100 (size: the 400-kB commit limit)"`.
- Four kits on esfersami50 (v5), each printing `@@KIT <name> ALL OK@@`:

| Kit | What it checks |
|---|---|
| units | `e28k100_unit_tests.py` on Py3.12 (at least 42 checks, only E4 skipped); tf-ref dataset version and payload shas; the real `arc2_duck.py` pre-sha from the mounted payload. |
| e2e-rerun | RERUN_PROD on a fake 6-task hidden set with `nb5_fake_server.py`; scheduler applied, plain order, I8 first try. Retry mode: rung 1 INT8 OOM → I8 drop → same rung BF16 OOM → rung 2 boots. |
| e2e-commit | E28EVAL120 on a fake 3,000-s deadline: the 120 tasks scored, window shortened (unequal), CELL 0 fallback kept, deadman armed and not fired, E28 report VALID with A/B1/B2/C rows, `--max-model-len 131072`. |
| real | K100Q's real kit: real Merlin install, the klee100 merge and digests, #60490, worker extension, scales; every rung of both profiles through `build()`, `host_phase`, `--help=all` and argparse (top rung 24 seqs, captures cover the PP2 micro-batch); method-map probe; Q-defer apply/revert; CELL 0 vs E18 R2. Plus the E28 checks: scheduler apply/revert/re-apply, `e28_i8_static` SUPPORTED_STATIC on the real tree, Merlin's INT8 quantizer on 8,192 real lm_head rows. |

**Reuse for a direct notebook:**
- The `real` kit carries over almost unchanged if cells 0–5 and 8 stay identical. Its `E28V` scheduler check needs `E28_SCHED_REC.applied`, so it fails if the profile drops `e28_sched`. Remove or retarget that check.
- The `units` kit needs a new unit-test file, because the provenance checks are tied to this notebook.
- The e2e kits need new checks and a fake server that answers plain no-tool chat requests. I did not verify that `nb5_fake_server.py` supports this.
