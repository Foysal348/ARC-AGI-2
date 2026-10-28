# E21 CPU harness

Runs every code cell of a real E21 notebook (NB1 INT4 S20 or NB2 FP8 SSM16 S16) on CPU against a fake vLLM server, then
checks what the notebook produced. No GPU, no install, no Kaggle.

## Files

- `fake_vllm_duck_server.py`: fake vLLM OpenAI server. The notebook's own engine cell launches it as
  `[sys.executable, fake] + flags` (`E11_HARNESS=1`, `E21_FAKE_SERVER=<this file>`). It accepts both argv forms:
  `--model <path> ...` (vLLM 0.19 api_server, half A) and `[serve] <path> ...` (vLLM 0.32 `vllm serve`, half B).
  - It refuses to start (exit 2) if a Duck serving flag is missing (`--enable-auto-tool-choice`,
    `--tool-call-parser qwen3_coder`, `--reasoning-parser qwen3`, `--default-chat-template-kwargs` with preserve_thinking
    true, `--enable-prefix-caching`, `--max-model-len >= 131072`), or if the served name is a Duck alias.
  - It also refuses unknown flags ("unrecognized arguments"); the accepted set is the flags the E21 notebooks pass.
  - Startup lines follow the vLLM hybrid layout (rev 2):
    - "Setting attention block size to B tokens" (B = 1,600, or 832 with `--mamba-ssm-cache-dtype float16`).
    - "GPU KV cache size: num_blocks // 4 x B tokens".
    - "Maximum concurrency for M tokens per request: num_blocks / (cdiv(M, B) + 15)x" (15 = 3 x (2 + MTP-3)).
    - num_blocks is 846 (FP8), 1,069 (int4/w4a16 path) or 1,628 (B 832). `FAKE_NUM_BLOCKS=<form>:<n>` overrides it.
  - Every 2 s while busy it prints the `Engine 000: ...` stats line and the `SpecDecoding metrics: ...` line:
    - every 4th busy line has ", Preemptions: 1" (0.32 also ", Deferred: 1 reqs"), counted in
      `vllm:num_preemptions_total`;
    - the prefix-cache field is absent until the first prefix query;
    - `FAKE_ACCEPT_LEN=<form>:<x>` sets the MTP acceptance.
  - Rev 2 behaviours:
    - A tool call cut at max_tokens (the scripted CUT_CODE step) answers finish_reason "tool_calls" with
      completion_tokens == max_tokens, as vLLM 0.19 / 0.32 do. Complete replies stop before max_tokens.
    - The contract probe ("Use the python tool exactly once") gets reasoning + one python call in turn 1 and a plain
      answer in turn 2.
    - The cache probe ("Reply with exactly this code: QZ...") echoes the code.
  - Endpoints: `/health`, `/v1/models`, `/metrics` (vllm:* gauges and counters), `/fake_stats`, and
    `/v1/chat/completions`. For chat requests it runs the E12 rev 6 scripted Duck program from Exp 14 `e14_fake_server.py`.
  - Reply shape: the thinking goes in `reasoning`. `prompt_tokens_details.cached_tokens` is present only when the value
    is > 0 and `--enable-prompt-tokens-details` is set. A request with prompt + max_tokens > max-model-len gets HTTP 400.
  - Input, as vLLM 0.19 / 0.32 handle it: only `reasoning` of an assistant history message counts toward the prompt.
    `reasoning_content` is ignored, and the fake counts how often each key arrives.
  - Prefix cache: whole 1,600-token blocks of the longest prefix shared with earlier prompts and outputs.
  - Running requests are capped at `--max-num-seqs`; extra requests wait in FIFO order.
  - Fault injection, all optional:
    - `FAKE_REJECT_FLAG=<flag>` rejects a launch with that flag.
    - `FAKE_REJECT=<0.19|0.32|*>:<substring>,...` rejects a launch per argv form.
    - `FAKE_NEVER_READY=1|0.19|0.32` keeps the server from ever becoming ready.
    - `FAKE_DIE_AFTER_SEC=<s>` makes the server die s seconds after its first Duck tool request. Narrow it with
      `FAKE_DIE_FORM`. `FAKE_DIE_TIMES=<n>` (default 1): the first n servers of that form die, the next survives.
    - `FAKE_DROP_BLOCK_LINE=<form>` omits the block-size line (the notebook must refuse the config).
    - `FAKE_QUIET=<form>`: no stats lines. The log stays silent, so the watchdog probes /health.
    - `FAKE_HOLD_ONE=<form>:<n>:<sec>` holds the n-th Duck tool request for sec seconds (deadline test).
    - `FAKE_METRICS_FAIL=<form>:<a>:<d>`: /metrics answers 500 from (first tool request + a) for d seconds.
  - The fake prints vLLM's `non-default args: {...}` line, e.g. `'language_model_only': True`.
  - On Windows it binds without SO_REUSEADDR, so a port that another server holds fails loudly.
  - Evidence: `FAKE_STATE_DIR/launches.jsonl` gets one row per start (argv, form, outcome: ready / refused / rejected /
    never_ready / died), and `stats_<pid>.json` is refreshed every 2 s.
- `harness_e21_nb.py`: the notebook harness. Its setup follows Exp 15 `harness_e15b_nb.py`: the Windows shims, the
  fixtures, the payload in the dataset shape, the INT4 / drafter model dirs in the real Kaggle path shape, and staging to a
  local `shm`.

## Usage

```
E21_MODE=<mode> python harness_e21_nb.py "<notebook.ipynb>"
```

Env:
- `E21_HARNESS_ROOT`: where the run folders go. Default `D:/ARC AGI 2/_scratch/e21/harness_runs`. The notebook's
  working dir is `<root>/run_<NB>_<mode>_<time>/`.
- `E21_FAKE_SERVER`, `E14_FIXTURES_DIR`, `Q27X_PAYLOAD_DIR` (a flat folder of the q27x_*.bin is fine): the defaults are
  the local paths.
- `E21H_PROFILE`: default `EVAL6`, 6 eval tasks.
- `E21H_HALF_SOLVE_SEC`: default 90 s; 1,500 s in death mode (a relaunch needs >= 12 min before the solve end; the solve
  still ends when every attempt is done); 150 s in gateAB.
- `E21H_MAX_SEC`: hard limit for the whole run, default 1800 s.

The harness also sets these for the notebook:
- `Q27X_TIME_BUDGET=10800`. A long budget is needed: `e21_begin_half` skips the fallback when fewer than 30 min are
  left, and CELL 1 asserts that 2 x window + reserve fits. Rerun mode uses 2400 instead.
- `E21_B_SETUP_RESERVE_SEC=300`, and `E21_ANY_PROFILE=1` (CELL 1 otherwise only allows E21X2EVAL120 / RERUN_ALL).
- `Q27X_GPU_WAIT_MAX=1`. There is no nvidia-smi, so `_wait_gpu_release` would otherwise wait 180 s per call.
- `E18_SILENCE_SEC=120`, or 15 in halfAfail mode.
- Short Duck budgets, as in the E15B harness: `E12_MAX_GEN=6000`, `E12_B_ATTEMPT_TOKENS_B1=4000`,
  `E12_B_MAX_ATTEMPTS=3`.

Output: one line per check, `[e21 PASS] ...` or `[e21 FAIL] ... -> info`, then a `[e21 SUMMARY]` line. When every check
passes it prints `@@E21_HARNESS_OK@@` and exits 0. Otherwise it prints `@@E21_HARNESS_FAIL@@` and exits 1.

Fixed ports (1234 for half A, 8011 for half B): run the modes one after another, never in parallel. Before it starts,
the harness waits up to `E21H_PORT_WAIT_SEC` (900 s) for both ports to be free. If they stay busy it prints
`@@E21_HARNESS_PORT_BUSY@@` and exits 4.

If `e14_heartbeat.log` has a gap > 300 s (the host was suspended), the harness prints `[e21 HOST_SUSPENDED]` and
`@@E21_HARNESS_HOST_SUSPENDED@@` and exits 3 instead of giving a verdict.

If the fake servers saw conversations of tasks this run does not have (another harness run on the same host used the
fixed ports at the same time), it prints `[e21 FOREIGN_TRAFFIC]` and `@@E21_HARNESS_FOREIGN_TRAFFIC@@` and exits 5.

## Modes

| mode | fault | what must happen |
|---|---|---|
| commit | none | Half A (0.19) runs, then half B (0.32), each on its primary config, both scored. |
| rerun | `KAGGLE_IS_COMPETITION_RERUN=1` | Half A only, on every test task. The submission is half A's generated attempts. Half B never launches. |
| fallbackA | `FAKE_REJECT=0.19:w4a16` (NB1) or `0.19:--mamba-ssm-cache-dtype` (NB2) | Half A's primary exits and its fallback becomes ready (NB1: FP8 seqs 16; NB2: no SSM flag). Half B repeats half A's fallback config (`E21_B_FOLLOWS_A`, `spec_note`). |
| halfAfail | `FAKE_NEVER_READY=0.19` | Both half-A configs stall. Half A records no engine and does not raise. Half B runs. The submission stays valid. |
| death | `FAKE_DIE_AFTER_SEC=8`, `FAKE_DIE_TIMES=2` (0.19) | The half-A server dies twice mid-solve. The watchdog relaunches twice; the second relaunch switches to the half's other config (`relaunch_cfg_switched`), and the last server serves Duck. |
| gateAB | `FAKE_NUM_BLOCKS=0.19:510` (NB1) or `850` (NB2), `FAKE_ACCEPT_LEN=0.19:1.2`, profile E15B (20 tasks) | Half A gets 10 streams and half B its full count. Half B's server really runs that many requests at once, and its fresh Duck gate never blocks. Half A gets `mtp_acceptance_warning` 1.2. |
| wdhold | `FAKE_QUIET=0.19`; the harness holds half A's watchdog in its first /health until half B is solving | Half A's watchdog exits after the release (`e15b_watchdog_alive_after_join` true for A), and there is no relaunch in A or B. |
| overrun | `FAKE_HOLD_ONE=0.19:6:600` | Exactly one half-A response is held in flight past the solve deadline. The solver still returns within 120 s of its deadline (stragglers are information only: Duck's per-request deadline ends the held call), half A ends <= solve start + window + 300 s, and the submission stays complete. |
| kvoutage | `FAKE_METRICS_FAIL=0.19:3:150`, profile E15B, 420-s window | `probe_failures` > 0. In the outage's first 115 s (7-s grace) no new branch reaches the server while requests run. After 120 s Duck's own fail-open applies (`probe_fail_open` > 0). Duck finishes (no deadlock). |
| gpuheldB | `E21_FAKE_GPU_HELD=1` set at half B's begin (harness wrapper of `e21_begin_half`) | Half B makes 0 launches, its fatal is "GPU memory still held before the launch", and `config_status` is NOT_RUN. Half A runs normally. |
| reasoningkey | none (= commit) | Names the reasoning-key checks. Those checks run in every mode. |

## Checks (every mode, where they apply)

- The run:
  - Every code cell ran without an uncaught exception.
  - The notebook has 10 code cells.
  - The halves that ran are `["A","B"]`, or `["A"]` in rerun.
- submission.json: complete and valid (every task, every test input, two grids each), with no temp file left.
- Launches:
  - Every port-1234 launch used the 0.19 form.
  - Every port-8011 launch used the 0.32 form. In rerun there was no 0.32 launch at all.
  - No launch was refused for missing Duck flags.
  - The primary configs match the variant: NB1 is INT4 with seqs 20; NB2 is FP8 with seqs 16 and the float16 SSM flag.
  - The served names are the E21 names.
- Capacity:
  - On every fake server, real running stayed <= `--max-num-seqs`.
  - The capacity check gave the expected streams (20 or 16), and Duck's streams and workers match it.
- Reasoning key:
  - The assistant history carried `reasoning`, and no message or request carried only `reasoning_content`.
  - `ARC3_REASONING_HISTORY_KEY` is `reasoning`.
- No request was over the context limit.
- Per half:
  - The engine is the primary or fallback config the mode expects, and the smoke passed.
  - Duck ran with no solver error, branches started, and the isolation preflight passed.
  - `b_summary` was built from the vLLM probes, and the stats-line summary has lines.
  - There are no worker errors and cached tokens are > 0.
  - The half was scored (commit modes).
  - The artefacts exist: `q27x/half<H>/e12_run_report.json`, `vllm_server_half<H>_<tag>.log`,
    `sglang_decode_summary.json`, `duck/`, and `Q27X_E21_<V>_HALF<H>_DUCK_BUNDLE.zip`.
  - The runner recorded no step error, and the server was stopped at the half's end.
- Order: half B began after half A ended, and every B launch came after every A launch.
- Report:
  - `q27x/e21_report.json` has one row per half with the score / engine / attempts / errors fields.
  - `Q27X_E21_<V>_BUNDLE.zip` holds the report and the submission.
- Profile: the Kaggle default `E21X2EVAL120` has 120 tasks and a 5.5-h window, and both windows fit the budget.
- Rev 2 / 2b, per half where Duck ran:
  - The capacity numbers come from the block-size and concurrency lines (B, num_blocks, F = 15).
  - The contract probe passed: one tool call, turn-2 shortfall <= 0, cache probe.
  - Every fake cut call was restored to "length" (`length_restored` == the fake's count), and Duck dropped or saved it.
  - `e21_report` has `recomputed_prompt_tokens`, `counters`, `contract`, `gate`, `stage1`, `score_attempt1_strict_pass1`,
    `window_sec` and `prefix_cache_hit_pct_last`.
  - `mtp_acceptance_warning` appears only where the fake logs acceptance < 1.5.
  - The last half does not wait for the GPU release; a non-last half does.
- Rev 2 / 2b, run level:
  - Both halves have the same window (`windows_equal`).
  - `tool_calls_with_content` is recorded.
  - The vLLM 0.32 background install thread exists in two-half runs and not in rerun.
  - `E12_TOKENIZER_JSON` exists.
- Rev 2c, every two-half mode:
  - The harness wraps `E12A.solve_all_r6` and invokes half A's callback with an accepted record after half B's solve.
    Half B's records, `branches_accepted` and submission.json bytes must stay identical, and A's
    `late_cross_half_callbacks` goes up by 1.
- Rev 2c, every half with an engine:
  - `solve_overrun_sec <= 120`, no `deadline_violation`, and `end_sec <= planned_end_sec + 300`.
  - `lmo_in_log` is True.
- Rev 2c, run level:
  - `config_status` is NOT_RUN in halfAfail, FALLBACK_BOTH in fallbackA, SWITCHED_MIDHALF in death (after a
    config-switching relaunch), and PRIMARY_BOTH otherwise.
  - `comparison_valid` is True only for two-half runs with PRIMARY_BOTH or FALLBACK_BOTH; gateAB (10 vs full streams) still counts as valid by the notebook rule, since capacity is part of the
    engine difference.
- Rev 2d:
  - A non-last half leaves no Duck scheduler thread (`handoff_scheduler_threads_left` empty, `handoff_wait_sec` <= 120).
  - The SMOKE notebooks (`*_SMOKE.ipynb`) run in commit mode with their own default profile and budget, `E21SMOKE`
    (12 tasks, 2 h). Q27X_PROFILE and E21_ANY_PROFILE are not set; only `E21_HALF_SOLVE_SEC` shortens the window.
    The harness asserts PROFILE == E21SMOKE.
