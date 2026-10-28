# Experiment 19: vLLM 0.19 MTP / prefix-cache / async / KV correctness on the S1-60K stack (planned)

**Question.** Do any of these silently corrupt Qwen3.8-27B outputs on our exact production stack?
- vLLM 0.19's MTP speculative decoding;
- prefix caching;
- async scheduling;
- the fp8 KV cache.

The production stack is S1-60K (LB 34.58).

**Status.** Built and validated on the local CPU harness. Not pushed and not run on Kaggle.

## Base

S1-60K, Experiment 10: `Q27-XHIGH-v1_4_4_E10_S1_60K_POETIQ_FIX_ORDER2.ipynb`, sha `ed530071`.

These cells are byte-identical to S1. The builder asserts this.
- CELL 0: mode and the identity fallback submission.
- CELL 2: vLLM 0.19.0 + transformers 4.57.6 from `mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3`, and the tokenizer.
- The task-order cell.
- The Poetiq core: `poetiq_build_message`.

These functions are inserted verbatim:
- S1 CELL 5's `ensure_libcuda_linkable` + `MAX_JOBS`;
- S1's `e10_render_request`.

CELL 1 is S1's, plus the E19 profiles and config block.

## Engine

The engine is S1's `ENGINE_CONFIG`. It runs as a vLLM 0.19 OpenAI server from the kernel's own Python (`python -m vllm.entrypoints.openai.api_server`). There is one server per arm. Between arms it is killed: SIGTERM, then SIGKILL to the process group, then any GPU PIDs are killed, then we wait until every GPU uses < 1 GB.

S1 flags:
- the FP8 repack;
- TP4, mp executor;
- dtype auto, safetensors;
- `--language-model-only`;
- max_model_len 106,496;
- 16 seqs;
- 4,096 batched tokens;
- memory 0.90;
- custom all-reduce off;
- seed 20261005;
- attention backend FLASHINFER if S1's `-lcuda` link probe passes, else TRITON_ATTN (S1's own rule);
- `fp8_e4m3` KV;
- MTP 3;
- prefix caching.

`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` is inherited from CELL 1.

Two frontend-only additions on every arm. Neither changes the engine's computation.
- `--enable-prompt-tokens-details`: cached tokens appear in the usage chunk.
- `VLLM_SERVER_DEV_MODE=1`: enables `POST /reset_prefix_cache`.

## Arms

Each arm changes only its own switch.

| arm | change from S1 | role |
|---|---|---|
| D | none (S1 exactly) | the production engine |
| E | `--no-async-scheduling` | the cheapest workaround for the prime suspect: the async-scheduling race (#53919, num_accepted_tokens reordered on row moves) |
| B | prefix caching OFF | MTP noise calibrator |
| K | `--kv-cache-dtype auto` (bf16 KV); relaunched at max_model_len 65,536 only if the KV check fails | fp8 KV question |
| C | MTP OFF | prefix caching without MTP |
| A | MTP OFF, prefix caching OFF | pure batch-noise calibrator / reference |

Priority: D, E, B, K, then C, A.

| notebook | default commit profile | arms (first, then second) | account |
|---|---|---|---|
| `Q27-XHIGH-E19_MTPPC_P1_D_C.ipynb` | E19P1 | D, then C (22 min reserved) | foysalemonshanto |
| `Q27-XHIGH-E19_MTPPC_P2_E_A.ipynb` | E19P2 | E, then A (22 min reserved) | esfersami50 |
| `Q27-XHIGH-E19_MTPPC_P3_B_K.ipynb` | E19P3 | B, then K (18 min reserved) | hosen42 |

Budget for each notebook:
- 60 min total, hard stop at 57 min.
- The first arm must end by `HARD_STOP - 90 s - the reserve for the second arm`.
- An arm is not launched with less than 14 min before its deadline.

The `E19ALL` profile runs all six arms on one account (about 2.5 h). It is for the local harness.

## Probes

Every arm sends identical token ids.

The prompts are 12 real Poetiq prompts (iteration 0) for the 12 Experiment 14 gate eval tasks, 3.3k to 9.4k tokens. They are rendered by S1's own code and tokenized as S1 does (xhigh). All requests are greedy (temperature 0). Each request uses:
- logprobs 2 keyed by token id;
- `return_token_ids`;
- `stream_options.include_usage`.

Stages, in order:

1. **serial**: 6 prompts, every other one by length. Each runs one at a time: reset, cold (1,024 tokens), warm (the same prompt: a prefix hit), then for prefix-ON arms: reset, cold2 (A/A).
2. **churn**: the 6 serial prompts × 4 = 24 requests, up to 14 at once.
   - max_tokens is staggered over 256 / 512 / ... / 2,048.
   - Arrivals are staggered by 0.5 s and refilled as requests finish, so rows join, finish and move continuously.
   - Each row is compared with the same prompt run alone in the same arm: serial warm when the churn row was a cache hit, else serial cold.
3. **long8k** (the other 6 prompts) and **long30k** (the 6 serial prompts), 6 at once.
   - natural = prompt + a deterministic filler (the task's own grids written out inside `<think>`). It ends 256 tokens before a 1,600-token block boundary. Greedy 1,024.
   - cont = prompt + filler + the server's own first 512 tokens. Greedy 512. It must reproduce natural's tokens 512 to 1,024.
4. **/metrics every 10 s**: MTP accepted / draft tokens per window, preemptions and running requests.

Recorded per request:
- token ids;
- top-1 / top-2 margins;
- positions where the emitted token is not its own logprob top-1;
- finish_reason and [DONE];
- prompt_tokens, cached tokens, TTFT, generated tokens, seconds;
- the decoded text (full outputs kept).

Every row is appended to `q27x/e19/e19_rows.jsonl` with flush + fsync as soon as its request ends. `e19_state.json` is replaced atomically after every row, so a crash keeps all finished probes.

## Verdicts

The rules are in `src/e19_judge.py`, the notebook's E19-J cell. `e19_compare.py` uses the same file.

**Large-margin flip:** at the first token divergence, a known top-1 minus top-2 margin is > 1.0 nat. Low-margin divergences are tolerated: greedy + MTP is not bit-exact across batch compositions.

**Corrupt row:** any of these:
- empty output;
- degenerate text;
- 64+ identical token ids in a row;
- 8+ CJK characters;
- a greedy token that is not the top-1.

**In-arm FAIL** (the main discriminator), if any of these:
- a cache-hit / churn / continuation row is corrupt while its partner is clean;
- cold vs warm or natural vs continuation has more large-margin flips than the arm's own serial A/A;
- an MTP arm has a /metrics window (≥ 300 draft tokens) with acceptance < 0.10.

**Churn verdict:** calibrated by arms A and B.
- A qualifies if it is not FAIL and its churn flips are ≤ its own A/A flips.
- B qualifies if it is not FAIL and its churn flips are ≤ A's.
- An arm FAILS if its churn flips exceed the other qualified calibrators' churn flips, or its corrupt rows exceed theirs.
- Without a calibrator, the arm is judged strictly against its own serial A/A. This is marked "uncalibrated".

**The token comparison against arm A is informational only.**

**Statuses** (worst wins): FAIL > LAUNCH_FAILED > INCOMPLETE (server death / errors / missing margins) > NOT_TESTED_BUDGET (planned rows not run; never PASS) > INCONCLUSIVE (cache not exercised, reset ineffective, no MTP windows) > PASS.

The three accounts run different arms. Calibration across them happens offline:

```
python src/e19_compare.py <out_P1> <out_P2> <out_P3 or Q27X_E19_BUNDLE.zip> --out <dir>
```

## Files

- `Experiment Notebooks/Build/build_e19.py`: builds the three notebooks (LF, < 400 kB, verbatim-cell asserts, `_REBUILD` guard for pushed shas).
- `Experiment Notebooks/Build/make_push_e19.py`: writes the push folders under `D:/ARC AGI 2/_scratch/push_e19/`. It does not push.
- `src/e19_judge.py`, `src/e19_core.py`, `src/e19_run.py`: the E19 cells.
- `src/e19_compare.py`: cross-account verdicts.
- `src/harness/fake_vllm019_server.py`, `src/harness/harness_e19_nb.py`: the CPU harness.
  - Scenarios: normal, launchfail, death, corrupt, race, collapse, kvretry, budget, rerun, split.
  - Logs: `src/harness/h_<scenario>.txt`.

## Validation (local CPU harness, 2026-10-27)

All scenarios pass on Python 3.14. The real notebook cells run against the fake vLLM 0.19 server.

normal and death also pass on Python 3.12.10 with transformers 4.57.6 / tokenizers 0.22.2, the Kaggle image's stack (`h312_*.txt`). This is not yet the Kaggle CPU kit.

Notebook shas: P1 `ad7f1eb5`, P2 `f17eac9e`, P3 `e46c1430`. Each is 154 kB.

| scenario | checks | what it proves |
|---|---|---|
| normal (6 arms) | 47 | flags = S1 + each arm's switch; identical prompt ids across arms; caches exercised (warm / continuation hits on every prefix-ON row, no cold hits, generated tokens inside C's continuation hits); reset before every cold row; churn = 24 requests, 14 at once, 8 lengths, refilled arrivals; benign near-tie divergences tolerated; churn calibrated by A + B; MTP windows recorded; E logs async disabled; servers stopped between arms; every row records prompt / cached tokens, TTFT, generated tokens, finish_reason |
| launchfail | 22 | B LAUNCH_FAILED, GPU wait, the other arms PASS, calibration falls back to A |
| death | 22 | D INCOMPLETE, its finished rows kept, the rest `server_dead`, later arms PASS |
| corrupt (D only) | 21 | D FAIL (corrupt cache-hit rows + large-margin flips), the others PASS |
| race (MTP + async arms) | 22 | D, B, K FAIL; E, A, C PASS; B disqualified as a calibrator; serial rows untouched |
| collapse (D) | 21 | D FAIL on the MTP acceptance windows alone |
| kvretry | 21 | K relaunched at 65,536, PASS |
| budget | 10 | D cut mid-arm (partials kept) and C not launched: both NOT_TESTED_BUDGET, nothing PASS |
| rerun | 3 | no server, identity submission |
| split | 11 | the three account notebooks with their default profiles + `e19_compare.py` over two folders and one bundle zip: six arms, all PASS, calibrated |

## Caveats

- **Masking after a benign divergence.** The flip rule judges only the FIRST divergence of a pair. A benign low-margin divergence early in a row hides any later corruption in that row from the comparison. The row-level corruption flags (degenerate text, CJK, token runs, greedy-not-top-1) and the MTP acceptance windows still cover it. In the race scenario only 1 of 24 churn rows in D showed the injected flip, because the fake's near-tie rate (1 / 23 positions) is high.

- **Greedy, not S1's temperature 1.0 sampling.** Greedy exercises the same engine paths and makes rows comparable token by token.
- **Mamba cache alignment with MTP.** With `mamba_cache_mode=align` and MTP, decode steps rarely end on a block boundary (E18 finding). So D / E / K continuation hits mostly cover prompt + filler. Arm C (MTP off) exercises hits over server-generated tokens.
- **Filler alignment.** It assumes the 1,600-token block of fp8 KV (logged by every S1 run). bf16 KV (800) stays aligned.
- **Server vs in-process.** The server runs the same V1 engine core as S1's in-process `LLM()`. The async-scheduling state is read from each server's log (`async_lines` in the state).
- **Commit mode only.** On a competition rerun the notebook only writes CELL 0's identity submission.
