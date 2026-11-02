# Suffix-decoding / MTP hybrid replay on the real ARC agent workload (offline, CPU only)

Date 2026-11-01. I used no GPU, no Kaggle and no git, and I modified no existing files. I never read or printed raw text. Every number comes from scripts that tokenize and aggregate.

**Report file not written.** The harness refused me writing `REPORT_suffix_replay_2026-11-01.md` ("Subagents should return findings as text, not write report files"). The full report is below, so please save it as that file in the folder listed under "Files". The scripts and `results_suffix_replay.json` are already written.

## Verdict: NO-GO for building the hybrid as a throughput lever
- On our workload suffix proposals are weak.
- A calibrated hybrid gains +0.9 % tokens/step at D=24, +0.6 % at D=8 and -0.1 % at D=3. It verifies about 13 % more tokens per step (4.54 vs 4.00 at D=24).
- A hybrid gated by segment (suffix only in code, grids, tool-call code and protocol) gains at most +1.5 %.
- An oracle that knows each suffix draft's true accepted length in advance reaches only +8 %.
- The raw tree score is badly over-confident. A naive rule "suffix when its score beats MTP's 1.88" loses 15 %.

## Data and method
- **Workload:** the E26 K100Q run (klee100, 30 tasks x 2 branches, multi-turn Duck harness).
  - Files: `D:/ARC AGI 2/_scratch/e26_gpu_out/q27-xhigh-e26-k100-quality/q27x/phase_Q_K100_FP8_K3/duck/<task>_b<k>/nb5_requests.jsonl` and `prompts/prompt.log`.
  - Each request carries reasoning, content, tool_calls, t0, t1 and completion. From `prompt.log` I used only the SYSTEM and USER blocks.
  - That gives 60 branches, 741 requests with outputs, 2,536,647 generated tokens and 309,698 prompt-segment tokens.
- **E27 and E25:** E27 has no `nb5_requests.jsonl`. E25 was not used. E26 is the only run where I found per-request model outputs of the real harness by name and keys.
- **Tokenizer:** the Flash-Next `tokenizer.json` at `_scratch/e13_rc2_local/kit_tmp/tmp_main/fn_tok/`. It is not in the e25 klee kit folder.
  - I rendered the stream as reasoning + `</think>` + content + Qwen XML tool call, then tokenized it.
  - Check: rendered tokens / reported completion tokens = 0.9997 (2,536,647 vs 2,537,367).
- **Segments:** token labels from line classes at character level, mapped through token offsets.

  | segment | share of tokens |
  |---|---|
  | reasoning prose | 87.5 % |
  | code (fenced or code-like lines) | 3.0 % |
  | numbers / grid lines (digit-dense) | 2.0 % |
  | tool-call code (parameter values) | 6.7 % |
  | tool/JSON protocol (wrapper tokens, `</think>`) | 0.8 % |
  | content / final answer | 0.09 % |

- **Suffix simulation:** my own pure-Python reimplementation (`sim_suffix.py`), run causally.
  - Requests run in wall-clock t0 order.
  - The per-request tree holds the prompt segments of that turn, all earlier turns of the branch, and the tokens generated so far.
  - The global tree holds responses only, those with t1 <= this request's t0.
  - The index is a hash chain, not a trie. It keeps the last 48 occurrences of the 3-gram and of the 8-gram ending at the context end. The backward match length l (cap 24) is measured exactly.
  - Up to 4 distinct match lengths m are tried. For each, the occurrence group {l >= m} is walked greedily: take the most frequent continuation, cumulative probability = product of count / group size, stop below 0.1.
  - Draft length <= min(D, m x spec_factor 1.0, 24 - m), to mimic the depth-24 tree.
  - Score = sum of cumulative probabilities. The best m and the best tree (local or global) win. Matches under 3 tokens are not proposed.
  - A step is the real verify walk: the draft predicts gen[p..], accepted a = leading exact matches, then p += a+1 (bonus token). Pure-suffix verify tokens per step = draft length + 1.
  - I also ran a position probe at every 8th position with no jumping, for unbiased per-position numbers.
- **MTP baseline:** K3 with conditional acceptance 0.864 / 0.731 / 0.607.
  - That gives 1.879 expected accepted drafts, 2.879 tokens/step and 4 verify tokens/step.
  - It reproduces the stated mean accepted length of about 2.8.
  - The same rates are applied in every segment. MTP outcomes are sampled with a fixed seed in the hybrid walk.
- **Hybrid rules:** use suffix when its estimated accepted drafts exceed 1.879, otherwise MTP.
  - `raw`: the estimate is the tree score.
  - `cal`: the tree score is mapped to the mean actual accepted length measured in the pure-suffix walk (10 quantile bins, in-sample).
  - `orc`: oracle, uses the true accepted length (upper bound).

## 1. Suffix decoding alone (walk, 2.54 M tokens)

| D | tokens/step | verify tokens/step | steps with proposal | mean draft length when proposing | mean accepted when suffix proposes |
|---|---|---|---|---|---|
| 3 | 1.374 | 2.721 | 58.7 % | 2.93 | 0.638 |
| 8 | 1.402 | 3.249 | 57.9 % | 3.88 | 0.694 |
| 24 | 1.404 | 3.283 | 57.9 % | 3.94 | 0.698 |

- Pure suffix gives 1.40 tokens/step against MTP's 2.88.
- Sensitivity run with earlier-turn reasoning removed from the local tree: 1.372 / 1.401 / 1.403 tokens/step. That is no change, so the matches do not come from re-read earlier reasoning.
- Position probe (every 8th position, over 277k positions in prose alone, more in total):
  - Proposals appear at 69.9 % of positions.
  - Mean accepted when proposing is 0.852 (D=3), 1.176 (D=8) and 1.221 (D=24).
  - Mean draft length is 2.95 / 4.70 / 4.95.
  - Mean tree score when proposing is 3.8 to 6.9 per segment, while actual accepted is 0.96 to 5.5, so the score over-states roughly 3x in prose.

### Per segment, D=24 (walk)

| segment | steps | tokens/step | proposal share | accepted per proposal |
|---|---|---|---|---|
| reasoning prose | 1,641,710 | 1.352 | 57.9 % | 0.607 |
| code | 46,524 | 1.615 | 59.6 % | 1.032 |
| numbers / grid | 22,750 | 2.193 | 86.3 % | 1.382 |
| tool-call code | 89,764 | 1.886 | 49.0 % | 1.809 |
| tool/JSON protocol | 4,711 | 4.498 | 76.9 % | 4.735 |
| content / final answer | 1,818 | 1.195 | 39.7 % | 0.490 |

- D=8 and D=3 per-segment tables are in `results_suffix_replay.json`.
  - D=8: protocol 4.35 tokens/step, tool-call code 1.87, grids 2.18, code 1.61, prose 1.35.
  - D=3: protocol 3.19, tool-call code 1.72, grids 2.00, code 1.55, prose 1.33.
- Only protocol tokens (0.8 % of the stream) beat MTP's 2.88 on their own.
- Per turn, D=24 tokens/step: turn 0 1.445, turn 1 1.404, turns 2-3 1.411, turns 4-7 1.383, turns 8+ 1.421.
  - Proposal share is 52 % to 59 %.
  - Accepted when proposing goes from 0.86 at turn 0 to 0.67-0.72 later. There is no real trend in tokens/step.
- Position probe per segment, D=24 (accepted when proposing / proposal share):
  - prose 0.96 / 68.9 %
  - code 1.79 / 75.0 %
  - grids 2.11 / 93.4 %
  - tool-call code 3.19 / 71.5 %
  - protocol 5.46 / 95.2 %
  - final answer 0.66 / 46.9 %

## 2. Hybrid (walk, MTP outcomes sampled)
MTP baseline: 2.879 tokens/step, 4.00 verify tokens/step, 0.720 tokens per verify token.

| rule | D | tokens/step | vs MTP | verify tokens/step | tokens per verify token | steps taking suffix | step share MTP / local tree / global tree |
|---|---|---|---|---|---|---|---|
| raw | 3 | 2.438 | -15.3 % | 3.999 | 0.610 | 45.9 % | 54.1 / 29.9 / 16.0 |
| raw | 8 | 2.434 | -15.4 % | 4.830 | 0.504 | 49.7 % | 50.3 / 28.2 / 21.5 |
| raw | 24 | 2.441 | -15.2 % | 4.896 | 0.499 | 49.6 % | 50.4 / 28.0 / 21.6 |
| cal | 3 | 2.876 | -0.1 % | 4.000 | 0.719 | 0.0 % | 100 / 0 / 0 |
| cal | 8 | 2.897 | +0.6 % | 4.473 | 0.648 | 11.8 % | 88.2 / 7.6 / 4.1 |
| cal | 24 | 2.905 | +0.9 % | 4.544 | 0.639 | 11.9 % | 88.1 / 7.6 / 4.3 |
| oracle | 3 | 2.997 | +4.1 % | 3.998 | 0.750 | 15.2 % | 84.8 / 12.2 / 3.0 |
| oracle | 8 | 3.102 | +7.8 % | 4.290 | 0.723 | 13.1 % | 86.9 / 9.6 / 3.5 |
| oracle | 24 | 3.109 | +8.0 % | 4.327 | 0.718 | 13.0 % | 87.0 / 9.5 / 3.5 |

**Verify cost (expected verify tokens per step) at D = 3 / 8 / 24:**
- raw: 4.00 / 4.83 / 4.90
- calibrated: 4.00 / 4.47 / 4.54
- oracle: 4.00 / 4.29 / 4.33
- Per verify token the hybrid is at or below MTP-only (0.72) in every realistic case. Only the oracle at D=3 and D=8 is higher.
- With the calibrated rule at D=24, 13.6 % more tokens are verified per step for +0.9 % tokens per step.

**Which segments gain** (calibrated, D=24, tokens/step against 2.879):

| segment | tokens/step | vs MTP | suffix share of steps |
|---|---|---|---|
| reasoning prose | 2.859 | -0.7 % | 10.8 % |
| code | 3.018 | +4.8 % | 18.4 % |
| numbers / grids | 3.099 | +7.7 % | 31.3 % |
| tool-call code | 3.323 | +15.4 % | 16.8 % |
| tool/JSON protocol | 4.768 | +65.6 % | 42.8 % |
| final answer | 2.811 | -2.3 % | 5.7 % |

- Per turn (calibrated, D=24): 2.97 / 2.92 / 2.90 / 2.90 / 2.91 tokens/step for turn 0 / 1 / 2-3 / 4-7 / 8+. Suffix share is 11-12 % throughout.
- Oracle D=24: prose +5.7 %, code +15.2 %, grids +24.1 %, tool-call code +28.6 %, protocol +99.8 %.
- Segment-gated hybrid (calibrated suffix only in code, grids, tool-call code and protocol; MTP forced elsewhere; computed from the per-segment results):
  - +1.4 % at D=8 and +1.5 % at D=24.
  - Verify is 4.10 to 4.12 tokens/step.
  - Oracle-gated is +2.7 %.

**Why the gain is small:**
- 87.5 % of generated tokens are reasoning prose. There suffix proposals average 0.6 accepted drafts against MTP's 1.88.
- The segments where suffix clearly wins (tool-call code, grids, protocol) are only 9.5 % of tokens.
- The calibration table at D=24 shows only the top decile of tree scores has an actual mean accepted length (1.98) above MTP's 1.88. Every other score bin is 0.20 to 0.89.

## Caveats
- **Approximation of Arctic:** my own approximation, not the real library.
  - It caps recency at 48 occurrences per anchor.
  - It uses only 3-token and 8-token anchors.
  - It tries at most 4 match lengths.
  - It proposes nothing below a 3-token match.
  - It walks a group instead of a true count-branching tree.
  - The recency cap most likely understates suffix on frequent anchors such as code and protocol.
- **MTP rates:** I assumed identical MTP acceptance in every segment. In code, grids and protocol MTP is probably above average, which would shrink the hybrid gain exactly where it shows up.
- **Calibration:** it is in-sample (fit on the pure-suffix walk, applied to the hybrid walk over the same data). Out-of-sample would be slightly worse. The oracle is an upper bound, not achievable.
- **Prompt context:** approximate. Blocks from `prompt.log` are assigned by the count of assistant markers before them. Earlier-turn outputs are re-rendered from my own tokenization, not the exact server prompt. The no-reasoning variant shows no sensitivity. Later-turn tool results in `prompt.log` may be incomplete.
- **Segment labels:** heuristic line regexes. Final answer is tiny because turns end in tool calls.
- **Global tree timing:** the global tree takes a response when it completes (t1), so concurrent streams only see each other's completed responses.
- **Single run:** one MTP seed, one dataset, 30 tasks.
- **Verify cost:** counted in tokens only. Real 4xL4 cost also depends on kernels with ragged draft lengths and the MTP-head cost, which are not modeled.

## Files
- **Folder:** `D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Results/prep_0gpu/suffix_replay/`
- **`prep_tokens.py`:** tokenizes and labels. It writes `D:/ARC AGI 2/_scratch/prep_suffix/data.pkl`.
- **`sim_suffix.py`:** the replay.
  - Phase 1: `python sim_suffix.py 1 full p1_full.json` and `... 1 noreason p1_noreason.json`.
  - Copy `p1_full_cal.json` to `p2_full_D<D>_cal_in.json`.
  - Phase 2: `python sim_suffix.py 2 full p2_full_D<D>.json 1000000 <D>` for D in 3, 8, 24.
  - The intermediate JSONs are under `D:/ARC AGI 2/_scratch/prep_suffix/`.
  - Phase 1 takes about 20 min. Phase 2 takes about 15 min per D, with the three D values run in parallel.
- **`summarize.py`:** writes `results_suffix_replay.json`, which holds all tables above, per segment, per turn, the calibration and the step shares.
- **Missing:** `REPORT_suffix_replay_2026-11-01.md` (see the top of this message).