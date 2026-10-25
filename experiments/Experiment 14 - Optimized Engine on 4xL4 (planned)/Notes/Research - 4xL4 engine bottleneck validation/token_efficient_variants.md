# Token-efficient variants and settings of Qwen3.8-27B for the 12 h / 240-task ARC-AGI-2 run

Research date: 2026-10-23. Tags: **[M]** measured (by us or by a named third party, with its setup), **[C]** claimed by the vendor or author and not reproduced, **[I]** my inference.
Project rule: the thinking floor is 60,000 tokens (user rule, 2026-10-18; [memory](C:/Users/hp/.claude/projects/d--ARC-AGI-2/memory/no-token-cuts-below-base.md)). Nothing below proposes cutting the base model's budget. The token-efficient **models** are listed as an option for the user to decide on.

Scratch evidence produced for this note (byte-level checks of HF checkpoints and the chat template): `D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/hf/` and `.../scratchpad/scripts/mtpcmp.py`, `.../scratchpad/scripts/maincmp.py`.

## 1. Each variant's card: token reduction, accuracy, effort levels, method, independent replication

### Takeaway
- **Vendor claims:**
  - Swift 1.0: about 41 % fewer mean thinking tokens on GPQA at xhigh (median 58 %).
  - Swift 1.5: 42 % mean (median 58.5 %).
  - ThinkingCap-Qwen3.8: 37 % mean (median 43 %), for under 1 point of average accuracy.
  - All three lose 2.7–4.7 points on AIME 2026, the hardest math test.
- **The savings shrink as the base model's traces get longer.** Long-trace benchmarks such as Terminal-Bench (52–73k tokens) save only 11–16 % on mean. ARC xhigh is in that long-trace regime at about 60k tokens per request.
- No card reports HLE or any ARC-AGI-1/2 result.
- Independent replication is thin and mixed. I found no controlled third-party re-run of either vendor's benchmarks.

### Cited Findings

**ukisai/Swift-Qwen3.8-27b (Swift 1.0)** [C] — [card](https://huggingface.co/ukisai/Swift-Qwen3.8-27b)
- **Method:** identify reasoning-marker tokens that "trigger overthinking" and fine-tune while penalizing them. It "also includes a transfer component derived from BottleCap AI's ThinkingCap-Qwen3.6-27B". The card does not give RL details or data. Training ran on 8×H100 (NVIDIA Innovation Lab).
- **BF16 results, base → Swift (mean / median thinking-token reduction):**

  | Benchmark | Base → Swift | Mean ↓ | Median ↓ |
  |---|---|---|---|
  | GPQA-Diamond | 88.38 → 88.28 | 41.0 % | 58.3 % |
  | MMLU-Pro | 85.47 → 84.95 | 46.2 % | 28.3 % |
  | AIME 2026 | 98.67 → 94.00 | 26.7 % | 50.2 % |
  | HMMT Nov 2025 | 99.33 → 96.00 | 31.1 % | 45.9 % |
  | LiveCodeBench v6 | 76.76 → 81.55 | 24.3 % | 45.8 % |
  | Terminal-Bench 2.1 | 66.74 → 65.84 | 26.5 % | 38.7 % |
  | IFBench | 73.53 → 71.80 | 42.2 % | 50.5 % |
  | ERQA | 67.45 → 66.30 | 50.6 % | 54.6 % |

- **Setup:** 5 seeds, T 1.0, top_p 0.95, top_k 20, output caps 16,384–250,000 depending on the benchmark.
- **Effort levels:** mean thinking reduction is 41.0 % at xhigh, 22.7 % at medium and 25.8 % at low. On GPQA, the card notes that Swift at xhigh still uses about 2× the tokens of the base at medium.
- **Weights:** the card says "the same base plus the Swift adapter" and the repo carries a `lora` tag. The repo itself holds 18 merged BF16 shards (55.6 GB) with no adapter files [M] ([tree](https://huggingface.co/ukisai/Swift-Qwen3.8-27b/tree/main)).

**ukisai/Swift-1.5-Qwen3.8-27b** [C] — [card](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b)
- **Method:** tokens linked to "pathological overthinking" are penalized "without directly capping reasoning length". Accuracy is then recovered with "RL and OPD", and post-training focuses on long-horizon, agentic and coding tasks.
  - "OPD" is not expanded on the card. MindStudio reads it as on-policy distillation ([MindStudio](https://www.mindstudio.ai/blog/qwen3-8-27b-thinking-fine-tunes-comparison)); HackerNoon reads it as "outcome-based preference optimization" ([HackerNoon](https://hackernoon.com/the-qwen38-27b-variant-built-to-stop-overthinking)). This is unresolved.
  - Data: `ukisai/Qwen3.8-27B-multi-turn-agent-sft`, re-sampled into RL environments.
- **BF16 results via vLLM 0.27.1, xhigh, 5 seeds, base → Swift 1.5:**

  | Benchmark | Base → Swift 1.5 | Mean ↓ | Median ↓ | Mean tokens |
  |---|---|---|---|---|
  | GPQA-D | 88.28 → 88.59 | 41.9 % | 58.5 % | 15,014 → 8,717 |
  | AIME 2026 | 98.67 → 96.00 | 40.0 % | 48.5 % | 22,014 → 13,203 |
  | HMMT Nov 2025 | 99.33 → 97.33 | 32.1 % | 47.8 % | — |
  | LiveCodeBench v6 | 76.76 → 81.71 | 24.5 % | 46.3 % | — |
  | Terminal-Bench 2.1 | 69.21 → 72.13 | 16.3 % | 0.1 % | 52,265 → 43,733 |
  | ERQA | 67.45 → 65.40 | 53.9 % | — | — |
  | IFBench | 73.53 → 72.07 | 38.5 % | — | — |

  Mean-token figures are as relayed in the [HackerNoon summary](https://hackernoon.com/the-qwen38-27b-variant-built-to-stop-overthinking).
- **Effort levels on GPQA, base → Swift 1.5:** xhigh 88.28 → 88.59 (−41.9 % thinking); medium 84.14 → 82.22 (−24.8 %); low 84.04 → 84.85 (−28.7 %).
- **Weights:** full merged BF16 shards, no adapter files [M] ([tree](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b/tree/main)).
- **Inconsistencies:**
  - HackerNoon says "58.5 % fewer mean" tokens, but the card's own GPQA token counts give 41.9 % ([HackerNoon](https://hackernoon.com/the-qwen38-27b-variant-built-to-stop-overthinking)).
  - MindStudio names the vendor "Yucus AI"; the card says UkisAI ([MindStudio](https://www.mindstudio.ai/blog/qwen3-8-27b-thinking-fine-tunes-comparison)).

**bottlecapai/ThinkingCap-Qwen3.8-27B** [C] — [card](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B); [blog](https://bottlecapai.com/post/thinkingcap-qwen3-8-27b/)
- **Headline:** the 12-benchmark average falls from 15,735 to 12,144 mean thinking tokens (−37.2 %), and the median from 7,148 to 4,615 (−42.9 %). Matched per-question geometric-mean Δ is −41.8 %. Accuracy goes from 86.65 % to 85.79 % (−0.86 pp).
  - Mean answer tokens are flat (1,712 → 1,754), so total tokens fall 36.5 %.
  - Truncation falls from 0.51 % to 0.34 %; looping from 0.06 % to 0.05 %.
- **Per benchmark (accuracy, mean thinking tokens):**

  | Benchmark | Accuracy | Mean thinking tokens | Reduction |
  |---|---|---|---|
  | GPQA-D | 89.93 → 88.04 | 12,772 → 7,267 | −43.1 % |
  | AIME 2026 | 98.13 → 94.27 | 15,663 → 10,934 | −30.2 % |
  | HMMT Feb 2026 | 95.83 → 94.70 | 23,211 → 18,099 | −22.0 % |
  | LiveCodeBench v6 | 91.14 → 91.21 | 28,395 → 22,645 | −20.3 % |
  | Terminal-Bench 2.1 | 75.84 → 75.28 | 72,871 → 65,092 | −10.7 % |
  | MMMLU | 85.38 → 84.09 | 1,656 → 571 | −65.5 % |

  - Seeds range from 4 to 32 (MMLU-Pro and MMMLU used 1).
  - Hardware: H200, vLLM 0.29.0, MTP k=3. MTP was checked as accuracy-neutral: AIME26 over 32 seeds gave 98.23 % without it and 98.13 % with it.
- **Effort levels,** 11 benchmarks, Δ against the base at xhigh ([blog](https://bottlecapai.com/post/thinkingcap-qwen3-8-27b/)):

  | Model and effort | Accuracy Δ | Mean thinking Δ |
  |---|---|---|
  | Base, medium | −9.16 pp | −52.1 % |
  | ThinkingCap, medium | −9.90 pp | −60.2 % |
  | Base, low | −9.71 pp | −55.4 % |
  | ThinkingCap, low | −10.79 pp | −62.3 % |
  | Base, thinking off | −28.57 pp | — |
  | ThinkingCap, thinking off | −34.25 pp | — |

  - The vendor recommends xhigh.
- **Method:** not disclosed for the 3.8 release.
  - The earlier 3.6 release used "a curated set of problems covering multiple domains and difficulty levels", and its "training objective rewarded efficient reasoning rather than simply rewarding correctness" ([3.6 blog](https://bottlecapai.com/post/thinkingcap-qwen3-6-27b/)).
  - 3.6 results: −45.8 % out-of-domain thinking for −0.7 pp; GPQA-D 85.5 → 83.8.

**Other Qwen3.8-27B efficiency tunes found**
- **bytkim/Qwen3.8-27B-pi ("Qwen Pi"), Apache-2.0** [C] ([card](https://huggingface.co/bytkim/Qwen3.8-27B-pi)):
  - Method: SFT on successful Pi-agent sessions, then GRPO with a "custom reasoning-efficiency reward". Effort levels: low, medium, xhigh.
  - Claims: at medium it matches base xhigh on Terminal-Bench 2.1 with about 41 % fewer output tokens; SciCode at xhigh about 23 % fewer tokens. No GPQA numbers in the text.
  - MindStudio calls it "only worthwhile if you run the Pi agent harness" ([MindStudio](https://www.mindstudio.ai/blog/qwen3-8-27b-thinking-fine-tunes-comparison)).
  - It is the only Qwen3.8-27B efficiency tune I found under an OSI licence (the HF API reports `license: apache-2.0`) [M].
- **Flash-Next versions** exist from both vendors, which is relevant to the E13 plan: `bottlecapai/ThinkingCap-Qwen3.8-Flash-Next`, `ukisai/Swift1.5-Qwen3.8-Flash-Next` and `ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ` [M] (HF search API; [ukisai org](https://huggingface.co/ukisai)).
- **Training-free alternative: logit penalty on overthinking markers** ("wait", "but") in quantized reasoning models [C] ([arXiv 2606.00206](https://arxiv.org/abs/2606.00206); Lotfi, Kirichenko, Li, Liu, 2026-05-29). This is the paper Darshan cites in 746149.
  - In up to 52 % of quantized failures, the right answer appeared in an intermediate step but was not output.
  - The penalty cut CoT length by 12–23 % with accuracy "preserved or improved", on 5 models of 1.5B–32B. Model names are not in the abstract.
- **MIST** (token-saliency CoT compression, [arXiv 2608.31066](https://arxiv.org/abs/2608.31066)): the abstract gives no numbers.

**Independent checks (all differ from the vendor setups)**
- **MindStudio, Swift 1.0 vs base** [M, n=1] ([post](https://www.mindstudio.ai/blog/qwen3-8-27b-swift-vs-original-thinking)):
  - Setup: one Hermes-agent coding task, vLLM on 80 GB. Quantization, effort and sampling not stated.
  - Swift used **60,572 vs 11,744** output tokens and 21 vs 4 tool calls; wall clock 35:52 vs 47:07.
  - Swift's output stalled mid-task; the base's ran end to end.
- **zephel01 (note.com), ThinkingCap GGUF** [M] ([post](https://note.com/zephel01/n/nd65298dbf391?hl=en)):
  - Setup: Q4_K_M and Q6_K, llama.cpp, T 0.35 (the card recommends 1.0), max_tokens 49,152.
  - L6 resolved 96.7 % / 98.3 %; L7 "grandmaster" 62.5 % / 75.0 %.
  - Half of the L7 failures ended mid-thought ("inference budget reached" or loop).
  - There was no same-condition base comparison, so the token claim is untested.
- **kaushikvira, NInfer NVFP4 ThinkingCap vs Swift 1.5 on one RTX 5090** [M] ([repo](https://huggingface.co/kaushikvira/Qwen3.8-27B-thinkingcap-nvfp4full-dflash2-NInfer-v3)): IFBench mean completion tokens 3,988 vs 4,805 (−17 %); GSM8K 95.5 vs 95.0. This compares two tunes, not tune vs base.
- **HF discussions on Swift 1.5, user reports** [M, anecdotal]:
  - [#4](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b/discussions/4): at IQ3_XXS with `reasoning_effort: low`, Swift 1.5 produced **20–44 % more** tokens than Swift 1.0 and often hit the 8k budget.
  - [#3](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b/discussions/3), "Totaly lobotomized": looping, ignoring fetched sources, fabrication in agentic coding. No numbers.
- **Press coverage repeats the cards:**
  - MarkTechPost notes that the 37.2 % figure is a mean of per-benchmark reductions ([MarkTechPost](https://www.marktechpost.com/2026/09/24/bottlecap-ai-releases-thinkingcap-qwen3-8-27b-37-2-fewer-thinking-tokens-at-a-0-86pp-accuracy-cost/)).
  - The Kaitchup: Terminal-Bench thinking −10.7 % becomes total tokens −7.5 % ([Kaitchup](https://kaitchup.substack.com/p/thinkingcap-qwen38-27b-less-thinking)).

### Inferences
- **[I] Savings fall with trace length,** for both vendors.
  - ThinkingCap: MMMLU (1.7k) −65 %, GPQA (12.8k) −43 %, AIME (15.7k) −30 %, HMMT-Feb (23k) −22 %, LCB (28k) −20 %, Terminal-Bench (73k) −11 %.
  - Swift 1.5: GPQA (15k) −42 %, AIME (22k) −40 %, Terminal-Bench (52k) −16 %.
  - Our ARC xhigh requests run at about 60k (cost-derived, see §2). The relevant expectation is therefore nearer **−10 to −25 % mean** than the headline 37–42 %.
  - Medians fall more than means on several benchmarks, so the savings are concentrated on easy and median items, not the long tail. On ARC the long tail is where cap hits and coverage loss live.
- **[I] Coverage arithmetic.** At fixed throughput, a −15 % to −25 % mean-token cut buys about 1.18–1.33× more task-requests in 12 h. The headline −40 % would buy about 1.67×. This is "a throughput gain" only if ARC accuracy per request holds.
- **[I] The accuracy cost lands on the hardest reasoning.** AIME 2026 loses 2.7–4.7 pp in every tune, while easy and knowledge benchmarks are flat. ARC-AGI-2 at a 42 % base solve rate is a hard-reasoning workload, so it is the case most likely to lose.
- **[I] No vendor or third party has run ARC-AGI** on any of these tunes. An EVAL run on our own harness is the only way to know.

### Gaps
- No HLE or ARC-AGI-1/2 result for any variant.
- No disclosed training method for ThinkingCap 3.8, and only a sketch for Swift.
- No independent same-condition replication of the token savings.
- The Swift and ThinkingCap per-effort figures are on general benchmarks, not ARC.

## 2. Effort levels on the base model: tokens per level, the medium < low anomaly, capped xhigh vs lower effort

### Takeaway
- **ARC Prize semi-private:** xhigh 42.4 %, low 22.8 %, medium 13.2 %, none 1.5 %. Cost-derived output is about 60k / 32k / 38k / 7k tokens per request (about 172k / 90k / 109k / 20k per task).
- **Medium is below low** because Qwen's chat template adds an instruction for xhigh and low but **none for medium**. This is confirmed by Baseten and by our byte check of the template.
- No public source measures xhigh with a capped thinking budget against a lower effort. Our own internal data (49k vs 60k) shows that capping costs LB points.

### Cited Findings
- **Scores** [M, by ARC Prize] ([arcprize.org](https://arcprize.org/results/alibaba-qwen3-8-27b)):

  | Effort | ARC-AGI-2 semi-private | ARC-AGI-1 |
  |---|---|---|
  | xhigh | 42.4 % | 87.5 % |
  | medium | 13.2 % | 68.7 % |
  | low | 22.8 % | 69.2 % |
  | none | 1.5 % | 34.0 % |

  - xhigh cost is $0.4469 per task. Pricing basis: $0.425/M in, $2.55/M out on Baseten dedicated.
  - The page says medium uses more completion tokens than low, and "Baseten reportedly confirmed that Qwen's chat template adds effort-specific instructions for low and xhigh, but not for medium".
- **Template check** [M] (our byte check of `Qwen/Qwen3.8-27B/chat_template.jinja`, identical byte for byte to Swift 1.5's template; saved under `scratchpad/hf/`):
  - Default effort: `reasoning_effort|default('xhigh')`.
  - xhigh text: "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness…".
  - low text: "Reasoning effort is set to low. Keep your thinking brief and focused…".
  - medium has no instruction branch.
- **Cost-derived tokens per level** [I from M] (local `D:/ARC AGI 2/_scratch/arcprize/effort_summary.txt`, from `evaluations.json`; assumes about 6k-token prompts and 2 attempts per test input):

  | Effort | Per task (semi-private) | Per request |
  |---|---|---|
  | xhigh | ~172k | ~60.1k |
  | medium | ~109k | ~38.2k |
  | low | ~90k | ~31.6k |
  | none | ~20k | ~7.1k |

  - The public-eval xhigh figure is about 63.7k per request.
  - Per-task public-eval solves: xhigh 40/120, medium 16, low 19, none 3; the xhigh ∪ low union is 48.
- **Other estimates are consistent:** Anil Thomas reports an xhigh median of about 64k per generation; Kris Dreemurr a median of 70k per task for pass@2 (CV 45 %) ([local summary](D:/ARC%20AGI%202/resources/Analysis%20and%20Journals/Kaggle%20discussions%20(2026-10-22)/00_SUMMARY.md), threads 745787 and 750546).
- **Our own low-effort run, W3H (low, EVAL120, 3 h, 2026-10-17)** [M] ([memory](C:/Users/hp/.claude/projects/d--ARC-AGI-2/memory/arcprize-qwen38-effort-scores.md)):
  - Solved 17/172 (9.88 %). All 15 solved tasks are a subset of the xhigh R5 tasks, so there were no complementary solves.
  - Median 30k tokens per request.
  - 78 % of low requests emitted EOS mid-thinking without `</think>`; xhigh never does this.
- **Capped xhigh, internal** [M] ([memory](C:/Users/hp/.claude/projects/d--ARC-AGI-2/memory/no-token-cuts-below-base.md)): S49 at 49k scored **LB 32.08**, below 60k's **34.58**.
- **Capped xhigh, external** [M, anecdotal] ([MindStudio local-run post](https://www.mindstudio.ai/blog/run-qwen3-8-27b-locally-reasoning-effort), search summary only, not fetched): a 4-bit local xhigh run "may exceed 50K thinking tokens without completing"; raising the budget to 64K let tasks complete.
- **General effort trade-off, BottleCap's base data** [C] ([blog](https://bottlecapai.com/post/thinkingcap-qwen3-8-27b/)): base medium −9.16 pp for −52 % thinking; base low −9.71 pp for −55 %. ThinkingCap at xhigh: −0.86 pp for −37 %.

### Inferences
- **[I] A fine-tuned short-thinker at xhigh dominates dropping the base to medium or low** on the vendor's 11-benchmark curve: about −1 pp for −37 % against about −9 pp for −52–55 %. On ARC the gap between efforts is far larger (42.4 vs 22.8 / 13.2), so dropping effort is clearly worse there.
- **[I] On ARC the effort setting is mostly a prompt instruction,** not a budget. Medium's "no instruction" state gives a weaker trace than low's "be brief" state. This is consistent with Baseten's explanation and our template check.

### Gaps
- ARC Prize publishes no direct token counts per effort; the figures above are derived from cost.
- No public measurement of xhigh with a hard thinking cap against low or medium at matched tokens.

## 3. Thinking-budget mechanisms in SGLang and vLLM; fine-tuned short-thinker vs budget truncation

### Takeaway
- **SGLang has three mechanisms:**
  - chat-template hints (no guarantee);
  - `--enable-strict-thinking` (grammar mask, exact cap);
  - a custom logit processor that forces newline + `</think>`.
- **The built-in Qwen3 processor hard-codes old token IDs,** so on Qwen3.5/3.6/3.8 hybrids it silently fails unless subclassed with 248068/248069. Strict thinking also failed in one Qwen3.6 report.
- **vLLM:** budget support is version-dependent and silently ignored without a reasoning parser.
- **Truncation vs a trained short-thinker:**
  - No head-to-head study exists on Qwen3.8.
  - The literature says hard truncation removes useful late reasoning, and our own LB data agrees (49k < 60k).
  - A trained short-thinker shortens the trace without cutting it.

### Cited Findings
- **SGLang docs** [C] ([thinking_budget](https://lmsysorg.mintlify.app/docs/advanced_features/thinking_budget)):
  - Chat-template `thinking_budget` "only has an effect if the model's chat template defines it. Otherwise it is silently ignored". `reasoning_effort` is forwarded into the template.
  - **Strict thinking:**
    - Server: `--reasoning-parser qwen3 --enable-strict-thinking` (xgrammar).
    - `/generate` needs `max_thinking_tokens` plus `require_reasoning: true`. Chat completions use `extra_body={"custom_params": {"thinking_budget": N}}`. There is a server-wide env cap, `SGLANG_MAX_THINK_TOKENS`.
    - At exhaustion only the `</think>` sequence is allowed.
    - Even without a budget, it blocks `<tool_call>`, `</tool_call>`, `<|im_end|>` and `<|endoftext|>` during thinking.
  - **Custom logit processor:** `--enable-custom-logit-processor`. At the budget it forces a newline and then `</think>`. The built-ins (`Qwen3ThinkingBudgetLogitProcessor`, GLM, DeepSeek-R1, Inkling) have hard-coded IDs.
  - The docs advise verifying with a temperature-0 A/B. They do not cover speculative decoding.
- **SGLang #25536 (closed, no upstream fix)** [M by reporter] ([issue](https://github.com/sgl-project/sglang/issues/25536)):
  - Setup: Qwen3.6-27B-FP8, SGLang 0.5.11–0.5.12. With `thinking_budget: 200`, the model produced about 1,350–1,430 reasoning tokens.
  - The built-in Qwen3 processor uses IDs 151667/151668; Qwen3.5/3.6 use **248068/248069**.
  - `--enable-strict-thinking` "did not enforce the budget".
  - Speculative decoding ran thinking tokens "before the main model's processor could intervene" (reporter's analysis).
  - Workaround: subclass with START=248068, END=248069, NEWLINE=198 and send it via dill as `custom_logit_processor`. A later commenter confirmed it works.
- **vLLM:**
  - A user on vLLM 0.18.0 found `thinking_token_budget` "silently ignored when passed via extra_args" ([forum](https://discuss.vllm.ai/t/thinking-token-budget-silently-ignored-when-passed-via-extra-args-in-vllm-0-18-0/2533), search summary).
  - PR #37112 (`reasoning_budget`, forces end-of-think) was closed. It notes the budget is "silently ignored when no reasoning parser is configured" ([PR #37112](https://github.com/vllm-project/vllm/pull/37112), search summary).
  - A third-party logits-processor plugin exists ([phishingupstream/vllm-thinking-budget](https://github.com/phishingupstream/vllm-thinking-budget), not fetched).
  - I could not confirm a first-class flag in current vLLM docs.
- **Literature on truncation vs trained brevity** (search summaries; full papers not read):
  - [arXiv 2605.07316](https://arxiv.org/html/2605.07316v1): accuracy on hard math "rises steadily as the allowed response length grows". Length-penalty methods compress more, but "gains in efficiency are often accompanied by noticeable performance degradation".
  - [arXiv 2603.08462](https://pith.science/paper/2603.08462): uniform length penalties and hard truncation "penalize essential reasoning steps just as much as redundant filler". A surprisal-based penalty claims about 48 % fewer tokens at matched accuracy.
  - No source runs a head-to-head on the same model.
- **Vendor framing:**
  - Swift 1.5 penalizes overthinking tokens "without directly capping reasoning length" ([card](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b)).
  - BottleCap 3.6: the model "learns to stop once it has enough information to answer confidently" ([blog](https://bottlecapai.com/post/thinkingcap-qwen3-6-27b/)).
  - ThinkingCap at xhigh beats the base at medium or low on the vendor's curve (§2).

### Inferences
- **[I] Budget truncation of the base model is excluded** by the 60k floor rule, and our 49k-vs-60k LB data argues against it anyway.
- **[I] A trained short-thinker at xhigh with the same 60k cap** keeps the rule intact. Tasks the tune finishes early free tokens and slots; hard tasks keep the full 60k.
- **[I] Speculative decoding and a logit-processor budget may conflict** in SGLang (#25536). Any forced-`</think>` mechanism used as a safety cap at 60k needs a temperature-0 A/B with DFlash2 on.

### Gaps
- No Qwen3.8-specific accuracy-vs-budget curve.
- No test of whether `--enable-strict-thinking` works with DFLASH in SGLang 0.5.21.

## 4. Serving compatibility: INT4 W4A16 builds, DFlash2 acceptance on a fine-tuned target, the MTP head

### Takeaway
- **INT4 W4A16 builds exist for both tunes.** `ukisai/Swift-1.5-Qwen3.8-27b-INT4` has the same format and layout as our `RedHatAI/Qwen3.8-27B-INT4`: compressed-tensors pack-quantized, g128, `actorder: static`, separate `model_mtp.safetensors`, 19.48 vs 19.47 GB.
- **The MTP head in both Swift releases is byte-identical to base** (kept, not retrained). The target trunk under it changed, so MTP acceptance can still shift.
- **DFlash2 (base-trained) runs on Swift 1.5 and ThinkingCap** with acceptance length about 3.6–4.2. No same-condition base comparison exists. Our base INT4 on L4 measured AL about 3.8–4.5 (median 4.55).

### Cited Findings
- **INT4 / W4A16 repos** [M] (HF API `config.json` reads, 2026-10-23):

  | Repo | Format | Size | Notes |
  |---|---|---|---|
  | `ukisai/Swift-1.5-Qwen3.8-27b-INT4` | compressed-tensors pack-quantized, 4-bit g128, `actorder: static` | 19.48 GB | ships `model_mtp.safetensors`; same layout as `RedHatAI/Qwen3.8-27B-INT4` (19.47 GB, same scheme) |
  | `ukisai/Swift-1.5-Qwen3.8-27b-W4A16-AWQ` | compressed-tensors, g128 | 19.58 GB | |
  | `ukisai/Swift-1.5-Qwen3.8-27b-W4A16-AutoRound` | auto-round, 4-bit g128 sym | — | keeps `in_proj_a/b`, `mtp.*`, `visual.*` at 16-bit |
  | `bjonor/Swift-1.5-Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` | gptq | — | |
  | `wasifb/ThinkingCap-Qwen3.8-27B-AutoRound-W4A16` | auto-round, g128 | 19.02 GB | |
  | `halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin` | compressed-tensors, mse observer | 17.05 GB | has `model-mtp.safetensors` and `kv_scales.safetensors` |
  | `Tomeamis/ThinkingCap-Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` | GPTQ (per name) | — | gated |

  - BottleCap itself ships only FP8, NVFP4, NVFP4A4-AWQ, GGUF and MLX ([card](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B)).
  - The Swift 1.0 card has quantized rows: W4A16 mixed GPQA 88.38 vs base 88.69; AWQ INT4 AIME 84.00 vs 82.67 ([card](https://huggingface.co/ukisai/Swift-Qwen3.8-27b)).
- **`ukisai/Swift-1.5-Qwen3.8-27b-INT4` card** [C] ([card](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-INT4)):
  - Built with LLM Compressor (AWQ + GPTQ). "The runtime must honor the saved FP8 KV-cache configuration."
  - Single-seed results:

    | Model | GPQA-D | IFBench | AIME 2026 |
    |---|---|---|---|
    | Swift 1.5 BF16 | 91.41 | 72.00 | 86.67 |
    | Swift 1.5 INT4 | 90.91 | 70.00 | 83.33 |
    | Base W4A16 reference | 86.36 | 72.00 | 76.67 |

  - Tokens vs the base reference: GPQA mean −45.8 % / median −64.4 %; AIME mean −22.4 % / median −34.0 %.
  - The card states that these "do not establish quality parity".
  - Serving: SGLang `--model-path ukisai/Swift-1.5-Qwen3.8-27b-INT4`; the evaluation used vLLM 0.29.0.
- **MTP head byte check** [M] (our HTTP range reads of the safetensors headers and data; script `scratchpad/scripts/mtpcmp.py`):
  - All 15 `mtp.*` tensors in `model-00018-of-00018.safetensors` hash identically in `Qwen/Qwen3.8-27B`, `ukisai/Swift-Qwen3.8-27b` and `ukisai/Swift-1.5-Qwen3.8-27b`. Small tensors were hashed in full; large ones over their first 2 MB.
  - Trunk tensors differ ([script](D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/scripts/maincmp.py)):
    - Swift 1.0 changed `layers.0` and `layers.63` MLP weights but kept `embed_tokens`, `lm_head` and `layers.5.linear_attn.out_proj` identical to base.
    - Swift 1.5 changed all of them.
    - The vision tower and `layers.61.input_layernorm` are identical in all three.
  - The card says the "published weights include the base model's MTP head" and does not say it was retrained ([card](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b)).
  - ThinkingCap is gated, so I could not byte-check it. Its card says "Qwen's original MTP drafter as well as" DFlash 2 "work with ThinkingCap out of the box" ([card](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B)).
- **Acceptance on fine-tuned targets:**

  | Target | Engine, setup | Method | Accept length | Accept rate | Source |
  |---|---|---|---|---|---|
  | ThinkingCap | vLLM 0.29.0, xhigh, H200 | MTP k=3, top-token | 2.5–3.0 | — | [card](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B) [C] |
  | ThinkingCap | same | MTP k=3, probabilistic | 2.8–3.3 | — | same |
  | ThinkingCap | same | DFlash2, 7 drafts, top-token | 3.1–4.2 | — | same |
  | ThinkingCap | same | DFlash2, 7 drafts, probabilistic | 3.3–4.9 | — | same |
  | Swift 1.5 AutoRound W4A16 | vLLM 0.31.0, 24-turn sessions | MTP k=3 | 2.97 | 0.66 | [club-3090 #1746](https://github.com/noonghunna/club-3090/issues/1746) [M] |
  | same | same | DFlash2, 5 drafts | 3.59 | 0.52 | same |
  | same | same | DFlash2, 7 drafts | 3.85 | 0.41 | same |

  - ThinkingCap card: acceptance is highest on GSM8K and lowest on LiveCodeBench.
  - club-3090 #1746: DFlash2 (7) gave 136.5 tok/s long decode vs 101.9 for MTP k=3, and acceptance was "about the same across weight precisions". The thread has no base run.
  - The kaushikvira and Pi repos ship the same `z-lab/Qwen3.8-27B-DFlash2` "unchanged". None retrained a drafter ([kaushikvira](https://huggingface.co/kaushikvira/Qwen3.8-27B-thinkingcap-nvfp4full-dflash2-NInfer-v3), [Pi](https://huggingface.co/bytkim/Qwen3.8-27B-pi)).
- **Base reference acceptance:**
  - Vendor, base Qwen3.8-27B, block 8, T 1.0, xhigh [C] ([z-lab card](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)): GSM8K 5.46, MATH-500 5.28, HumanEval 4.39, MBPP 4.79, MT-Bench 4.10. MTP on the same tasks: 5.02 / 4.72 / 3.91 / 3.99 / 3.74.
  - Our E11-05 on L4 (base INT4, production Poetiq, long ARC traces) [M]: DFlash2 AL 3.8–4.5, falling from 4.43 to 3.77 across context, median 4.55 ([local research note](D:/ARC%20AGI%202/research_notes/Qwen%2027B%20দ্রুততম%20সার্ভিং%20সেটআপ/community_evidence.md)).
- **General guidance on drafter/target mismatch:**
  - A drafter's acceptance drops when the target is fine-tuned away from what the drafter was trained on; a generic DFlash checkpoint may underperform on a different domain or reasoning mode ([Kaitchup DFlash training](https://kaitchup.substack.com/p/train-and-run-dflash-speculative), [Baseten](https://www.baseten.co/inference-engineering/book/05-techniques/speculative-decoding/); search summaries).
  - [Osprey, arXiv 2609.09338](https://arxiv.org/pdf/2609.09338) discusses the sensitivity of the DFlash target interface (preprint).
  - I found no study measuring DFlash or EAGLE acceptance on a Qwen3.8 target before and after fine-tuning.
- **Existing project guard** (memory [sglang-int4-dflash2-plan](C:/Users/hp/.claude/projects/d--ARC-AGI-2/memory/sglang-int4-dflash2-plan.md)): `--speculative-draft-model-quantization unquant` is mandatory. Without it, AL collapses to about 1.0 (SGLang #39087).

### Inferences
- **[I] Swift 1.5 INT4 is the closest drop-in** for our SGLang 0.5.21 + compressed-tensors stack: same scheme, file layout and size as RedHat INT4. ThinkingCap would need a third-party quant (wasifb AutoRound or halt95 compressed-tensors) or our own LLM-Compressor run.
- **[I] DFlash2 acceptance on the tunes looks modestly lower** than on base (ThinkingCap 3.1–4.2 vs base 4.1–5.5 on vendor-style short tasks). The setups differ, so this is not proven.
  - A 10–20 % AL drop would offset part of the token saving.
  - Swift 1.5 changed `embed_tokens` and `lm_head`, which DFlash-style drafters use from the target. That is a plausible mismatch channel.
  - The net coverage gain must be measured as **tasks finished per hour on L4**, not tokens per task.
- **[I] An unchanged MTP head on a changed trunk** still "works" (the cards and the ThinkingCap measurement say so), but MTP AL on ThinkingCap (2.5–3.3) is below the base vendor MTP figures (3.7–5.0). The setups differ.

### Gaps
- No same-condition AL comparison (base vs tune) with DFlash2 or MTP.
- No measurement on L4, at TP4, or across 60k thinking traces.
- ThinkingCap's MTP and trunk weights could not be byte-checked (gated).
- The quant quality of the third-party ThinkingCap INT4 builds is unknown.

## 5. Licensing vs the ARC Prize 2026 open-source requirement

### Takeaway
- **Neither licence is OSI-approved,** and both restrict use by organization size or revenue:
  - Swift Open License v1.0: free commercial use only below US$1M revenue.
  - ThinkingCap-Qwen3.8: PolyForm Small Business 1.0.0 (under 100 workers and under US$1M revenue) plus a BottleCap personal-use grant; gated.
- **Topic 750575 quotes the competition rule:** "open-source model and weights as defined by the Open Source AI Definition". OSAID requires parameters "under OSI-approved terms" and use "for any purpose without having to ask for permission", so **both appear incompatible with prize eligibility**.
- No host answer exists yet (746307 and 750575, as of the 2026-10-22 dump).
- Base Qwen3.8-27B and Qwen Pi are Apache-2.0.

### Cited Findings
- **Swift Open License v1.0** [M, text read] ([LICENSE](https://huggingface.co/ukisai/Swift-Qwen3.8-27b/blob/main/LICENSE)):
  - Grant: perpetual, worldwide, royalty-free copyright and patent licence to reproduce, modify, sublicense and distribute.
  - **Sec. 5:** commercial use "is not licensed" for entities with gross revenue of one million US dollars or more (including affiliates); those need a "Swift Enterprise License". Qualified non-profits are exempt for non-commercial or research use.
  - Derivatives are allowed. No competition or field-of-use clause. Automatic termination on non-compliance.
  - It does not claim to be OSI-approved.
  - HF API: `license: other`, `license_name: swift-open-license-1.0`, not gated [M].
  - It does not appear in the SPDX list (only "CryptoSwift" matched) [M] ([SPDX license-list-data](https://raw.githubusercontent.com/spdx/license-list-data/main/json/licenses.json)).
- **ThinkingCap-Qwen3.8-27B:**
  - "PolyForm Small Business 1.0.0 + BottleCap personal-use grant"; "Commercial license: contact BottleCap AI" ([card](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B)).
  - HF API: `gated: auto`, `license_name: polyform-small-business-1.0.0` [M].
  - PolyForm SB permits company use only if the company "has fewer than 100 workers" and had revenue "below $1,000,000 (2019 USD)", adjusted by CPI-U ([SPDX text](https://spdx.org/licenses/PolyForm-Small-Business-1.0.0.html)).
  - SPDX list data: `PolyForm-Small-Business-1.0.0 isOsiApproved: false` [M] ([SPDX JSON](https://raw.githubusercontent.com/spdx/license-list-data/main/json/licenses.json)).
  - By contrast, `ThinkingCap-Qwen3.6-27B` is `license: apache-2.0` on HF (also gated: auto) [M]. That is the version Darshan linked in 746149, and Swift 1.0's "transfer component" derives from it.
  - note.com flags that the 3.8 licence differs from other Apache-2.0 Qwen3.8 tunes ([note.com](https://note.com/zephel01/n/nd65298dbf391?hl=en)).
- **Base and other:** `Qwen/Qwen3.8-27B` is Apache-2.0 (OSI-approved per SPDX) [M]. `bytkim/Qwen3.8-27B-pi` is Apache-2.0 [M].
- **Competition rules:**
  - The Kaggle rules page did not render for fetch (JavaScript), so Rule 1.a could not be read directly. Its wording comes from the question in 750575: "Competition-Specific Rule 1.a (open-source model and weights as defined by the Open Source AI Definition)" ([local dump](D:/ARC%20AGI%202/resources/Analysis%20and%20Journals/Kaggle%20discussions%20(2026-10-22)/topic_750575.md)).
  - 746307 asks whether non-OSI open-weight licences (the Qwen Community License for Flash-Next) are eligible, "or must every model component carry an OSI-approved licence". Its only comment is empty, so there is no host answer ([local dump](D:/ARC%20AGI%202/resources/Analysis%20and%20Journals/Kaggle%20discussions%20(2026-10-22)/topic_746307.md)).
- **OSAID 1.0** ([opensource.org](https://opensource.org/ai/open-source-ai-definition)): "Use the system for any purpose and without having to ask for permission"; "Parameters shall be made available under OSI-approved terms".
- **arcprize.org competition page** ([2026](https://arcprize.org/competitions/2026/)):
  - "All leading participants are expected to open source their solutions to be eligible for a prize."
  - Own code must be under "a permissive public domain license" (CC0 or MIT-0).
  - Third-party code needs at least an open licence permitting public sharing, such as Apache-2.0 or GPLv3.
  - The page is silent on pre-trained model licences ([ARC-AGI-2 track](https://arcprize.org/competitions/2026/arc-agi-2)).
- **Related precedent:**
  - CPMP (NVARC): "You should always read the license of every model you use." Teams were told they acted against the rules when they fine-tuned his non-tunable model without permission ([topic 735058 local dump](D:/ARC%20AGI%202/resources/Analysis%20and%20Journals/Kaggle%20discussions%20(2026-10-22)/topic_735058.md)).
  - Topic 685621: externally trained pre-trained models are allowed ("yes you can", CPMP) ([local dump](D:/ARC%20AGI%202/resources/Analysis%20and%20Journals/Kaggle%20discussions%20(2026-10-22)/topic_685621.md)).
  - Topic 748741 (synthetic data from Claude or Codex) has no ruling; CPMP: "I cannot provide a legal advice" ([local dump](D:/ARC%20AGI%202/resources/Analysis%20and%20Journals/Kaggle%20discussions%20(2026-10-22)/topic_748741.md)).

### Inferences
- **[I] Prize eligibility.** Under Rule 1.a as quoted in 750575, a revenue-threshold licence (Swift) or a small-business licence (PolyForm SB) fails the OSAID "any purpose" freedom and the "OSI-approved terms" test. A solution built on either would likely not be prize-eligible.
  - Using one for **leaderboard-only** submissions is a separate question. The licences themselves permit personal, research and evaluation use, and Kaggle's 735058 precedent shows the hosts read model licences.
  - This is not legal advice; the user or the hosts must decide.
- **[I] The Apache-2.0 path,** if the user wants token efficiency with clean licensing, is:
  - (a) the base model plus a training-free intervention (the 2606.00206-style overthinking-marker logit penalty);
  - (b) Qwen Pi (an agent-harness-specific tune with unproven ARC fit);
  - (c) our own length-aware fine-tune of the Apache base, released openly.

### Gaps
- No host ruling on non-OSI model licences (746307 and 750575 are open).
- The Kaggle rules page could not be fetched, so Rule 1.a was not read verbatim from the source.
- Whether receiving prize money counts as "commercial use" under Swift Sec. 5 or PolyForm SB is untested legally.

## 6. Kaggle usage of these variants (ARC-AGI-2 / ARC-AGI-3)

### Takeaway
- No Kaggle team has reported results with Swift or ThinkingCap on ARC-AGI-2 or ARC-AGI-3.
- The only mention is Darshan Makwana's idea post (746149). He did not run them and said he is not seriously competing.
- Swift was already flagged internally as a candidate (E11-04 review, E14 gaps note) but never tested.

### Cited Findings
- **Darshan Makwana (746149, 2026-10-06)** ([local dump](D:/ARC%20AGI%202/resources/Analysis%20and%20Journals/Kaggle%20discussions%20(2026-10-22)/topic_746149.md)):
  - On Qwen3.8 27B Q4/Q2: "they spend a lot of tokens on overthinking and generating long CoT traces". He proposed fine-tuning "to penalize overthinking tokens similar to what swift qwen and even bottle caps has done … reduced thinking tokens by 50% with marginal drop".
  - His links: Swift-Qwen3.8-27b, **ThinkingCap-Qwen3.6-27B** (not 3.8), and arXiv 2606.00206.
  - He later wrote: "I was just researching for fun and am not considering seriously participating".
- **The 2026-10-22 ARC-AGI-2 discussion re-read** (35 threads) lists no Swift or ThinkingCap result among the Qwen3.8 high scorers. Those high scorers (mikelou1 69.72, NguyenThanhNhan 65.97, and others) do not disclose such variants ([00_SUMMARY](D:/ARC%20AGI%202/resources/Analysis%20and%20Journals/Kaggle%20discussions%20(2026-10-22)/00_SUMMARY.md)).
  - A grep of all dumps for "swift" or "thinkingcap" hits only 746149 [M].
- **Web search:** I found no Kaggle ARC notebook using these models ([search result set](https://hackernoon.com/swift-qwen38-27b-cuts-reasoning-tokens-without-sacrificing-much-accuracy)).
- **Internal prior mentions** (none tested):
  - [E11-04 rev2b self-audit](D:/ARC%20AGI%202/experiments/Experiment%2011%20-%20SGLang%200.5.21%20+%20RedHat%20INT4%20+%20DFlash2%20(planned)/Experiment%20Review/E11-04%20rev2b%20self-audit%20+%20goal%20check%20(2026-10-18).md): Swift as a "drop-in that could double coverage if ARC accuracy holds".
  - [E14 gaps note](D:/ARC%20AGI%202/experiments/Experiment%2014%20-%20Optimized%20Engine%20on%204xL4%20(planned)/Notes/05%20Claude%20-%20gaps%20found%20after%20full%20re-read%20(2026-10-23).md), item 7.

### Inferences
- **[I] There is no field evidence either way on ARC.** If the user chooses to test, the minimal evidence-producing step is a paired EVAL run against the current INT4 base. It would use the same harness, the same 60k floor and the same DFlash2 drafter. The candidate is `ukisai/Swift-1.5-Qwen3.8-27b-INT4`, the drop-in format.
- **[I] Metrics for that run:**
  - correct per task;
  - tokens per request (mean, median, cap-hit rate);
  - EOS-without-`</think>` rate (the W3H failure mode);
  - DFlash2 AL per 10k window;
  - tasks finished per GPU-hour.
- **[I] The licence question (§5)** decides whether a positive result is usable for a prize-eligible submission or only for the leaderboard.

### Gaps
- No ARC-AGI-3 usage found either.
- Closed-source top teams' model choices are unknown.
