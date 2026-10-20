# Deep research brief: what is behind the recent score jumps on the ARC Prize 2026 (ARC-AGI-2) Kaggle leaderboard?

You are a research agent with web search. Run as many searches as you need (aim for 150+ distinct queries across the
sources below), follow links, and read primary sources. I need evidence, not speculation. Today is 2026-10-04.

## Background

* Competition: Kaggle "ARC Prize 2026 - ARC-AGI-2" (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2),
  code competition, offline, 12 h runtime limit, hardware **4x NVIDIA L4 (96 GB total)**, no internet. Deadline
  2026-11-02. Public LB is scored on the semi-private ARC-AGI-2 set (120 tasks). The 85 % grand-prize threshold is on
  the private set.
* Public-notebook ceiling: the best public notebooks score ~33.89 (NVARC-style: Qwen3-4B-class model + test-time
  fine-tuning, the 2025 winning recipe and its forks, e.g. "ARC AGI2 LB33.89 Minimal Perfpatch").
* Our team sits at 32.64 with an NVARC-derived pipeline. Our own Qwen3.8-27B-FP8 REPL program-synthesis agent scores
  only 1.67 % on the 120 public evaluation tasks.
* A Kaggle discussion (topic 742796, "LLM-agent is all you need.", 2026-09-23) says reasoning-LLM agents are the way
  forward; one participant (Gladstone025) reports a Qwen3.8 agent submission at **13.75 %** LB; another (Russell Kirk)
  claims 30-50 % locally (unverified).

## The anomaly to explain (from a daily leaderboard-history monitor, https://arc3.huikang.dev/leaderboard/?comp=arc2)

Single-submission jumps far above the public ~34 ceiling, mostly by teams that are new or have very few submissions:

| team | current | submissions | first entry | jump(s) | runtime of best sub |
|---|---|---|---|---|---|
| Tufa Labs | 83.06 (#1) | 42 | 2026-08-22 | 0 -> 32.50 (08-24); 48.61 -> 49.86 (09-14); -> 58.33 (09-15); -> 71.11 (09-16); -> 73.47 (09-18); -> 83.06 (09-22) | ~723-733 min |
| Yi-Chia Chen | 55.28 (#4) | 14 | 2026-04-16 | 40.69 -> 48.89 (09-09); -> 54.72 (09-14) | ~630 min |
| _hans | 46.39 (#5) | 8 | 2026-08-26 | 30.14 -> **46.39** (10-02) | 694 min |
| ALbert RUehlman | 45.97 (#6) | 4 | 2026-09-04 | 28.06 -> **45.97** (10-03) | 689 min |
| S J Moudry | 45.83 (#7) | 148 | 2026-05-19 | 28.89 -> **45.83** (10-01) | 475 min |
| okiek | 37.36 (#9) | 11 | 2026-09-23 | 30.14 -> 37.36 (09-28) | 715 min |
| Anil Thomas | 36.81 (#11) | 15 | 2026-09-15 | 31.11 -> 35.69 (09-28) | ~530-840 min |
| rabbithole | 79.72 (#2) | 105 | 2026-05-31 | 50.42 -> 70.42 (08-23) | ~730 min |
| nvbanana | 76.67 (#3) | 133 | 2026-05-31 | steady 65 -> 76 | ~460-640 min |

Three different teams jumped from ~28-30 to ~46 (+16 to +18) in one submission each, on 2026-10-01, 10-02 and 10-03.
That looks like one shared trigger released around 2026-09-28 .. 10-01. Our working hypothesis (unverified): NVARC-style
TTT (~30) **plus** a reasoning-LLM agent (Qwen3.8-class, ~14 % alone) in one 12 h notebook, union of answers ≈ 40-46.

## Questions to answer

1. What did these teams (and team members) actually do? Find any public trace: writeups, GitHub repos, papers, blog
   posts, X/Twitter, LinkedIn, Discord, Kaggle profiles and their other notebooks/datasets/models, Hugging Face
   accounts. Team names may differ from personal handles; check Kaggle profile pages.
2. Was something released around 2026-09-25 .. 2026-10-03 that would let several teams gain +16 points at once? E.g.
   a new open-weight model or checkpoint (especially Qwen3.8 / Qwen3.x reasoning, DeepSeek, GLM, gpt-oss, Kimi,
   Mistral, Gemma), a fine-tuned ARC model on Hugging Face, a dataset of ARC programs/solutions/reasoning traces, a
   vLLM/SGLang feature (speculative decoding, FP8/NVFP4 kernels for L4), a public agent harness, a paper with code.
3. What does an LLM-agent ARC-AGI-2 solver that fits **4x L4, 12 h, offline, ~120-240 tasks** look like in practice?
   Model and size, quantization, reasoning effort / thinking budget per task, number of attempts per task, tool use
   (Python REPL, program verification on train pairs), how it is combined with TTT/NVARC, time split.
4. What exactly is Tufa Labs' approach (0 -> 83 in a month)? Any public statement, talk, paper, repo or hiring post.
5. Known techniques reported to work on ARC-AGI-2 in 2025-2026 (ARC Prize 2025 papers/writeups, ARC-AGI-2 technical
   report, arcprize.org blog, Kaggle 2025 top solutions: NVARC, ARChitects, MindsAI, CompressARC, TRM/HRM, Poetiq,
   Jeremy Berman's evolutionary program synthesis, SOAR, "product of experts", etc.), and which of them can run within
   the 4x L4 / 12 h limit.

## Where to search (all of these)

* Kaggle: competition discussion forum (all topics since 2026-08-15, including comments), every public notebook of the
  competition sorted by score and by date, the profiles/notebooks/datasets/models of the teams above, the
  ARC-AGI-3 competition forum (same people often post there), Kaggle Models.
* Hugging Face: models and datasets with "arc", "arc-agi", "arc-agi-2", "arc2", "nvarc", "barc", "re-arc", "qwen3.8",
  sorted by recently updated; HF forum and paper pages.
* GitHub: repos, issues and discussions mentioning arc-agi-2, arcprize, nvarc, "arc prize 2026"; recently created or
  updated repos; arcprize org repos.
* arXiv / OpenReview / Papers with Code / alphaXiv: ARC-AGI-2 papers from 2025-2026.
* arcprize.org (blog, leaderboard, technical reports), the ARC Prize Discord (public mirrors, summaries), Reddit
  (r/MachineLearning, r/LocalLLaMA, r/singularity), X/Twitter, LessWrong, Hacker News, YouTube talks, Medium/Substack,
  Chinese/Japanese/Korean/Vietnamese sources (Zhihu, Qiita, Velog, etc.: several top teams are from these communities).

## Rules for the answer

* Every claim needs a link and a date. Mark each item **verified** (primary source), **reported** (secondary source)
  or **inference** (your reasoning). Never present an inference as fact.
* Prefer primary sources (the team's own words, code, model cards) over commentary.
* If you find nothing for a question, say so explicitly and list what you searched.
* Do not stop at the first plausible story; actively look for evidence against the hypothesis above.

## Output format

1. **Top findings** (max 10 bullets), each with verified/reported/inference and links.
2. **Timeline 2026-08-15 .. 2026-10-04** of relevant releases/posts/events, with links.
3. **Per-team dossier** for each team in the table: what is known, sources, confidence.
4. **Concrete recipe(s)** that plausibly explain a +16 jump on 4x L4 in 12 h: model, quantization, inference engine,
   token budget, attempts, verification, ensembling with NVARC — with sources.
5. **Actionable recommendations** for a team at 32.64 (NVARC pipeline) with a Qwen3.8-27B-FP8 agent at 1.67 %, about
   4 weeks and ~15 4xL4-hours per week per account left. Ranked by expected gain / cost.
6. **Search log**: the queries you ran (grouped by source) and notable dead ends.
