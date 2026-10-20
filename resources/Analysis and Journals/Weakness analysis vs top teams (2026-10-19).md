# Where we are weak against the top teams: research + our own data (2026-10-19)

**Sources:**
- Four research agents: about 300 web searches/fetches, plus all 103 ARC-AGI-2 and 240 ARC-AGI-3 Kaggle threads pulled via the JSON API into `_scratch/webresearch/`.
- Our runs: R5, D66, L66, E11-05 EVAL30.
- The key forum quotes were re-checked in the raw dumps (marked **V**). **H** = hypothesis.

## 0. The baseline that frames everything

- **The public LB uses the same tasks as ARC Prize's semi-private set.** V (CPMP/nvbanana, thread 743880: "It is the semi private data used for the verified leaderboard").
- **Qwen3.8-27B xhigh by itself scores 42.4 % there.** V (arcprize.org). It used about 2 attempts × ~85k tokens per task (forum estimate from $0.4469 per task, thread 745787).
- **We are at 34.58, below the bare model.** The top teams are +20 to +45 above it with the same model, so the gap is the harness, not the model.
- **Someone did it with Qwen3.8 alone.** mikelou1 (rank 13) went 25 → 36 → 40 → **66 LB, about 60 local CV**, with no NVARC ensemble. The 40 → 66 step came from fixing a "tiny kernel issue". V (thread 748082). Whether it was 27B or Flash-Next was asked but not answered.

## 1. Our weaknesses, ranked

### W1. One long blind trace per attempt; the model cannot run code while thinking (biggest)

**Our data:**
- Forced closes waste most of the compute:
  - R5: natural closes → train pass **75 %**; forced closes → **21 %**. D66: 59 % vs 11 %.
  - About 74 % of R5's tokens went to forced iterations.
  - EVAL30 at 49k: 94 % forced.
- About 45–50 % of the thinking is hand-checking grids that code execution would do exactly:
  - 4 % test-grid simulation, 6–7 % dry runs of the code, 33–39 % checking train grids (sim_tokens.py).
- 57–73 % of all tokens went to tasks that never got a hit.

**Outside evidence:**
- The Oct 2–5 jump wave (_hans, ALbert, Majkel, Dawid) came right after the ARC-AGI-3 Milestone-2 solutions were open-sourced on Sep 30. All three winners are Tufa's Duck REPL harness. V (timing).
- Tufa's "The Duck": Qwen 27B with a live Python REPL, image + ASCII grid, and context eviction. V (arcprize blog).
- amn (rank 8, 70.28; LB snapshot `_scratch/webresearch/lb.json`, 2026-10-19). V. Their comment (2026-10-18, thread "Qwen3.8-27B / Flash-Next: ARC-AGI-2 results within Kaggle's compute limits?") says only: "FN with an acceleration (eg. quant?) **might** reach a better score according to my experiments, by letting it work as an agent". It does **not** say the 70.28 came from Flash-Next; the method behind 70.28 is unknown. (Corrected 2026-10-19: an earlier version of this line claimed "70.28 = Flash-Next agent".)
- Symbolica REPL agent: GPT-5.2 53.3 → 70.3. Dartmouth study: reflective orchestrator +9.75 pp, Qwen3-235B +6. V.

**Our history:**
- Q27-AGENT EVAL12 scored 0/22, because of my implementation bugs, not the method:
  - 59/82 submits had empty code;
  - vLLM HTTP 500s;
  - thinking capped at 6k per turn.
- The PIV REPL track got LB 1.67, a bad design: the tool was only allowed after `</think>`, and turns were small.
- We abandoned the agent route. **That was our biggest strategic mistake.**

**Caveat:** the public Qwen3.8 agent notebooks score only 15–27 (khoinguyennguyen, madarshbb). A REPL alone is not enough; implementation quality decides.

### W2. Too little compute per task (throughput)

- **Need vs have:** about 1,000 tok/s is needed to give 240 tasks the ARC-Prize budget (2 × 85k) in 12 h. We have 330 (SGLang INT4 + DFlash2, measured; DFlash acceptance only falls 4.43 → 3.77 over context, so no collapse). Others on 4×L4 report about 350 for FP8 (thread 745787). V
- **The 11-of-12 SGLang bug** (fix built in rev 3c) costs about 8 %.
- **Flash-Next (MoE, 6B active):**
  - About 4× fewer bytes per token, which suits the L4's low bandwidth. H
  - The ARC-AGI-3 winners run it at 946–1,159 tok/s, but on an RTX PRO 6000, not L4. V
  - On 4×L4 the W4A16 weights (~70 GB) make KV room tight. H
  - **The licence (Qwen Community License) may block prize eligibility.** The hosts have not ruled (thread 745079). V
- **A REPL agent lowers the per-attempt cost** (stop at the first all-train-pass, no hand simulation), so W1 also attacks W2.

### W3. No diversity: 1 expert, a weak attempt_2

- D66 28 + L66 34 → union 50 of 103. That fits independent attempts (1−(1−p)²).
- Our attempt_2 adds about +1.
- Dartmouth: 1 → 5 explorers took pass@2 from 45.5 to 57.5; turning refinement off cost −0.75 pp. Trelis and beetree: refinement does not beat more samples. V
- Li & Ellis: induction 38 + transduction 43 → ensemble 56.75. V
- Two independent branches per task give attempt_1 and attempt_2. The khoinguyennguyen agent does this.

### W4. Selection and overfit

- Of our train-verified programs, 35–47 % are wrong on test (EVAL30: 7/15).
- Imbue: fitness = 90 % correctness + 7 % "transfer" (an LLM judges whether the code fits the test inputs) + 3 % simplicity. Hard-coded programs are 5–20 % of candidates (Trelis). V
- Agreement across independent branches is the cheapest selector.

### W5. Process

- We validated the engine for days with many audits, while the method gap had no active track (self-audit 2026-10-18).
- 1 submission per day, about a 33 h queue, and rosters lock on **2026-10-26**. V

### Not weaknesses (checked)

| Item | Why it is not a weakness |
|---|---|
| Sampling (T 1.0 / 0.95 / 20) | Qwen's official setting. Presence penalty 0 is the card default; khoinguyennguyen uses 1.5 without proof. |
| Early-stop of hopeless traces | Does not work in the literature either (within-problem AUROC about 0.5). |
| FP8 KV | Already on. |
| Many shorter traces instead of one 60k (low effort) | Worse on ARC-AGI-2 (low 22.8 vs xhigh 42.4). |
| TTT / fine-tuning NVARC | Tapped out (±2–4 noise). |

### Open / unverified

- **Post-training:** LoRA SFT on traces.
  - Arcnights (68.06): "SFT on LLM works … cheap with LoRA". ARC Raiders: QLoRA on 27B "works well". NVIDIA's Nemotron-SFT-ARC-AGI-v1 is agentic SFT data. V quotes; size of the gain unknown.
  - Needs training compute outside the 12 h run.
- **adam99 claims hidden tasks are sized like training,** which contradicts the user's fact "hidden ≈ eval". Not verified, kept as the user's fact.

## 2. What to do (big jumps, bundled; user decides)

1. **REPL tool agent on our engine, built on battle-tested open code** (da-fr/arc-agi-3-solution and the Tufa Duck harness), not on our own failed agent code. It includes:
   - xhigh with `preserve_thinking`;
   - a Python tool that runs `transform` on all train pairs and returns diffs;
   - stop at the first all-train-pass;
   - 2 independent branches per task (attempt_1 and attempt_2);
   - context trimming with hysteresis;
   - priority scheduling.
   
   **Expected:** the largest gain (+10 to +30, H). It attacks W1, W2 and W3 together.
2. **Throughput:** rev 3c (11 → 12); then decide on Flash-Next. Speed vs the licence risk is the user's call; we have ARC-AGI-3 Flash-Next experience.
3. **Selection:** agreement across branches, a transfer/simplicity check, and a direct-grid answer as the fallback attempt_2.
4. **Keep the 60k floor per thinking turn** (user rule). In the agent it means: the first turn may think ≥ 60k, and the tool turns continue in the same context.

## 3. Addendum: the Kaggle agent (4th report; raw data in `_scratch/session_1c6db7aa/disc_1019`, `kpull`, `lb_1019`)

- **No public notebook is at or above 40** (best 33.89, NVARC style). V (Kaggle API, best public score per notebook).
- **The public Qwen3.8 notebooks are all weak:**

  | Notebook | What it is | Score |
  |---|---|---|
  | madarshbb agent | python + submit tools | 15.69 |
  | madarshbb direct b98k c16 | 2 samples, 98k thinking | 21.53, and 13 after a small change |
  | HyperQwen | INT4 + MTP-3, concurrency 4 | 10.42 |
  | russcore | run_python tool | 10.14 |
  | khoinguyennguyen agent | author's best 30.42 | unscored |

  V. The agent idea alone does not reach 40; it needs engineering on top.
- **NguyenThanhNhan:** now 41.25. Earlier pure reasoning, FP8 + MTP, 65k xhigh, forced commit at 49k: 23.33. He says a simple harness covers many more tasks. V
- **Run-to-run variance** of identical notebooks: 3–4 points. V
- **LB growth:** 16 teams at ≥ 40 on Oct 11 → 50 at ≥ 40, 28 at ≥ 50 and 17 at ≥ 60 on Oct 18 (3,032 teams). V (API snapshot)
- **Rosters, as reported by the agent from the API (NOT re-verified by me):**
  - Tufa Labs includes Jeroen Cottaar (the Duck author).
  - ARC Raiders includes keithtyser (Duck Qwen3.8 serving).
  - Yi-Chia Chen has custom DFlash drafters and on-policy-distilled models.
  - Domdolus Tolus patched library code for speed.
  - The "secret sauce works for ARC 2 and 3" quote was not found verbatim in the dumps.
- **Throughput (H for L4):** the HyperQwen author finds speculation helps only below about 8 concurrent requests. For us, DFlash2 at 12 still beat no-spec (404 vs 289 tok/s, E11-03), so this does not transfer directly.
- **Serving gotcha:** FlashInfer's FP8 attention needs SM90; on L4, use Triton for FP8 KV. V (forum)
