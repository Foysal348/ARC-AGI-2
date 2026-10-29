# Public evidence of ≥600 tok/s aggregate decode for 27B–32B models on 4×L4 or similar Ada multi-GPU setups, and what ARC Prize 2026 competitors say about speed, concurrency and coverage (state as of 2026-10-29)

**Scope and method.**
- **Builds on, does not repeat:** `research_notes/Qwen 27B দ্রুততম সার্ভিং সেটআপ/community_evidence.md` (state 2026-10-23). Items from that file are repeated only when needed for the throughput catalogue, and are marked "(prior note)".
- **Kaggle discussions:** 23 topics were fetched read-only on 2026-10-29 through the kagglesdk discussion API. Raw text is in `D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/disc/topic_<id>.txt`. The script is `fetch_list.py` in the same scratchpad.
  - Topic 750546 ("Qwen3.8 or something else?") was read in full, 38 comments, as of 2026-10-29 11:08 (`_scratch/kaggle_disc/topic_750546.json`).
- **Leaderboard:** the full ARC-AGI-2 leaderboard history (3,493 teams) was downloaded from Tong Hui Kang's monitor endpoint, `https://tonghuikang--arc3-leaderboard-monitor-get-history.modal.run?comp=arc2`. It backs the page [arc3.huikang.dev/leaderboard/?comp=arc2](https://arc3.huikang.dev/leaderboard/?comp=arc2) and is saved at `.../scratchpad/hk/lb2.json`.
- **Public notebooks:** the list of public ARC-AGI-2 notebooks was pulled by API on 2026-10-29. Four new notebooks were pulled read-only and never executed (`.../scratchpad/kpull/`).
- **Web:** about 20 searches and fetches covered Reddit mirrors, GitHub, AWS and other blogs, and FriendliAI model pages.
- **Labels:**
  - **[MEASURED]** = the author says they measured it.
  - **[CLAIM]** = self-report with no method or detail.
  - **[ESTIMATE]** = arithmetic or a guess.
- **Known coverage limits:**
  - The Kaggle search API returned only 70 ARC Prize 2026 topics. The forum cannot be listed directly (403).
  - Probing topic ids to find topics created after 2026-10-26 hit HTTP 429 rate limits and was abandoned.
  - So any ARC-AGI-2 topics posted 10-26 to 10-29, other than 752709, may be missing. See Gaps.

---

## 1. What ARC Prize 2026 competitors have said publicly about tok/s, concurrency, the 27B vs Flash-Next choice, engines and harness design

### Takeaway
- **The only concrete ≥600 tok/s claim on Kaggle 4×L4 comes from Darren Amadeus Martin (2026-10-27/28)** [CLAIM, with a stated definition]:
  - "top speed of 900 tok/s";
  - "980 tok/s on 1 minute harness work which includes tool call and prefill";
  - pure decode at harness concurrency "like 1100 tok/s";
  - **"average decode speed is 710 tok/s"** over a 6-hour, 2-pass public-eval run, which "could drop to 250 tok/s" under heavy prefill.
  - He credits "lots and lots of optimizations by claude code". He does not disclose the method, the model or the engine.
- **Other competitors report 250–500 tok/s:**
  - TheDivine Aurora: "250-400tps", earlier "300-500 tps";
  - Adarsh is "struggling to even get half of" 900;
  - our own measured 249–431 (prior note).
- **On 10-27 Anil Thomas wrote that "many of the leading teams are using Qwen3.8-Flash-Next alongwith the Duck harness".**
  - His arithmetic puts full single-pass coverage at a 64k mean length at ~365 tok/s, and a second pass on 20% of tasks at ~440 tok/s.
  - Adarsh counters that on L4×4, 27B and Flash-Next are "almost breakeven in speed and accuracy".

### Cited Findings

**Topic 750546, "Qwen3.8 or something else?" (opened by Anil Thomas on 2026-10-18; 38 comments as of 10-29)** — [Kaggle 750546](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750546)
- **Anil Thomas, 2026-10-18 (opening post):**
  - "the sudden flood of 55+ scores makes me think that there is an easy recipe out there. I was able to go from 35 to 45 by ensembling my NVARC-based solution with a Qwen3.8-27B run";
  - "a harness that includes a python REPL can help Qwen3.8-27B quite a bit. but squeezing it into a 12 hour window on 4x L4 is hard."
- **Anil Thomas, 2026-10-23:**
  - "I'm almost convinced that the recipe involves distilling ... a student that can fit into a single L4 probably makes the most sense. that way, the slow link between the GPUs doesn't hurt you."
  - mikelou1 replied the same day: "I'm no where near the top but I don't use distillation."
- **Anil Thomas, 2026-10-27** [CLAIM + ESTIMATE]:
  - "the cat is finally out of the bag! it is clear that many of the leading teams are using Qwen3.8-Flash-Next alongwith the Duck harness ... the beginning of the surge coincides with the publication of Daniel Franzen's milestone 2 solution";
  - "some rough modeling indicates that Flash-Next can get to a score around 75% with one pass and 80% with two passes";
  - "assuming a mean length of 64k per task, 100% coverage for a single pass requires a throughput of 240*64k/(12*60*60) ~ 365 tok/s";
  - "if you want to do a second pass for the unsolved tasks (can be detected fairly accurately by looking at demo success): (240+240*0.2)*64k/(12*60*60) ~ 440 tok/s";
  - "that looks like a significant engineering challenge on 4xL4. how are people doing this?! aiming for shorter generations may be easier".
- **Adarsh, 2026-10-27:** "my qwen 3.8 flash-next experiments are showing 27b and fn are almost breakeven in speed and accuracy on L4x4. On a singular GPU though, yes flash-next is faster. My experiments are based on franzen's duck harness setup".
- **TheDivine Aurora, 2026-10-28, replying to Adarsh:**
  - "your local evals are faulty. qwen FN has a lot more solve % on the public eval dataset than 27b. burned weeks on 27b";
  - "you can use openrouter and check. i tried 16 task dataset ... and 4-5 attempts per task. it solved all of them ☠️quantization hurts quality 🥀".
- **Darren Amadeus Martin, 2026-10-27 21:17** [CLAIM]: "With lots and lots of optimizations by claude code, our team manage to get a top speed of 900 tok/s. Your hunch is true that it's mostly engineering".
- **Darren, 2026-10-27 21:31** [CLAIM]: "our team managed to get a top speed of 980 tok/s on 1 minute harness work which includes tool call and prefill work as well. In our experience, 1 single stream on L4 is very slow, so we can only have high speed when we increase the concurrency. Now the secret sauce is how to have high concurrency when 1 problem itself can reason for more than 100k tokens."
- **Darren's definition of tok/s** (answering cm391, who asked whether prefill is counted, 2026-10-28 11:56):
  - "I think it's closer to the former [total generated output tokens / wall-clock harness time]. For example if in 1 minute the model on harness generated 60000 and prefill of 5000 token, then the 'decode speed under harness' is 1000 tok/s.";
  - "on pure decode only with same concurrency as the harness has like 1100 tok/s".
- **Darren, 2026-10-28 12:51** [CLAIM]: "The peak is 980 tok/s with my definition of decode speed mentioned before. When there's much prefill, it could drop to 250 tok/s. But on 6 hours and 2 pass of the public evaluation, tge average decode speed is 710 tok/s (due to the tail being starved of concurrency)".
- **Darren, 2026-10-28 06:29** (ARC-AGI-3): "it did not transfer sadly. Our team also decided to join arc agi 3, tried the same optimization and submitted once only to get a score of 4 lol".
- **Darren, 2026-10-28 13:21** (asked whether the optimizations work on a PRO 6000): "Actually it works there too, the speed itself is much faster than franzen's setup but it doesn't translate to the leaderboard which is fairly disappointing. However I can't tell you more than this as I would spill my team's method".
- **TheDivine Aurora, 2026-10-28** [MEASURED, self-report]: "im just curious to know floor and ceil of throughput in your harness. iam at 250-400tps range at the moment".
- **Adarsh, 2026-10-27:** "900 is so insane 😮 struggling to even get half of that".
- **Anil Thomas, 2026-10-28:** "900 tok/s — this is the way".
- **mikelou1:**
  - 2026-10-20: "With time constraints at around 4xL4 speed I got Qwen to 87%";
  - 2026-10-19: "I currently get around 79% ± 5% in CV runs";
  - "The CV run used LB-ish environment".
- **Darren on CV vs LB, 2026-10-19:**
  - "our cv also got scores easily above 75% but it doesn't translate that well to LB though";
  - "my team uses the public CV 120 problems and time limit of 6 hours but it seems that the correlation didn't stand up well".
- **TheDivine Aurora, 2026-10-20** (prior note, re-confirmed): "at around 300-500 tps i had ~19-20 % score. only attempted half the dataset. so practically 40 % is possible with 600-800 tps, 1 attempt per task ... it just takes 128k - 256k context window".
- **Kris Dreemurr, 2026-10-18** (prior note, re-confirmed): "median 70k tokens for pass@2 (so 35k per slot) gets %45 solves or about 52~ out of 120 ... The issue is I cant fit it in the box"; the NVARC solver runs "on all 240 tasks in about 4 hours".

**Topic 745787, "Qwen 3.8 token generation speed" (Darren, 2026-10-04)** — [Kaggle 745787](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/745787)
- **New comments since 2026-10-21:** none.
- **Darren's opening post:** "4 L4 seems to be very slow ... due to lower bandwidth and slow gpu communication. Has anybody got like an aggregate above 500 tok/s ?"
- **Darren on NguyenThanhNhan's setup:** "With 65k tokens/task (assuming you only did 1 pass), 240 task and 12 hours, you'd need roughly 360 tok/s on aggregate. Not only that, you'd also need to fit the kv cache."
- **Darren's SGLang flags:** he posted an SGLang server config in this thread; the excerpt begins with `--kv-cache-dtype fp8_e4m3`. The rest of the config is in the prior note's Table A.

**Topic 746149 (2026-10-06)** — [Kaggle 746149](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/746149)
- **Darren** [ESTIMATE]: "the official ARC website states that it uses $0.08/task at $0.28/million output so roughly 250k tokens per task. Assuming it's 2 passes, 1 pass costs roughly 125k tokens. to solve roughly 240 pass at 12 hours, we'd need roughly 700 tok/s"; "Qwen 3.8 flash next might have a better shot at 4 L4 since we can offload its embedding layer into ram". This was about dots3-note.

**Topic 752709, "Are solutions using Qwen3.8-Flash-Next eligible for prizes?" (Darren, 2026-10-26)** — [Kaggle 752709](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/752709)
- **Darren:** "Many teams in both competitions have spent a lot of time building pipelines around this model".
- **Mark Barney, 2026-10-27:** "the ARC Prize team confirmed on Discord that Qwen3.8-Flash-Next is allowed for competition use. Permaximum also said he had raised it with Greg, who confirmed it too" ([digest](https://arc.markbarney.net/news/2026-10-26-flash-next-cleared-for-arc-prize)).
- **Status:** a Discord confirmation relayed second-hand, not a Kaggle host post.

**Topic 748082 (Flash-Next / 27B results within limits)** — [Kaggle 748082](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/748082)
- **New comments after 10-21:** none.
- **The Silverhead Engineer:** "I can get qwen 3.8 to 25 too ... im milking the life out of the inference speed".

**Infrastructure statements bearing on coverage (topic 751344, 10-21 to 10-26)** — [Kaggle 751344](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/751344)
- **Dustin (Kaggle staff), 2026-10-23:**
  - "we were experiencing instability with the dataset cache the past few days in some regions";
  - "I strongly suggest putting Models in Kaggle Models instead of datasets ... so that we can place them on the right type of cache";
  - "Fuse isn't necessarily a ton slower than NFS, there's a ton of variables".
- **NguyenThanhNhan, 2026-10-23:** "I have re-uploaded my models in Kaggle Models. The save run showed that Model was mounted over NFS."
  - Earlier in the thread: "57422625 (65.97) was scored about 17.5 hours after submission";
  - "I need to read ~180 GB from /kaggle/input"; his startup stages include "kvcache allocation, mtp draft setup, cuda graphs for decode captured".
- **xz, 2026-10-22:** "5-10% lower score could be a CPU throttling issue. I found the throughput to be affected significantly by CPU despite my efforts to overlap CPU work with GPU."
- **shiiin9, 2026-10-23:** the same notebook scored 65.56 on 10-20, then 59.31 and 58.61 during the slowdown; "Resubmitting an identical notebook earlier moved our score by only ±0.4."
- **Trian, 2026-10-25:** a "Kaggle Error" "after ~18h".
- **Harp404, 2026-10-26:** "waiting for 17hrs for result".

**ARC-AGI-3 evidence on concurrency vs KV** (topic 750885, Francise Liang, 2026-10-20) [MEASURED] — [Kaggle 750885](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/750885)
- **Setup:** Franzen's M2 notebook on an RTX Pro 6000, with "ARC3_MAX_ACTIVE_STREAMS, MAXREQ and CUDAGRAPH_MAXBS from 10 to 14".
- **Score:** mean 29.90 vs 47.30.
- **Throughput:**
  - "The baseline held about 600-620 generated tok/s for the whole run";
  - "With 14 streams, cumulative throughput peaked around 470 tok/s and fell to about 344, so late in the run it was near 280 tok/s";
  - "Total generated tokens: 5.07M vs 8.74M".
- **Their reading:** "with the 136k context and FP8 KV cache, 10 streams is already close to what the KV cache can hold".
- **xz's reply:** "Your lanes are constantly getting preempted because there is not enough memory reserved for kv cache. This is very bad for throughput."

**ARC-AGI-3 notes on agents and tuning** (topic 744792) — [Kaggle 744792](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792)
- **Scott Le Grand:** "how useless the coding agents are for understanding vllm and sglang ... I just keep telling the agents to 'try everything you can' to optimize TPS".
- **NNbody:** "every time I get a TPS improvement, some errors show up to remind me there's no free lunch".
- **rellik13:** "the kv pool sat somewhere between 91-97% full and requests queued for cache space, so the memory was the actual limit".

**ARC-AGI-3 Duck run settings** (dustyhorizon, topic 751598, 2026-10-25) — [Kaggle 751598](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/751598)
- **Loading:** "model weights had loaded normally (185.40 GB, 754.5 s)".
- **Run parameters:** "standard 1-pass / concurrency 120 / 31920s".
- **Why it matters:** an ARC-AGI-3 Flash-Next Duck configuration, showing what the harness concurrency means as distinct from engine concurrency.

**Who said nothing new:**
- No new posts by Kris Dreemurr, nvbanana members (CPMP, Ivan Sorokin) or Tufa Labs (Jeroen Cottaar) about tok/s were found in the fetched topics.
- CPMP and Jeroen posted only about the storage incidents in 751344.
- **Tufa Labs on ARC-AGI-3 (2026-09-23):** "We still plan to release our final solution when the competition ends" ([Kaggle 742801](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/742801)).
- **CPMP (nvbanana), 2026-09-24:** "We will open source our code at the end of competition if we end up in gold".

### Inferences
- **Darren's 710 tok/s average implies a large token budget** (my arithmetic). Over the full 12 h minus ~30 min startup, 710 tok/s averages ~29 M generated tokens, about 122k output tokens per task across 240 tasks. That is enough for two ~60k passes per task, which matches his "2 pass" description.
  - Our measured 250–430 tok/s gives about 10–18 M tokens, or 43–75k per task.
- **Darren's model is undisclosed.** His 10-26 eligibility post says "many teams ... building pipelines around this model" (Flash-Next), and Anil says leading teams use Flash-Next + Duck. Together these suggest, but do not prove, that his 900–980 tok/s was on Flash-Next.
  - That is consistent with "1 single stream on L4 is very slow" and the need for high concurrency.
  - It is also consistent with Adarsh's finding that 27B and FN are near breakeven *in his setup*.
- **"Coverage" is limited by concurrency and the tail, not only peak decode.** Darren's own drop from 980 peak to 710 average is attributed to "the tail being starved of concurrency". The ARC-AGI-3 results show that more streams than the KV pool holds cut throughput sharply (600–620 → 280 tok/s).

### Gaps
- **Darren's team** is not named in the posts. The brief calls him "3rd place". The 10-29 leaderboard 3rd place is "Domdolus Tolus" (83.47), but nothing links the two. Unverified.
- **Not disclosed by Darren:** model, engine, quantization, spec-decode method, number of streams, context lengths.
- **No tok/s published by any team in the top 20** of the leaderboard (Tufa Labs, Yi-Chia Chen, nvbanana, rabbithole, mikelou1, etc.).
- **Possibly missing topics:** any ARC-AGI-2 topics created 2026-10-26 to 10-29, other than 752709, may have been missed (search API coverage plus 429 rate limits).

---

## 2. How leading teams achieve coverage of 240 tasks in 12 h: the leaderboard record and what top teams have said

### Takeaway
- **Scores have moved far above the Qwen-only region we sit in.** As of the 2026-10-29 snapshot of the leaderboard monitor (3,493 teams):
  - top 5: Tufa Labs **88.06**, Yi-Chia Chen @ Fields Model 84.72, Domdolus Tolus 83.47, nvbanana 81.39, rabbithole 80.83;
  - the 20th team is at 69.44, the 40th (Daniel Franzen) at 57.64;
  - Hack2Publish is at 34.58 (rank 123).
- **Methods:** no top team has published its method or tok/s. The only method statements are Anil's "Flash-Next + Duck" inference and Darren's "mostly engineering".

### Cited Findings
- **Top 40 of the ARC-AGI-2 public leaderboard** (rank, team, public score, submission count), from the monitor endpoint downloaded 2026-10-29 ([arc3.huikang.dev](https://arc3.huikang.dev/leaderboard/?comp=arc2)):
  - 1 Tufa Labs 88.06 (61)
  - 2 Yi-Chia Chen @ Fields Model 84.72 (35)
  - 3 Domdolus Tolus 83.47 (47)
  - 4 nvbanana 81.39 (153)
  - 5 rabbithole 80.83 (125)
  - 6 Ravi Teja Chilakala 76.94 (15)
  - 7 Majkel1337 76.81
  - 8 ALbert RUehlman 75.56
  - 9 Dawid Kopiczko 75.14
  - 10 guoqin gu 73.89
  - 11 ｶﾎｳｼｭｳｾｲ 73.47
  - 12 _hans 71.81
  - 13 ảc 71.53
  - 14 Youssef Nader 70.56
  - 15 artificialagencylab.com 70.56
  - 16 matu997 70.14
  - 17 mikelou1 69.72
  - 18 この遥かなる旅路に祝福を 69.72
  - 19 Sakayori Lab 69.58
  - 20 Lashen Learning 69.44
  - 21 mtg 69.31
  - 22 Top 8 Runner 69.31
  - 23 Nguyen 68.89
  - 24 kibuna 67.64
  - 25 S & M & N & S & Y 67.08
  - 26 masspeaks 66.67
  - 27 ibyyue 66.25
  - 28 復活の混テキスト 66.11
  - 29 Flying Pandas 65.97
  - 30 okiek 63.47
  - 31 Yanming (Owen) Dong 61.53
  - 32 Aneesh & Yash 61.39
  - 33 Laureano Arcanio 61.25
  - 34 KhaoticDeotterministicSethesis 60.69
  - 35 Lee Minseung 59.58
  - 36 Rick 59.44
  - 37 rxe 59.17
  - 38 Kajetan Schweighofer 58.19
  - 39 megsub 57.92
  - 40 Daniel Franzen 57.64
- **Others named in the brief:**
  - Hiểu Vy 53.33 (rank 48);
  - Anil Thomas 46.39 (59);
  - Hack2Publish 34.58 (123);
  - Kris Dreemurr 34.31 (126);
  - "ADARSH REDDY B" 33.19 (164; assumed to be Adarsh, unverified);
  - TheDivine Aurora 28.47 (1977).
- **Late climbs in the monitor history** (score per day, 10-22 → 10-27/28):
  - Domdolus Tolus 73.47 → 83.47;
  - Ravi Teja Chilakala 55.97 → 76.94;
  - Nguyen 50.14 → 68.89;
  - Rick 43.75 → 59.44.
  - Many teams jumped between 10-22 and 10-27, after Franzen's M2 publication and Anil's 10-27 post.
  - Source: same endpoint.
- **What the monitor's minutes field contains:** a "minutes_taken" value per submission.
  - Recent values for top teams are 1,074–1,525 min (Tufa Labs 10-25: 1,303; 10-27: 1,497).
  - These exceed the 12-h (720-min) run limit, and the page's own cutoff is 750 min. So the field evidently includes queue time; it is not a pure run time.
  - My reading of the field, from the page code (`formatMinutes`, `runtimeCutoffMinutes: 750`).
- **ToolNavs (2026-10-10), secondary report** [CLAIM]: "Tufa Labs reached 88.06%, taking first place"; the $150,000 bonus pool is "split among all teams scoring over 85%" ([ToolNavs](https://toolnavs.com/article/2447-arc-agi-2-high-score-refresh-tufa-labs-tops-the-board-at-8806-past-the-85-bonus)). The 88.06 matches the monitor data.
- **NVARC lineage:** CPMP says "all public notebooks being tiny variations on our last year solution" ([Kaggle 742801](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/742801)). nvbanana's July jump was "Different weights" per Van-Phuc Huynh, unconfirmed ([Kaggle 729509](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/729509)).
- **Coverage arithmetic by competitors, all [ESTIMATE]:**
  - Anil: 365 tok/s for one pass at a 64k mean; 440 tok/s with a 20% second pass (750546, 10-27).
  - Darren: 360 tok/s for 65k × 240 in 12 h (745787); 700 tok/s for a 125k pass (746149).
  - TheDivine Aurora: "600-800 tps" for ~40% with 1 attempt per task (750546).
- **cosmorin, "Why #1 Is #1 — ARC-AGI-2 Gap Analysis" (2026-10-25):**
  - "Time is a hidden score axis ... per-task processing times on a 120-task dev run ranged from ~525 s to ~1,234 s ... naive ordering starves the tail. A predicted-time-ordered queue (our predictor: R² 0.564, Spearman ρ 0.777) is worth real points";
  - this is about an NVARC-type pipeline ([notebook](https://www.kaggle.com/code/cosmorin/why-1-is-1-arc-agi-2-gap-analysis), pulled).

### Inferences
- **The top-5 scores (80–88) sit above even the most optimistic Qwen3.8-27B-alone projections in the threads.**
  - Anil puts Flash-Next at "around 75% with one pass and 80% with two passes".
  - mikelou1's CV is 79% ± 5%.
  - So the top teams likely combine a fast Flash-Next (or 27B) harness with something else, or have a much stronger recipe.
  - No public evidence identifies what that is.
- **The coverage gap between us and the 60–70 band is roughly 2× in tokens per task.**
  - The arithmetic above shows the 60–70 band is reachable with about 360–440 tok/s *sustained* if mean tokens per task stay near 64k.
  - Our measured 249–431 tok/s is peak-ish, not sustained.

### Gaps
- **No team in the top 40 has published a solution, notebook or tok/s for ARC-AGI-2.** Tufa Labs and nvbanana say they will release at the end of the competition.
- **The monitor gives no pure run time,** so how much of the 12 h each top team uses cannot be determined.

---

## 3. Public Kaggle notebooks serving Qwen3.8 on 4×L4 with vLLM/SGLang: flags and reported speeds

### Takeaway
- **No public notebook newer than 2026-10-14 serves Qwen3.8 (27B or Flash-Next) on 4×L4 for ARC-AGI-2.** This is from the API list of competition notebooks, sorted by run date, on 2026-10-29.
- **None of the public Qwen3.8 notebooks reports a measured aggregate tok/s.** The six known ones (Adarsh ×2, russcore, khoinguyennguyen, delongyang, yjsh2003) are documented in the prior note's Table A.
- **The single new relevant notebook** is delongyang's Qwen3.6-35B-A3B on 4×L4 (vLLM 0.19, TP4, `--max-num-seqs` 16), also with no tok/s.

### Cited Findings
- **Notebook list on 2026-10-29** (`kernels_list(competition='arc-prize-2026-arc-agi-2', sort_by='dateRun')`, queries "flash next", "qwen3.8", "duck", "27b", "vllm", "sglang"):
  - most recent Qwen3.8 notebook: `madarshbb/arc-agent-sub-q38-direct-b98k-c16-792ef9`, last run 2026-10-14;
  - then `khoinguyennguyen/arc-agi-2-agentic-multi-turn-reasoning` (10-10), `madarshbb/arc-qwen-3-8-27b-agent` (10-09) and `delongyang/arc-agi-2-inference-qwen3-8-27b-fp8-within-4-hours` (10-06).
  - Newer notebooks, e.g. `evgendvorkin/arc-2-2026-4-4` (10-28, 68 votes, best LB "32.64%"), are NVARC Qwen3-4B variants.
  - Notebook URLs: [madarshbb](https://www.kaggle.com/code/madarshbb/arc-agent-sub-q38-direct-b98k-c16-792ef9), [evgendvorkin](https://www.kaggle.com/code/evgendvorkin/arc-2-2026-4-4).
- **delongyang, "arc-agi-2 inference qwen3.6-35b-a3b within 3 hours"** (run 2026-10-03, pulled):
  - Stack: `mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3`; weights `giovannimuniz/qwen3-6-35b-a3b`.
  - Serving: "OpenAI-compatible server with TP=4 and a 131K context window"; `N_PARALLEL` 16, "Must be ≤ `--max-num-seqs`".
  - Env and flags: `VLLM_USE_FLASHINFER_SAMPLER=0`; `VLLM_ATTENTION_BACKEND=TRITON_ATTN`; `--tool-call-parser qwen3_coder`, `--reasoning-parser qwen3`.
  - Notes: "A KV-cache dtype override and a speculative-decoding (MTP) config are both plausible" (not enabled); "vLLM 0.19 wants cu128".
  - No throughput is reported.
  - Source: [notebook](https://www.kaggle.com/code/delongyang/arc-agi-2-inference-qwen3-6-35b-a3b-within-3-hours).
- **Prior-note notebooks:** for exact flags (MTP ladders, fp8 KV + `TRITON_ATTN`, `max-num-seqs` 16–24, `max_model_len` 65k–262k), see `Qwen 27B দ্রুততম সার্ভিং সেটআপ/community_evidence.md` §1 Table A.
  - The only rate anywhere in them is Adarsh's planning default `"tok_per_s": 140`, not a measurement.
- **ARC-AGI-3 public notebooks (RTX PRO 6000) with measured speeds:** see the prior note §2.
  - Franzen M2: P90 aggregate decode 836.6 tok/s, harness-side 609.0 tok/s; peak 946.
  - lordhansolo: 1,135 tok/s peak. sirikilohit: 1,159 tok/s peak.
  - Francise Liang, re-running Franzen's notebook: "600-620 generated tok/s for the whole run" ([750885](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/750885)).

### Inferences
- **Every ≥600 tok/s Qwen3.8 figure on Kaggle hardware is either ARC-AGI-3 (one RTX PRO 6000, ~1.8 TB/s, TP1) or Darren's undisclosed 4×L4 claim.** The public 4×L4 ARC-AGI-2 notebooks give no baseline above our own measurements.

### Gaps
- **No saved notebook output (vLLM "Avg generation throughput" log lines) was retrievable** for any public 4×L4 Qwen3.8 notebook.
- **No public Flash-Next-on-4×L4 ARC-AGI-2 notebook was found.**

---

## 4. Catalogue of public throughput measurements for 27B–35B models on 4×L4, Ada multi-GPU and comparable PCIe setups

### Takeaway
- **No independent public measurement of ≥600 tok/s for a 27B–32B dense model on 4×L4 exists.**
- **The ≥600 tok/s measurements that do exist for this model class all use GPUs with roughly 3× the L4's bandwidth:**
  - 4× L40S, 686 tok/s at C32 in BF16;
  - 4× RTX 3090 with SGLang DFlash2, 927.5 at c16;
  - 1× RTX 3090 with HyperQwen, ~1,035 at c64;
  - 1× RTX 4090 D with INT4 and no speculation, 649 at bs20.
- **Bandwidth-similar analogs are much lower:**
  - 4× RTX PRO 2000 Blackwell (about L4 bandwidth per card, my background figure): 386.9 tok/s at c8 with NVFP4 + MTP3;
  - our own 4×L4 measurements: 249–431.
- **Two patterns repeat across sources:**
  - replicas beat TP over PCIe;
  - speculation stops paying around 4–8 concurrent requests on the GDN hybrid unless the state pool is sized for it.

### Cited Findings

**Table B — throughput catalogue.** All entries are [MEASURED] by their authors unless stated otherwise. tok/s is aggregate output decode unless stated otherwise.

| # | Source (date) | Model / quant | Engine | GPUs, link, TP | Concurrency, lengths | Throughput |
|---|---|---|---|---|---|---|
| 1 | AWS ML blog, "Accelerate ... G7e" (2026-04-20) | Qwen3-32B BF16 (dense) | native vLLM container, prefix caching | ml.g6e.12xlarge = **4× L40S**, TP4 | ~1,000 in / ~560 out | C1 37 agg (37.1/req); C8 242; C16 416; **C32 686** (21.5/req). 1× RTX PRO 6000 TP1: C32 592 |
| 2 | dgallitelli blog, Qwen3.6-27B on SageMaker (2026-04-27) | Qwen3.6-27B FP16 / FP8 | vLLM 0.19.0 + MTP | g6e.24xlarge **4× L40S**, TP4 | 11 in / 50 out; saturates at conc. 256 (text says ~128) | **19.82 req/s (FP16), 20.19 req/s (FP8)**; ~1K input: 1.61 / 1.73 req/s; g6.48xlarge (8× L4, TP8, FP8): **CUDA OOM**; "42% failure rate at concurrency 128" on g6e |
| 3 | club-3090 #1288 (prior note) | Qwen3.8-27B W8A8 INT8 + DFlash2 (8) | SGLang v0.5.19 | **4× RTX 3090**, 220 W, NVLink pairs, TP4 | max_running 16 | c4 487.7 / 893.9; c8 730.6 / 1,418.3; **c16 927.5 (narrative) / 1,811.0 (code)** |
| 4 | syv-ai/HyperQwen README (latest dated item 2026-10-18) | Qwen3.8-27B, requantized int8/Marlin mix | vLLM 0.31.0 + patches | **1× RTX 3090** at 250 W | 64 concurrent, 128 in / 512 out | **~1,035 tok/s decode, 948 e2e; ~1,222 with every layer int8**; single stream 127 (MTP); "Speculation wins below ~8 concurrent users, plain batching above"; "Multi-GPU works at TP=2 and TP=4", no TP numbers; issue #135: with four 3090s "two of them already give _less_ aggregate throughput than one" |
| 5 | vLLM #55533 (prior note) | RedHatAI Qwen3.8-27B-INT4 / GPTQ Marlin | vLLM | **1× RTX 4090 D (sm89)**; reproduced on RTX 6000 Ada | bs1 / bs8 / bs20 | no-spec 56.4 / 367.1 / **649.0**; MTP k=1 72.0 / 197.0 / 210.9 (scheduler ran "exactly 3 of the 8 sequences") |
| 6 | FriendliAI page for sakamakismile/Qwen3.8-27B-MTP-NVFP4 (undated, ©2026) | Qwen3.8-27B NVFP4 | vLLM v0.22.0, `qwen3_5_mtp` n=3, fp8 KV, 128k ctx | **4× RTX PRO 2000 Blackwell, TP4** | c1 / c2 / c4 / c8 | no-MTP 49.0 / 93.2 / 180.9 / 318.3; **MTP3 72.6 / 110.9 / 232.8 / 386.9**; KV 1,025,977 tokens; "boards without P2P need `NCCL_P2P_DISABLE=1` and `--disable-custom-all-reduce`" |
| 7 | alexander-ollman, "From 25 to 283 tok/s" (benchmarks 2026-04-30) | Qwen3.6-27B Lorbus AutoRound INT4; Qwen3.6-35B-A3B AWQ | vLLM 0.19.2 nightly | **2× RTX 3090 PCIe, no NVLink** | 1024 in / 1024 out | 27B **2 replicas (TP1) + nginx**: C1 81.8, C2 154.0, **C4 225.4**, C8 167.0; 27B TP2 ~55–68 single; 35B-A3B TP2+EP: C4 **282.7**, C8 282.2; MTP n=5 single-stream 100.3 |
| 8 | FriendliAI page avyukth/Qwen3.8-27B-AWQ-INT4 | Qwen3.8-27B AWQ INT4 | vLLM (version not stated), max_model_len 8192 | **1× RTX 3090** | 256 out, `ignore_eos` | c1 41.1; c4 136.4; **c8 221.5**; c16 216.9; c32* 211.8 ("saturates at concurrency 8") |
| 9 | r/LocalLLaMA "Finding the 4x 3090 Sweet Spot", u/anitamaxwynnn69 (snapshot 2026-05-23) | Qwen3.6-27B FP16 | vLLM v0.20.2, TP4 | **4× RTX 3090, PCIe Gen3 x16/x8/x8/x4** | not stated | output **29 t/s** at 250–390 W, 27 at 220 W; commenter: "the tests measured a PCIe 3.0 x4 all-reduce bottleneck" |
| 10 | NVIDIA forum DGX Spark TP/DP study (prior note) | Qwen3.8-27B | vLLM | 4× DGX Spark | C8 | TP2 + DP2 36.0 tok/s per request vs TP4 27.0; "two TP=2 replicas are the more effective configuration" |
| 11 | club-3090 #1092 / #1746 (prior note) | Qwen3.8-27B W4A8 AutoRound; Swift-1.5 | vLLM 0.27.1 / 0.31.0, TP2 | 2× 3090 (PCIe / NVLink) | C1–C32 | MTP4 vs off: C32 241 vs 308; DFlash2 OOM at C8 on 24 GB; speculation "stops paying at 4+ clients" |
| 12 | tonyd2wild/Qwen38-Flash-Next-4x3090 (prior note) | Flash-Next W4A16 experts + FP8 n-gram on NVMe | vLLM nightly, TP4 + EP, MTP3 | 4× RTX 3090 | 6 agents | **252.6 aggregate** (317 sustained / 570 peak) |
| 13 | Kaggle 732723, Dominic (prior note) | Nex-N2-mini (Qwen3.5-MoE 35B-A3B), fp8 per-tensor | vLLM 0.26 | **4× L4** | not stated | 438 / 452 tok/s |
| 14 | Our team (prior note) | Qwen3.8-27B FP8 + MTP; RedHat INT4 + DFlash2 | vLLM 0.19; SGLang 0.5.21 | **4× L4, TP4** | 16 running; 6–12 running | 249–353.5; 321–431 |
| 15 | Darren Amadeus Martin, Kaggle 750546 (2026-10-27/28) [CLAIM] | not disclosed | not disclosed | **4× L4** | "high concurrency" | peak 900–980 (harness output / wall clock), pure decode ~1,100, 6-h 2-pass average **710**, low 250 under prefill |

- **Sources for Table B:**
  - Row 1: [AWS blog](https://aws.amazon.com/blogs/machine-learning/accelerate-generative-ai-inference-on-amazon-sagemaker-ai-with-g7e-instances/)
  - Row 2: [dgallitelli](https://dgallitelli.github.io/blog/posts/qwen36-sagemaker-benchmark/)
  - Row 3: [club-3090 #1288](https://github.com/noonghunna/club-3090/discussions/1288)
  - Row 4: [HyperQwen](https://github.com/syv-ai/HyperQwen)
  - Row 5: [vllm#55533](https://github.com/vllm-project/vllm/issues/55533)
  - Row 6: [FriendliAI sakamakismile](https://friendli.ai/models/sakamakismile/Qwen3.8-27B-MTP-NVFP4)
  - Row 7: [alexander-ollman](https://alexander-ollman.github.io/qwen3.6-on-rtx3090/qwen3.6-on-rtx3090.html)
  - Row 8: [FriendliAI avyukth](https://friendli.ai/models/avyukth/Qwen3.8-27B-AWQ-INT4)
  - Row 9: [Reddit snapshot](https://reddit.sentinel-team.org/posts/1te9o18/snapshots/2026-05-23T00%3A36%3A34.219378Z)
  - Row 10: [NVIDIA forum](https://forums.developer.nvidia.com/t/comprehensive-qwen3-8-27b-study-on-dgx-sparks-quantization-speculative-decoding-and-tp-dp-scaling/381102)
  - Row 11: [club-3090 #1092](https://github.com/noonghunna/club-3090/discussions/1092), [#1746](https://github.com/noonghunna/club-3090/issues/1746)
  - Row 12: [tonyd2wild](https://github.com/tonyd2wild/Qwen38-Flash-Next-4x3090)
  - Row 13: [Kaggle 732723](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/732723)
  - Row 14: local experiment reports cited in the prior note
  - Row 15: [Kaggle 750546](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750546)
- **Caveats on individual rows:**
  - **Row 2:** the req/s are with very short outputs that often stopped early at EOS. The author's AIPerf run on 1× RTX PRO 6000 gives only 214 output tok/s (50 out, c32) and 521 tok/s (2,048 out, c8). The author says the two methods "differ by about 10x". So the 4× L40S req/s cannot be converted to decode tok/s.
  - **Row 6:** the page does not give input or output lengths for the decode benchmark.
  - **Row 8:** the c32 row "appears later on the page, near a `W4A16_ASYM` label", so it may belong to a different variant (per the fetch summary).
- **Other reports, single-stream only** (not aggregate, listed so they are not mistaken for batch numbers):
  - **RTX 6000 Pro Blackwell, vLLM 0.24.0, Qwen3.6-27B MTP2, `--max-num-seqs 2`:** tg32 BF16 ~59–68, FP8 ~97–103, NVFP4 ~158–171 tok/s (r/LocalLLaMA u/live4evrr, mirrored 2026-07-05). Source: [prismix](https://prismix.dev/news/d238dab26aa7).
  - **4× RTX 5090 (Chinese write-up):** "约 625 tokens/秒/卡" and "Avg gen throughput: 2000 tok/s". This is a search-result snippet only; I did not fetch the page. Source: [gitcode.csdn](https://gitcode.csdn.net/6a0ceb7b10ee7a33f273cb87.html).
- **Ada-specific kernel note:** "Block-FP8 has beautifully tuned kernels for Hopper and Blackwell datacentre cards, and none for Ada (SM89) ... quietly loses you around 40% of your decode"; AWQ INT4 runs on Marlin "well tuned for Ada". This is a search snippet, not fetched. Source: [houtini 48 GB 4090 tuning](https://houtini.com/articles/vllm-tuning-48gb-rtx-4090/).
  - Consistent with the prior note's HF #16 and vLLM #59232 findings.

### Inferences
- **Bandwidth scaling, row 1 (4× L40S, BF16 32B):** an L40S has ~864 GB/s vs the L4's ~300 GB/s (background figures from the NVIDIA datasheets, not fetched; ratio ~2.9). Row 1's 686 tok/s at C32 would correspond to roughly 240 tok/s on 4× L4 for a BF16 dense 32B at the same lengths.
  - Weight bytes per step dominate at C32 with short contexts.
  - INT4 weights (~4× fewer bytes than BF16) would raise the weight-bound ceiling roughly 3–4×. Long ARC contexts (60–120k) add KV reads that this benchmark does not have.
  - So rows 1, 3 and 4 are consistent with 600–1,000 tok/s on 4×L4 *only* with ~4-bit weights, high concurrency (≥16), and efficient speculation or very low overheads. This is my arithmetic, untested.
- **Row 6 (RTX PRO 2000 Blackwell) is the closest bandwidth analog to 4×L4 that I found.**
  - Its per-card bandwidth is about 288 GB/s (background figure, not fetched; unverified), and it has 16 GB per card.
  - It reaches 386.9 tok/s at c8 with NVFP4 + MTP3 and is still rising (318 → 387 from c4→c8 without/with MTP). It was not measured at c16+.
  - This is in line with our 321–431 at 6–12 streams.
  - If the curve keeps scaling, c16–c24 could reach ~600+. That is an extrapolation, not a measurement, and NVFP4 is Blackwell-only.
- **Two patterns appear independently in rows 4, 5, 7, 10 and 11, on PCIe consumer cards:**
  - replicas with TP1/TP2 beat TP4;
  - speculative decoding loses to plain batching above ~4–8 concurrent requests on Qwen3.x hybrid models, unless the recurrent-state pool is enlarged.
  - On 4×L4, a 27B INT4 (~19.5 GB) cannot run TP1 per card (24 GB), but 2× TP2 replicas fit. Untested on L4.
- **Darren's ~1,100 tok/s pure decode on 4×L4 is plausible only for a model with fewer active bytes per step than a 27B INT4 under TP4.** This is my inference from the bandwidth bound in the prior note (~800–950 tok/s at 16 sequences for the 27B before overheads), or it requires heavy speculative gains at high concurrency. Flash-Next (6B active) or a strongly speculated 27B are both consistent; nothing public settles which.

### Gaps
- **No independent measurement of any Qwen3.x 27B–35B model on 4× L4 above 452 tok/s** (Dominic's 35B-A3B) was found on GitHub, Reddit mirrors, blogs or HF.
- **No measurements found on these setups:** 4× RTX 4090 TP4 with Qwen3.8-27B at c16–c32; L40S or RTX 6000 Ada with INT4 at high concurrency; L20.
- **r/LocalLLaMA itself is not indexed.** Reddit was reachable only through the `reddit.sentinel-team.org` snapshot mirror, which captures a subset of comments.
- **The houtini and CSDN 4×5090 figures** come from search snippets only and were not fetched.
- **Unverified hardware figures:** the per-card bandwidth figures for L40S, L4 and RTX PRO 2000 Blackwell are my background knowledge, not fetched.
