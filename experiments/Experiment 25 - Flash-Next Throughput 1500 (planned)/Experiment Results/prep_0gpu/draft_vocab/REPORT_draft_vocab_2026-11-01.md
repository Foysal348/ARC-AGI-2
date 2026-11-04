I built and checked all five draft-vocabulary tables, but the report file was not written. The tool blocks subagents from writing report .md files, so `REPORT_draft_vocab_2026-11-01.md` does not exist yet. The full report text is below; please save it to that file.

All 5 tables load with Merlin's own `load_draft_vocab_ids(path, 248320, tp_size=2)` (also at tp 1 and 4, and with `VLLM_DRAFT_VOCAB_SHA256` set). Held-out coverage is 99.36–99.91 %. The modeled acceptance-length (AL) loss on real Duck is −0.74 % at 16k down to −0.11 % at 64k. Two full runs produced byte-identical files.

**Tables recommended for the GPU arms:**
- **32,768**, the main arm: sha `a164d1a71d27dab600c014673c34accfdeb8695141a63be8aa4bfb4026695248`.
- **65,536**, the safe arm. The plan's "65k + int8 head" arm uses this same file: sha `a915fa6d5ad711817976de5c988a3950f43e7b2fbc058b04325e75c8501cf022`.
- **16,384**, optional aggressive arm: sha `81ed1d6e7f072dbd6dc763de9cff420e628476a53c41c77e73b4fcd71de9e63c`. It saves only 0.5 ms more than 32k, but its modeled AL loss is about 2× larger, so it tests whether head bytes really are the cost.

---

# Exp 25 item 1-1 (0 GPU): ARC-derived reduced draft vocabulary for the Qwen4Exp MTP drafter (2026-11-01)

CPU only. No GPU, no Kaggle, no git. Response text was never printed or viewed: every step ran through scripts that print only counts, ids and hashes.

## 1. Result in one table (held-out, 2-fold task split, mean of both folds; chosen fill = `gen_plus_e25`)

| N | held-out coverage | modeled real-Duck AL (2.8145 base) | worst-case AL bound | head MiB / PP1 rank (TP2) | 3 draft-head reads (scaled from 7.5 ms) | saving vs BF16 full head | saving vs I8 full head |
|---|---|---|---|---|---|---|---|
| 16,384 | 99.356 % | 2.7938 (−0.74 %) | 2.7766 (−1.35 %) | 40.0 | 0.49 ms | 7.01 ms | 4.20 ms |
| 24,576 | 99.538 % | 2.7996 (−0.53 %) | 2.7873 (−0.97 %) | 60.0 | 0.74 ms | 6.76 ms | 3.95 ms |
| 32,768 | 99.642 % | 2.8029 (−0.41 %) | 2.7933 (−0.75 %) | 80.0 | 0.99 ms | 6.51 ms | 3.71 ms |
| 49,152 | 99.781 % | 2.8074 (−0.25 %) | 2.8014 (−0.46 %) | 120.0 | 1.48 ms | 6.02 ms | 3.21 ms |
| 65,536 | 99.906 % | 2.8114 (−0.11 %) | 2.8089 (−0.20 %) | 160.0 | 1.98 ms | 5.52 ms | 2.72 ms |

All five files load with Merlin's own `load_draft_vocab_ids(path, 248320, tp_size=2)`. They also load at tp 1 and tp 4, and with `VLLM_DRAFT_VOCAB_SHA256` set. A wrong sha is refused.

**Recommended GPU arms:**
- **32,768** (main arm);
- **65,536** (safe arm; the "65k + int8 head" arm also uses this file);
- **16,384** (optional aggressive arm).

## 2. Files

Folder: `D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Results/prep_0gpu/draft_vocab/`

| file | sha256 | torch fingerprint (`draft_vocab_fingerprint`) |
|---|---|---|
| draft_vocab_k100_arc_n16384.pt | `81ed1d6e7f072dbd6dc763de9cff420e628476a53c41c77e73b4fcd71de9e63c` | n=16384:sha256=2e3518ec465556faedb209d9ce8ef64be85aa11fd9899d404da6c84c0a66fb8b |
| draft_vocab_k100_arc_n24576.pt | `b0edd5aea005f9584af1920b469797b79840eae28e41e3bf48228282f049af16` | n=24576:sha256=c22559720ad043ece9af7971623c03b4cdda91063e1dbeb5ea1400fa45eedab8 |
| draft_vocab_k100_arc_n32768.pt | `a164d1a71d27dab600c014673c34accfdeb8695141a63be8aa4bfb4026695248` | n=32768:sha256=ac3c4b19dffc837e0129eeb769cceebb5336760dc0d525ceb9919c715ee89944 |
| draft_vocab_k100_arc_n49152.pt | `7b50acb0ad55a5e981783a7dc4de24d7470876ab252e16c432f25978da0d34c1` | n=49152:sha256=9378c8f90e1d4c18081fb12afb8b17ec95c38aefa80acdc01a23dd9fe06872ae |
| draft_vocab_k100_arc_n65536.pt | `a915fa6d5ad711817976de5c988a3950f43e7b2fbc058b04325e75c8501cf022` | n=65536:sha256=2258b96d9f272c4301ab98b22caf387b12e1f41630ba5dbd25893d0f25999a35 |
| build_draft_vocab.py | `097897ae833f198c487767a50292487f3221644dc747221f5e4d259c5fb12b3e` | re-runnable script: `python build_draft_vocab.py` (not `-I`: numpy is in the user site), about 90-150 s on CPU |
| manifest.json | `687f6d67f922ab9ed9b383e93a109d90d9e08a330ac5e495867e718a7f18d9f3` | every number in this report, plus per-fold detail |

- **Reproducibility:** two full runs produced byte-identical .pt files (same sha256).
- **Format** (what the loader requires): `torch.save` of a 1-D contiguous int64 CPU tensor, sorted, unique, with 0 < N < 248,320.
- **Alignment:** every N is divisible by 128 × TP: by 256 at TP2 and by 512 at TP4.
- **Id range:** min id 0, max id 248,076. No padding rows (ids ≥ 248,077) are included.

**GPU use:** set `VLLM_DRAFT_VOCAB_FILE=<file>` and `VLLM_DRAFT_VOCAB_SHA256=<sha256 above>`. The loader hashes and loads the same bytes. Pin the shipped files: re-saving can change the sha.

## 3. Data and task sets

| source | model | requests | tokens (re-tokenized) | server `completion` sum | tasks | use |
|---|---|---|---|---|---|---|
| E26 K100Q Duck (`_scratch/e26_gpu_out/.../phase_Q_K100_FP8_K3/duck/*/nb5_requests.jsonl`) | klee100 3bpw-MTP, FP8 KV, MTP K3 | 741 | 2,537,356 | 2,537,367 | 30 | **main**: build + 2-fold held-out |
| E25 NB5 BF16 Duck | Flash-Next W4A16 Merlin p1-3, BF16 KV | 278 | 599,859 | 599,876 | 15 (subset of the 30) | secondary build data (by fold) + held-out |
| E25 NB5 FP8 Duck | same, FP8 KV | 273 | 670,891 | 670,902 | 15 (same 15) | same |
| E25 NB6 submit smoke Duck | Flash-Next W4A16 Merlin | 39 | 23,885 | 23,892 | 3 (1818057f, 2ba387bc, 45a5af55) | same |
| E27 NBDF arm Q speed bench (`responses_nbdf_q.jsonl.gz`) | klee100 | 102 (1 duplicate dropped) | 220,263 saved ids | — | prompts = concatenated evaluation tasks | held-out only (synthetic bench) |
| E25 k3_v2 klee100 load + sweeps (`traces/requests.jsonl`) | klee100 | 85 | 188,052 saved ids | — | same kind of prompt | held-out only |

**Task sets: all ARC-AGI-2 evaluation tasks.**
- All 30 E26 tasks, the 15 E25 NB5 tasks and the 3 NB6 tasks are in `arc-agi_evaluation_challenges.json` (120 tasks).
- No training-task responses exist locally (no `arc-agi_training_challenges.json` and no runs on training tasks). So the plan's "train tasks only" could not be followed.
- A token-frequency table holds no solutions, grids or answers: it is only a set of token ids.
- The 30 tasks: 136b0064, 16b78196, 1818057f, 20270e3b, 221dfab4, 247ef758, 2b83f449, 2ba387bc, 2d0172a1, 35ab12c3, 38007db0, 3dc255db, 45a5af55, 58f5dbd5, 64efde09, 65b59efc, 7491f3cf, 7666fa5d, 800d221b, 898e7135, 8e5c0c38, 8f215267, a395ee82, b5ca7ac4, bf45cf4b, d59b0160, db0c5428, e8686506, edb79dae, f931b4a8.
- 2-fold split (deterministic sha256 order, seed `e25-1-1-draft-vocab`):
  - **Fold A:** 1818057f, 247ef758, 2b83f449, 2d0172a1, 58f5dbd5, 7491f3cf, 7666fa5d, 800d221b, 898e7135, 8e5c0c38, 8f215267, a395ee82, db0c5428, e8686506, edb79dae.
  - **Fold B:** 136b0064, 16b78196, 20270e3b, 221dfab4, 2ba387bc, 35ab12c3, 38007db0, 3dc255db, 45a5af55, 64efde09, 65b59efc, b5ca7ac4, bf45cf4b, d59b0160, f931b4a8.

The E26 records carry no token ids, only reasoning / content / tool_calls strings. The script rebuilds each request's generated stream in the chat template's assistant layout:
- `reasoning` + `\n</think>\n\n` + content;
- `<tool_call>\n<function=NAME>\n<parameter=K>\nVALUE\n</parameter>\n</function>\n</tool_call>`;
- `<|im_end|>`.

It then tokenizes the stream. The check against the server's own counts:
- 2,537,356 vs 2,537,367 tokens over all 741 requests;
- median per-request difference 0, p90 0, max 6.

## 4. Tokenizer check (06b95093 vs 0997f410)

- `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`: the klee100 tokenizer, 19.99 MB (`_scratch/e13_rev6/harness/fixtures/tok_int4/tokenizer.json`).
- `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`: the earlier Flash-Next / Merlin p1 tokenizer, 12.8 MB (`.../tok_fp8/tokenizer.json`).
- Only these two shas exist under `_scratch`: 454 copies of 06b95093 and 168 of 0997f410.

**Structure:**
- These parts are identical: vocab (248,044 entries), merges (247,587), the 33 added tokens (248,044-248,076), the NFC normalizer and the post-processor.
- **Differences:**
  - **Pre-tokenizer regex:** 06b95093 uses `\p{L}+` / `[^\s\p{L}\p{N}]`; 0997f410 uses `[\p{L}\p{M}]+` / `[^\s\p{L}\p{M}\p{N}]`. Only combining marks are affected.
  - **ByteLevel decoder options.**

**On the data, the ids are identical:**
- Duck: 1,331 / 1,331 requests (3,831,991 tokens) give the same ids with 06b95093, 0997f410 and transformers 5.20.2 `AutoTokenizer`.
- Bench: re-encoding the server-decoded text reproduces the saved token ids exactly in 187 / 187 responses, with all three tokenizers.
- So the server's generated ids are canonical BPE, and coverage on saved ids equals coverage on re-tokenized text (also checked per table).

**Used:** 06b95093 (klee100), through `tokenizers.Tokenizer.from_file`. Saved token ids were used directly for the bench sets.

## 5. Method

### 5.1 Always-in set (352 ids)
- 33 added / special tokens (248,044-248,076): pad `<|endoftext|>`, `<|im_start|>`, eos `<|im_end|>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`, `<think>`, `</think>`, fim / vision / audio / tts markers.
- 256 byte tokens: the ByteLevel base characters. This BPE has `byte_fallback=false`, so these 256 ids are the byte level.
- 73 numeric tokens: every vocab entry whose decoded text, without spaces, is all ASCII digits, or a single Unicode numeric character.

### 5.2 Fill order
After the always-in set:
1. generated-token frequency, descending;
2. prompt-context frequency (the system, tools, user and tool-response sections of the build tasks' `prompt.log`, counted by script), descending;
3. ascending token id (a stand-in for BPE merge rank) among ids < 248,077.

Ties are broken by id. Padding rows 248,077-248,319 are never generated (0 such ids in 408k saved bench ids) and are never included.

### 5.3 Fill variants tested in the 2-fold split
Held-out coverage, %:

| N | gen_only (E26 only) | gen_ctx (+ prompt context) | **gen_plus_e25** (+ E25 Duck of the build tasks + context) | id_only reference (no frequency) |
|---|---|---|---|---|
| 16,384 | 99.239 | 99.242 | **99.356** | 91.530 |
| 24,576 | 99.427 | 99.427 | **99.538** | 94.579 |
| 32,768 | 99.556 | 99.557 | **99.642** | 95.968 |
| 49,152 | 99.717 | 99.718 | **99.781** | 97.225 |
| 65,536 | 99.865 | 99.865 | **99.906** | 98.715 |

- `gen_plus_e25` is best at every N, so the shipped tables use it, built on all 30 tasks.
- Frequency data adds +7.8 points over the id-only table at 16k and +1.2 at 64k.

**Distinct generated tokens are few.**
- In each fold, build has 9,165 / 9,047 distinct ids. The held-out half has 2,418 / 2,536 distinct ids unseen in build, which make up 16,062 / 14,824 tokens (1.26 % / 1.18 %).
- The final tables have:
  - 13,376 ids from generated frequency;
  - 104 from prompt context;
  - the rest from ascending-id fill: 2,904 at 16k, 11,096 at 24k, 19,288 at 32k, 35,672 at 48k, 52,056 at 64k.
- For N ≥ 16k the tail is therefore chosen by the id prior, not by our data. The id prior still catches unseen tokens (99.36 → 99.91 % from 16k to 64k).
- In-sample E26 coverage is 100.000 % for every table.

### 5.4 Segment labels
Labels are token-level and set inside the script:
- **Protocol:** template / tool scaffolding.
- **Code:** values of `python` tool parameters.
- **Final answer:** content plus `submit` parameter values.
- **Thinking:** the reasoning text.
- **Numbers / grids:** inside thinking or code, any all-digit token, or any token made only of ` ,[]()\n\t` that touches a digit token. This label overrides thinking / code.

E26 totals:

| segment | tokens | share |
|---|---|---|
| thinking | 1,440,799 | 56.8 % |
| numbers / grids | 926,953 | 36.5 % |
| code | 144,974 | 5.7 % |
| protocol | 22,454 | 0.9 % |
| final answer | 2,176 | 0.09 % |

## 6. Held-out coverage per table (chosen variant, mean of 2 folds)

| N | overall | fold A→B | fold B→A | thinking | code | numbers / grids | protocol | final answer | misses / 1k tokens |
|---|---|---|---|---|---|---|---|---|---|
| 16,384 | 99.356 | 99.324 | 99.388 | 99.069 | 97.999 | 99.999 | 100.000 | 99.011 | 6.44 |
| 24,576 | 99.538 | 99.498 | 99.577 | 99.325 | 98.617 | 100.000 | 100.000 | 99.295 | 4.62 |
| 32,768 | 99.642 | 99.609 | 99.676 | 99.479 | 98.915 | 100.000 | 100.000 | 99.674 | 3.58 |
| 49,152 | 99.781 | 99.738 | 99.823 | 99.679 | 99.355 | 100.000 | 100.000 | 99.810 | 2.19 |
| 65,536 | 99.906 | 99.911 | 99.901 | 99.873 | 99.625 | 100.000 | 100.000 | 99.864 | 0.94 |

Where the misses fall (% of held-out misses):

| N | thinking | code | numbers / grids | protocol | final answer |
|---|---|---|---|---|---|
| 16,384 | 82.1 | 17.7 | 0.0 | 0.0 | 0.1 |
| 24,576 | 82.8 | 17.1 | 0.0 | 0.0 | 0.1 |
| 32,768 | 82.6 | 17.3 | 0.0 | 0.0 | 0.1 |
| 49,152 | 83.1 | 16.8 | 0.0 | 0.0 | 0.1 |
| 65,536 | 77.0 | 22.8 | 0.0 | 0.0 | 0.1 |

Code has the highest miss rate per token. Thinking has most of the misses by count. The final-answer segment is tiny (2,176 tokens), so its numbers are noisy.

Joint windows (W_k = probability that k consecutive generated tokens are all in the table) and other held-out sets, %:

| N | W1 | W2 | W3 | E25 NB5/NB6 Duck (other checkpoint, tasks not in the build fold) | E27 bench (klee100 ids) | E25 k3_v2 bench (klee100 ids) |
|---|---|---|---|---|---|---|
| 16,384 | 99.356 | 98.732 | 98.122 | 99.417 | 99.669 | 99.634 |
| 24,576 | 99.538 | 99.089 | 98.649 | 99.541 | 99.834 | 99.808 |
| 32,768 | 99.642 | 99.293 | 98.949 | 99.677 | 99.872 | 99.844 |
| 49,152 | 99.781 | 99.564 | 99.349 | 99.800 | 99.928 | 99.897 |
| 65,536 | 99.906 | 99.812 | 99.719 | 99.904 | 99.960 | 99.950 |

- The shipped full-data tables on the benches:

| N | E27 bench | E25 k3_v2 bench |
|---|---|---|
| 16k | 99.751 % | 99.694 % |
| 24k | 99.892 % | 99.849 % |
| 32k | 99.906 % | 99.863 % |
| 48k | 99.946 % | 99.910 % |
| 64k | 99.968 % | 99.954 % |

- No EOS / pad id appears in either bench (`ignore_eos` runs), so cutting at the first EOS changes nothing.
- **The synthetic bench text is easier to cover than real Duck: its miss rate is 2-3× lower.** GPU speed arms on the bench will understate the Duck acceptance loss.

## 7. Acceptance-effect model

**Baselines:** unconditional per-position acceptance P_k (vLLM "Per-position acceptance rate"), with AL = 1 + Σ P_k.

| baseline | P1 | P2 | P3 | AL | conditionals |
|---|---|---|---|---|---|
| E26 K100Q real Duck, klee100, measured here from `server_merlin_fp8_mtp_pin508.log` (908,916 drafts) | 0.7906 | 0.5926 | 0.4313 | 2.8145 | 0.750 / 0.728 |
| NB5 FP8 Duck | 0.795 | 0.599 | 0.439 | 2.833 | |
| klee100 synthetic 20k sweep (the 0.864 / 0.731 / 0.607 in the brief) | 0.864 | 0.731 | 0.607 | 3.202 | |

The brief's "AL about 2.8 on real Duck" is the E26 / NB5 row. With these unconditional rates, the 0.864 / 0.731 / 0.607 triple gives AL 3.20 (the synthetic bench), not 2.8.

**Model.** A drafted token outside the table is treated as rejected: q is 0 outside the table, and the target reaches those tokens only through the residual. Position k is accepted only if it was accepted at baseline and all k target tokens are in the table:
- **joint model** (main): AL' = 1 + Σ_k P_k · W_k;
- **independent model:** AL' = 1 + Σ_k P_k · c^k;
- **worst-case bound:** AL' ≥ 1 + Σ_k max(0, P_k − (1 − W_k)). It follows from α' = Σ_{x∈T} min(p, q/Q_T) ≥ α − Σ_{x∉T} min(p, q).

| N | real Duck (E26), joint | independent | Δ joint | worst-case | NB5 FP8 Duck, joint | synthetic 3.202, joint | E27 bench (synthetic base, bench W_k) |
|---|---|---|---|---|---|---|---|
| 16,384 | 2.7938 | 2.7935 | −0.74 % | 2.7766 (−1.35 %) | 2.8120 (−0.74 %) | 3.1758 (−0.82 %) | 3.1884 (−0.43 %); worst 3.1823 (−0.61 %) |
| 24,576 | 2.7996 | 2.7994 | −0.53 % | 2.7873 (−0.97 %) | 2.8179 (−0.53 %) | 3.1831 (−0.59 %) | 3.1951 (−0.21 %); worst 3.1921 (−0.31 %) |
| 32,768 | 2.8029 | 2.8028 | −0.41 % | 2.7933 (−0.75 %) | 2.8213 (−0.41 %) | 3.1874 (−0.46 %) | 3.1967 (−0.16 %); worst 3.1944 (−0.24 %) |
| 49,152 | 2.8074 | 2.8073 | −0.25 % | 2.8014 (−0.46 %) | 2.8258 (−0.25 %) | 3.1930 (−0.28 %) | 3.1990 (−0.09 %); worst 3.1977 (−0.13 %) |
| 65,536 | 2.8114 | 2.8114 | −0.11 % | 2.8089 (−0.20 %) | 2.8299 (−0.11 %) | 3.1981 (−0.12 %) | 3.2003 (−0.05 %); worst 3.1996 (−0.07 %) |

The joint model is likely pessimistic: out-of-table (rare) tokens are the ones the drafter already misses most. Renormalizing q over the table can also slightly raise acceptance of in-table tokens.

## 8. Head memory and expected head-time effect

**Extra head memory per PP1 rank (TP2)** = N × 2560 × 2 B / 2:

| N | 16k | 24k | 32k | 48k | 64k |
|---|---|---|---|---|---|
| extra head memory | 40.0 MiB | 60.0 MiB | 80.0 MiB | 120.0 MiB | 160.0 MiB |

- The full BF16 head is 606.25 MiB per rank.
- The draft-owned head is extra; the target's full `lm_head` stays.
- **Also allocated:** a persistent scatter buffer `_draft_logits_full` of max_num_seqs × 248,320 × 2 B per last-stage rank (unsharded):

| max_num_seqs | 24 | 32 | 64 |
|---|---|---|---|
| buffer size | 11.4 MiB | 15.2 MiB | 30.3 MiB |

- **Scatter cost:** writes of about 36 MB per tick at 24 rows × 3 steps, about 0.1 ms. This is not in the saving column.

**Head-read time** (projection, not measured):
- Inputs: the measured 7.5 ms for the 3 BF16 draft lm_head reads (E25 progress report: stage 1 about 33 ms, W4, 8 streams), scaled linearly by bytes. The I8 full head is taken as 0.626 × BF16 (A7 bytes ratio).
- With reduced N, the 3 reads take 0.49 / 0.74 / 0.99 / 1.48 / 1.98 ms (16k → 64k).
- Saving vs BF16: 7.01 / 6.76 / 6.51 / 6.02 / 5.52 ms.
- Saving vs I8: 4.20 / 3.95 / 3.71 / 3.21 / 2.72 ms.

**Throughput** (rough): assume stage 1 binds at 33 ms (30.2 ms with I8) and stays binding down to stage 0's 21.6 ms, times the AL ratio:

| N | vs BF16 full head | vs I8 full head (production Q) |
|---|---|---|
| 16k | ×1.26 | ×1.15 |
| 24k | ×1.25 | ×1.15 |
| 32k | ×1.24 | ×1.14 |
| 48k | ×1.22 | ×1.12 |
| 64k | ×1.20 | ×1.10 |

These are upper-side projections: the klee100 stage split at 16-24 streams is unmeasured (item 1-9). Small-N GEMMs may be launch-bound. The draft sampler still works over the full vocab.

## 9. Loader check (Merlin's own code)

`models/qwen4_exp/nvidia/draft_head.py` was imported directly from `_scratch/e23/p2/merlin-src/vllm/`. It imports only hashlib, io, os, re and torch, so no stubs were needed. Results for each file:
- `load_draft_vocab_ids(path, 248320, tp_size=1 / 2 / 4)`: OK, int64, correct N;
- with `VLLM_DRAFT_VOCAB_SHA256` = file sha: OK;
- with a wrong sha: refused (ValueError);
- `draft_vocab_table_factor` equals `draft_vocab_fingerprint`;
- `d2t_offsets` round-trips;
- `select_head_rows` + `scatter_to_full`: correct rows, and exactly V − N `-inf` columns per row.

The loader rules read from source:
- 1-D int32/int64 tensor;
- 0 < N < V;
- ids in [0, V);
- strictly increasing;
- N divisible by 128 × TP.

The head lives only on the last PP stage. `tie_word_embeddings` must be false: klee100 has `false`. Synthetic and adaptive verification are refused while the knob is set (`mtp.py` 525-600, `config/speculative.py` 1881-1928).

## 10. Caveats

1. **Source of the frequencies.** The tables come from evaluation-task responses: 30 tasks, about 3.8 M tokens. No training-task responses exist locally. A frequency table holds no solutions.
2. **The tail is set by the id prior.** Only about 13.4k distinct generated ids exist, so every table's tail (2.9k-52k ids) is filled by ascending id. More saved Duck runs would sharpen it. The 2-fold numbers already include this effect.
3. **AL is modeled, not measured.** The joint model assumes independence; the worst-case bound is given beside it. Real numbers need an MTP arm on real Duck (NB-S3), not only the synthetic bench: the bench has a 2-3× lower miss rate.
4. **Head-time savings are projections** from one 7.5 ms stage-1 measurement (W4, 8 streams), scaled linearly by bytes.
5. **The secondary Duck sets come from a different checkpoint** (W4A16 Merlin). Their held-out coverage agrees with E26 within ±0.06 points.
6. **Segment labels are heuristic token-level markers.**
7. **Re-saving changes the sha.** The file sha256 depends on torch serialization; pin the shipped files.
8. **"65k + int8 head" is not checked here.** Whether the reduced draft-owned head combines with `draft_lm_head_quant` is for the kit to verify (`speculator.py:118-135, 585-605`).

## 11. Scratch

`D:/ARC AGI 2/_scratch/prep_vocab/` holds:
- structure-only inspectors: `inspect_keys.py`, `agg_e26.py`, `agg_e27.py`, `scan_sources.py`, `scan_dirs.py`, `tok_inspect.py`;
- run logs: `run1.log`, `run2.log` (counts only);
- report helpers: `summ.py`, `gen_tables.py`.

Nothing existing was modified. No GPU, Kaggle or git was used.