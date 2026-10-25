# Rev 2 of the load path, and the re-prefill root cause found in Duck (Claude, 2026-10-23)

**Context:**
- The user confirmed that foysal, hosen42 and esfersami50 are separate people on one team. This is saved in memory.
- The user then said "শুরু করো" ("start") for the CPU work listed in Notes/10.

**Scope of this work:**
- No GPU, no Kaggle push, no notebook built.
- No Duck code changed. The fix in section B needs the user's approval.

## A. E14-0 load path, rev 2

**Files:**
- `src/e14_loadpath.py` (893 lines);
- `src/test_e14_loadpath.py` (**56 tests**);
- before/after diff: `Experiment Review/E14-0 load path rev1 to rev2.diff`;
- the rev 1 copy is in the scratchpad (`e14_rev1/`).

**Result:** **56 passed, in 6 of 6 full runs.**

| Change (Notes/10 item) | What the code does now | Test(s) |
|---|---|---|
| Short-write bug (A5, B9) | `_write_all` loops until every byte is written; a 0-byte write raises | `test_short_writes_still_give_an_exact_copy` (each write writes 1/3), `test_write_all_zero_write_raises` |
| Atomic publish (B5) | Copy into `<name>.partial/`, verify, then one `os.replace` to `<name>/`. Any failure deletes both. Stale dirs from earlier attempts are deleted first | `test_stale_partial_and_final_are_replaced`; `no_dirs_left` in every failure test |
| Byte integrity (B6) | blake2b of each range, from the source bytes as copied; after the copy every destination range is re-read and compared. Small files are compared byte for byte. Full sha256 per .safetensors is recorded and compared with `expected_sha256` when given | `test_one_byte_flip_is_caught_by_range_hash`, `test_sha256_recorded_and_expected_checked` |
| Safetensors layout (B7) | Sorted by begin; first begin 0; no hole or overlap; last end == buffer; bytes == dtype × shape. A problem the source itself has is accepted as published and reported | `test_safetensors_layout_rules` (header order ≠ offset order passes; hole, overlap, wrong size, bad start, extra buffer, 2-byte file and bad JSON fail), `test_layout_problem_of_the_source_is_accepted_as_published` |
| Ranges cover [0, size) (B8) | Asserted in `_ranges` | `test_ranges_cover_exactly` |
| Failure injection (B10) | — | zero-byte read once (retried, `retries == 1`) / forever (no dir); ENOSPC (no dir; `stage_models` fails open and warms); worker exception (no dir); hung read; stall; deadline; transient EIO |
| Worker ramp (B4) | Start at 4. After 512 MiB at a level, go 8 → 16 → 24 only if MB/s rose ≥ 15 %; otherwise drop back one level and stay. A retry also drops back one level and stays. The levels and MB/s go into the stats | `test_ramp_goes_up_while_throughput_scales` (2 → 4 → 8), `test_ramp_falls_back_when_storage_saturates` (a 2-stream limit: back to 2; active workers never above the allowed level), `test_ramp_backs_off_on_retry` |
| cgroup memory (B3) | `cgroup_memory()` (v2 `memory.max`/`memory.current`, v1 `limit_in_bytes`/`usage_in_bytes`, "max" and huge v1 values = unlimited). `host_headroom()` = min(MemAvailable, limit − usage). /dev/shm needs headroom ≥ copy + 24 GiB | `test_cgroup_v2_v1_and_headroom`, `test_dev_shm_skipped_when_cgroup_is_tight` |
| Phase record (B12) | `E14State`: PREPARE → COPY → VALIDATE → LAUNCH → READY / FALLBACK, with `deadline_in`, info and `terminal_reason`. Written as JSON (write + rename) at every change | `test_state_record_written_at_every_change` |
| No second source read (B12) | A model copied and verified is used even if a later model's copy fails; only the failed ones are warmed in place | `test_copied_model_is_not_read_again_after_a_later_failure` |
| Off switch (B1) | `E14_LOADPATH=0` → nothing is copied; the notebook keeps the old `cat` prefetch | `test_off_switch` |
| JIT fingerprint (B14) | `jit_manifest()` requires 13 fixed fields: sglang, sgl_kernel, flashinfer, torch, triton, cuda_runtime, nvcc, driver, gpu_name, sm, wheels, jit_sources, env. A missing or unknown field raises. `wheel_inventory` (wheel names + sizes; hashing 5 GB on GCSFUSE would take minutes). `source_fingerprint` (sha256 of the installed .py/.cu/.cuh/.h/... files). `kernel_env` (kernel-affecting env vars, cache paths excluded). A restore names the fields that differ | `test_jit_cache_manifest_mismatch` (sglang, driver, sources, env each reject), `test_jit_manifest_must_be_complete`, `test_kernel_env_and_fingerprints` |

**Kept from rev 1:**
- The stall watchdog, with rchar + page faults.
- The daemon threads: a hung read never blocks exit.
- Fail-open staging.
- The traversal-safe tar.
- The known-answer check.

**Still not tested (unchanged honest list from Notes/07):**
- real Linux `/proc` and cgroup (fake trees only);
- the real GCSFUSE speed;
- the /dev/shm size and RAM on Kaggle;
- SGLang loading from tmpfs;
- the JIT time saving.

## B. Re-prefill: what really happens (server numbers, not reconstruction)

**Data:**
- the `usage` list of all 16 G4B rev 6d branches (`branch_result.json`): prompt, completion and **cached** tokens for every request, as reported by the server;
- 150 requests that have a predecessor.

**Script:** `src/analysis/e14_cache_accounting.py`. Rows are in `Experiment Results/G4B_rev6d_cache_accounting_rows.json`.

**How the classes are defined:** cache hits land on a 64-token grid.
- **reply_reused:** cached ≥ floor((prompt_prev + completion_prev)/64) × 64 − 64.
- **reply_lost_at_prefill_end:** cached within 64 of floor(prompt_prev/64) × 64.
- **deeper_miss:** cached more than 64 below that.
- **partial_reply:** in between.

**Request classes:**

| Class | Requests | Prefix tokens prefilled again |
|---|---|---|
| reply_lost_at_prefill_end | 68 (45 %) | 217,541 |
| reply_reused | 49 (33 %) | 3,446 |
| deeper_miss | 18 (12 %) | 66,771 |
| partial_reply | 14 (9 %) | 2,442 |
| history_shrank (trim/reset) | 1 (1 %) | 23,099 |
| **Total** | 150 | **313,299 of 397,650 uncached prompt tokens (79 %)** |

**Where each class stops:**
- **Lost replies** stop at the end of the previous prompt, at cached − prompt_prev = −63 … +2 (median −34). That holds even for 7,088-token replies, so the match breaks inside the first ~64 tokens of the reply.
- **Reused replies** are matched almost to their end (for example 3,621 → 3,525).
- **Partial replies** stop inside the reply.

### Root cause found (strong evidence, not yet exact proof)

**Duck rewrites the model's reasoning before it goes back into history.**

- `vendor/inference/agent/tool_agent.py` L3414-3418, `_normalize_message_content`:
  - `"\n".join(line for line in cleaned.splitlines() if line.strip())`, then `.strip()`;
  - it also removes `<think>`/`</think>` tags (L93).
- `_extract_reasoning_text` (L3431-3435) applies it to the server's `reasoning_content`.
- That normalized text is what is stored: `assistant_message["reasoning_content"] = reasoning` (L6904-6908, key from `ARC3_REASONING_HISTORY_KEY`, arc2_duck.py L41).
- **So every blank line the model wrote in its reasoning is deleted from the history.** The re-rendered prompt differs from the generated tokens at the first paragraph break, and the radix/Mamba match falls back to the last checkpoint before it.
- arc2_duck does not override this. It only calls the same functions for metrics (L1643-1666).

**Evidence:**
1. **0 of 180** `[THINKING]` blocks in the G4B transcripts contain a blank line. Each branch has 42,762–146,150 characters of reasoning. Natural reasoning without a single paragraph break is practically impossible, so the stored text is the normalized one. The transcript writer (L6906) gets the same normalized `reasoning`.
2. **Reasoning length separates the classes** (from META `reasoning_chars`, i.e. the normalized length):

   | Class | Median reasoning chars | Requests with ≤ 300 chars |
   |---|---|---|
   | reply_reused | 116 | 41/49 |
   | reply_lost_at_prefill_end | 1,712 | 12/68 |
   | partial_reply | 521 | 3/14 |
   | deeper_miss | 2,046 | 4/18 |

   - Long replies that were still reused mostly had short reasoning and long code. Tool-call arguments are not normalized. Examples: (reasoning 136 chars, completion 391), (241, 695), (298, 569).
   - Two reused replies had long reasoning: (2,121, 1,093) and (7,940, 3,621). That is possible only if their reasoning had no blank line, which cannot be checked offline.
3. **This is why my token-diff (Notes/06) found 0/170.** It compared normalized text with normalized text. GPT notes 08/09 warned that reconstruction cannot prove the generated sequence. This is a concrete case.

**What it does not explain:**
- **12 lost replies with ≤ 300 chars of reasoning.** A short text can still contain a blank line; otherwise these are the other candidates (non-canonical sampled tokens, Mamba checkpoint mechanics).
- **The 18 deeper misses (66,771 tokens), a separate cause.**
  - 6 of them fall on the 2nd request (k = 1), at cached 1,536–1,664 of a 2,186–4,575-token first prompt, i.e. inside the system/first user message.
  - So something in an **earlier** message changes between requests: a status/budget line, the grid-row suppression (`grid_rows_suppressed` 6–93 per branch), or the opener sentinel rewrite (`resume_rewritten_after_trim`: 4 events).
  - Only 1 trim (`trim_reasoning_only_blocks`) occurred.
  - **Can be found exactly on CPU:** run Duck against the local fake server (E11 harness), record every request's message list, and diff consecutive requests.

**What a fix is worth (G4B scale):**
- **Reply loss:** 217.5k + 2.4k tokens prefilled again. At the measured ~1,500 tok/s prefill, that is ~146 s of server prefill. Prefill blocks decode for every running request.
- **Deeper misses:** another 66.8k tokens (~45 s) if their cause is also a client edit.
- Together this is ~72 % of all uncached prompt tokens.
- The README's earlier estimate (G4B ~+10 % effective) stays the working number until it is measured.

**Fix options (Duck change, needs the user's approval; rev 6g rule):**
1. **Minimal:** store the server's **raw** `reasoning_content` in the history message. Keep the normalized text only for the transcript and metrics.
   - The template still applies `|trim` to the ends, so only a leading or trailing whitespace difference would remain.
   - In arc2_duck this is one override at the point where the assistant message is built.
2. **Complete:** token-level continuation (SGLang streaming sessions or token ids). Bigger change.

**Verification plan (E14-A, a few minutes, our own replay client, no Duck change):**
- Replay one recorded conversation twice, once with raw reasoning and once with the Duck-normalized reasoning. Compare `cached_tokens` and use `return_output_ids_in_sglext` to get the exact first divergent id.
- Expected: raw ≈ reply reused; normalized ≈ prefill end.

### Same code in the Franzen ARC-AGI-3 notebook (the user asked, 2026-10-23)

**Checked:** `dfranzen/arc-agi-3-milestone-2-solution`, pulled 2026-10-23 into `_scratch/ext/dafr_nb_pull/`. The user reports 34.30 for V1; his write-up states 27.89 for his first submission.

**What the notebook is:** Tufa's original Duck bundle (commit 7652836, attached unchanged) plus his patch (cell 2).

**What it does with reasoning:**
- The patch never changes `_normalize_message_content` / `_extract_reasoning_text`. They appear only as unchanged context lines. So the blank-line deletion is in **Tufa's original harness** and in his submission.
- It sends the reasoning back as `ARC3_REASONING_HISTORY_KEY=reasoning_content` (cell 4), as we do.

**His own observation (WRITEUP §2):** "due to re-tokenization, the tokenized conversation might not exactly match the generated token sequence from before". His server patch therefore keeps the end-of-prefill Mamba checkpoint and refreshes its LRU position. That limits the loss to the previous reply instead of the whole history. **He did not remove the cause.**

**What it costs him (estimate):**
- On RTX PRO 6000, prefill runs at 10,553 tok/s (p90) against 836.6 decode, so one re-prefilled token costs ≈ 0.08 of a generated token.
- He measured 93.43 % token reuse and a 15.25 % prefill time fraction. That 15.25 % also covers first prompts and trims.
- His replies are short: yield every 2,048 generated tokens, cap 12k.
- So the reply re-prefill can cost him at most a part of that 15.25 %, probably a few percent. It is not what limits his score. This cannot be measured without his request logs.

**What it costs us:**
- On 4×L4, prefill is 1,500 tok/s against ~400 decode (≈ 0.27 per token, 3–4× his relative cost), and our ARC-AGI-2 replies are long.
- G4B: 45 % of requests lost the reply, 217.5k tokens, ≈ 146 s of prefill during which every running stream's decode waits.
- It is a throughput (coverage) brake, not an accuracy cap. It affects only the Duck line; our best LB 34.58 (S1-60K) does not run Duck.
- **Unknown:** whether the model reasons differently when it reads its own old reasoning without paragraph breaks. Tufa and Franzen scored well with it; no evidence either way.

## C. Corrections to earlier notes

- **Notes/06:** "the template is not the cause" stays true for the template. But the reply re-prefill **is** caused on the client side, by Duck's reasoning normalization (strong evidence). Notes/06 got a pointer to this note.
- **Notes/10 A4:** the after-trim share is now measured: 1 request with a shrunk history (23,099 tokens) plus up to 4 opener rewrites. The bigger unexplained part is the 18 deeper misses.
