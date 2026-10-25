# Claude's validation of GPT notes 08 and 09, and what changes in the plan (2026-10-23)

**Inputs:**
- `Notes/08 GPT - review of the final report and P0 decision, fourth note`.
- `Notes/09 GPT - independent assessment with five gaps, fifth note`.
- Both are read in full; the verbatim copies are kept.

**Checked against:**
- our code: `src/e14_loadpath.py` and the token-diff script;
- the SGLang 0.5.21 source;
- the Duck vendor code;
- the INT4 tokenizer;
- the safetensors format rules.

**Status:**
- Changes to code are listed as **to do**. The user asked only for the notes to be updated this turn, so no code was changed.
- No GPU, no Kaggle push.

## A. New facts found while checking (not in any earlier note)

1. **Exact token ids can be recorded without touching Duck.**
   - SGLang 0.5.21 has server-wide flags `--return-input-ids` and `--return-output-ids` (`arg_groups/fields/serving.py` L193-200).
   - It also has per-request `return_input_ids_in_sglext` / `return_output_ids_in_sglext` and the headers `x-sglext-return-input-ids` / `-output-ids` (`entrypoints/openai/serving_chat.py` L786-791, L1164-1170). The ids come back in the response's `sglext` field, including when streaming (L1948-1994).
   - **But Duck does not store the raw response.** It does `payload = response.json()` (`tool_agent.py` L5325), and its transcript keeps only meta (`raw_tool_calls`, L2548).
   - So exact ids from a real Duck run would need a Duck logging change, which needs the user's approval.
   - Without a Duck change, the E14-A replay client (our own code, not Duck) can set the flag per request.
2. **The ambiguity GPT describes is real for our tokenizer** (tested on the INT4 tokenizer fixture). Each pair below decodes to the same text, and re-encoding always gives the canonical ids:

   | Text | Canonical ids | Different ids, same text |
   |---|---|---|
   | `"\n\n"` | `[271]` | `[198, 198]` |
   | `"\n\n\n"` | `[1358]` | `[198, 198, 198]` |
   | `"</think>\n\n"` | `[248069, 271]` | `[248069, 198, 198]` |
   | `" grid"` | `[5646]` | `[220, 70, 81, 72, 67]` |
   | `"def f(grid):"` | 4 ids | 12 ids |

   - So my check C ("reply tokens equal", 170/170) is a **reconstruction** result. It cannot show the model emitted the canonical ids.
   - The reply boundary `</think>` + `\n\n` is exactly where a `198 198` vs `271` split could occur. That keeps "non-canonical sampled tokens" as candidate 1.
3. **Verified: `preserve_thinking=True` in production,** both as the server default (`build_e12_duck.py` L313, `--default-chat-template-kwargs {"preserve_thinking": true}`) and in each request (`ARC3_PRESERVE_THINKING` defaults to "1", `tool_agent.py` L1326).
   - With it the template keeps the reasoning of every old turn (`chat_template.jinja` L116).
   - So the token-diff script's `preserve_thinking=True` matches production.
4. **The token-diff scope was narrower than I reported.** The script skipped four kinds of turn:
   - malformed or empty tool calls (no `raw_tool_calls`);
   - an assistant turn not followed by a tool result (the episode's last turn, plain replies);
   - turns after a history trim.

   **Turns after a trim:** Duck rev 6 trims whole old blocks (arc2_duck.py L1501-1546). After a trim the prompt prefix changes by design, so the request MUST re-prefill from the trim point. That is a fourth cause, separate from the three in Notes/06, and the share of re-prefill it explains is not yet counted.
5. **A real bug in my copy code** (GPT gap 2, item 2):
   - `_copy_range` ignores the return value of `fo.write(b)`. With `buffering=0`, a raw write can be partial.
   - A partial write would leave unwritten bytes in a preallocated file. The head/tail check could miss them if they are in the middle.
   - It must loop until every byte is written.

## B. Point-by-point verdict on GPT notes 08 and 09

| # | GPT point | Verdict | Status / change |
|---|---|---|---|
| 1 | Keep the working loader as fallback; new path optional | Correct | Already: `stage_models` never raises and returns the original paths on any failure. To add: an off switch (`E14_LOADPATH=0`) so the old `cat` prefetch path stays reachable |
| 2 | Do not assume /dev/shm holds the model | Correct | Already: `choose_dest` free-space + MemAvailable ≥ size + 24 GiB check, then /tmp, then in place |
| 3 | Host/cgroup memory after the copy, not only MemAvailable | **Correct, and a gap** | `/proc/meminfo` in a container can show the host, not the cgroup limit. Add: read `/sys/fs/cgroup/memory.max` and `memory.current` (cgroup v2) or `memory.limit_in_bytes` (v1), take the smaller headroom, and log both |
| 4 | Do not default to 16–32 workers; sweep first | **Accepted, idea changed** | Default 24 dropped. New: adaptive ramp 4 → 8 → 16 (cap 24). Each step is measured over ≥ 512 MiB of real copy work and kept only if MB/s rises ≥ 15 %. Any error or stall at a level → back off one level. The data copied during the ramp counts (no wasted reads) |
| 5 | Atomic publish via temporary name + rename | **Accepted** | Copy into `<dst>.partial/`, rename to `<dst>/` only after validation. A launch can never see a partial directory, even after a kernel crash mid-copy |
| 6 | Byte checksum, not only size + header | **Accepted, idea changed** | Head/tail 1 MiB is too weak. New: a blake2b hash of each 64 MiB range, computed from the source bytes as they are copied. After the copy, every destination range is re-read (tmpfs, fast) and its hash compared. Plus a full sha256 per shard, recorded on the first run and compared on later runs (catches FUSE returning bad bytes) |
| 7 | "Last JSON entry end offset" is unsafe | **Does not apply to our code**, but the stronger check is right | We already take `max(end)`, not the last entry. To add, following the format rule that the byte buffer is fully indexed with no holes: sort tensors by begin; first begin == 0; each end == the next begin; last end == buffer size; and end − begin == dtype size × prod(shape) |
| 8 | Ranges disjoint and cover [0, size) | Correct (true by construction) | Add the explicit assert anyway (cheap) |
| 9 | Short reads/writes handled | Reads: already. **Writes: bug** (A5) | Fix: write loop |
| 10 | Failure-injection tests: interrupted copy, zero-byte read, short write, one-byte corruption, ENOSPC, malformed header, worker exception | Partly covered | Covered: hung read, stall, deadline, transient EIO, unwritten tail, truncation. To add: zero-byte read, short write, one-byte flip in the middle, ENOSPC, malformed header, worker exception → none may leave a valid-looking final directory |
| 11 | No `posix_fadvise(DONTNEED)` yet | Agree | Not planned |
| 12 | State machine PREPARE → COPY → VALIDATE → LAUNCH → READY; absolute deadline and terminal reason per phase; no second source read after a good copy; check remaining time before every fallback | **Accepted** | Our code has the pieces but no single record. Add: `E14_STATE` with phase, t_start, deadline and terminal_reason, written to `/kaggle/working` at every change (JSON, flushed), so a killed run still shows where it stopped |
| 13 | First-cell instrumentation cannot prove anything when the runner never reaches the notebook | **Correct** | The 10-22 G4B ERROR (empty log) is that case. Its cause stays "unknown, mount stall likeliest", never "confirmed". The instrumentation helps only the slow-storage case |
| 14 | JIT fingerprint must cover SGLang, Triton, CUDA, torch, sm, driver, kernel source | **Accepted** | The manifest gets a fixed field list (not caller-defined): sglang, sgl-kernel, flashinfer, torch, triton versions; CUDA runtime and nvcc versions; driver version (nvidia-smi); GPU name and sm; the sha256 of every wheel in the wheelhouse (stands in for kernel source); plus the env flags that change kernels. Any mismatch → cold compile. Known-answer test after restore stays |
| 15 | `nv/` exclusion is not compatibility proof; #42917 was on another checkpoint/GPU | Agree | It is a safety measure, not evidence that our checkpoint is affected |
| 16 | Token-diff: name it `reconstructed_divergence`; exact ids only from a GPU run | **Accepted, Notes/06 corrected** | Notes/06 now says: reconstructed divergence 0/170 on tool-call turns. That is no proof about the generated ids (A2) |
| 17 | Test with two different token sequences of the same text | Accepted, and done in spirit (A2) | Add it as a unit test of the token-diff tool. The tool must label a canonical match "reconstructed", never "identical ids" |
| 18 | Separate cases: plain reply, long thinking, submit(), check(), malformed call, history trimming | **Accepted** (A4) | Extend the script: classify every assistant turn (tool call / plain / malformed / last / after-trim) and report counts and reconstructed divergence per class. `submit()` / `check()` are calls inside the python tool, so they were already in the 170, but not counted separately |
| 19 | Not all of the 245k re-prefilled reply tokens are recoverable | Correct | Split per cause: after-trim (unavoidable as built) / boundary tokens / checkpoint eviction |
| 20 | E14-A: measure the unchanged W1 baseline first, then #44401, then graph bs 1..13 | **Accepted, plan changed** | Three arms in order, in one session, same requests and logs. The patch applies only if the target function's source hash equals the expected 0.5.21 hash; on a mismatch: fail loudly, skip the arm, never patch silently |
| 21 | Scheduler correctness is not "higher TPS" | Accepted | Gate: held-row count during chunked prefill, never negative free-slot accounting, 12 genuine running (server metrics), a replay of the 11-of-12 latch case |
| 22 | McNemar p ≥ 0.01 is not equivalence; same prompts ×150 ≠ 150 tasks | **Correct, gate redesigned** | See section C |
| 23 | NCCL matrix too large for 35–45 min | **Accepted** | First round: 3 arms only: baseline, `NCCL_P2P_LEVEL=SYS` (only if the topology check shows P2P is possible), `NCCL_SHM_USE_CUDA_MEMCPY=1`. Further knobs only after the first result |
| 24 | Re-read quota and timer just before launch; 80-min cap ≈ 2 h 40 quota, 90 min ≈ 3 h | Agree | Already a standing rule; the written estimate goes to the user before any launch |
| 25 | Several Kaggle accounts only if they are real team members' own accounts; one person may not submit from several accounts (competition rules) | **Rules point, noted** | I cannot verify account ownership. Before more runs on foysal / hosen42 / esfersami50, the user should confirm these are separate team members' own accounts, used within the team |
| 26 | Ratings of report claims (115–140 ms measured; 140 all-reduces = static count; 35–55 ms = estimate; 2×TP2 170k = sizing estimate; 600–800 = target; G4B cause = probable) | Agree with every rating | The README's key-findings wording keeps measured vs estimated apart |

## C. Correctness gate, redesigned (replaces "D error ≤ 2 % and McNemar p ≥ 0.01")

The gate reports three results separately and never passes on a p-value alone.

1. **Exact greedy parity, D (DFlash2) vs T (target only).** Same inputs. Report the first divergence position per request and the logit margin there.
2. **A/A repeatability.** T vs T (two runs) and D vs D. This gives the baseline divergence rate and margin distribution.
   - A D-vs-T divergence at a margin inside the A/A tie band is counted as a numerical tie, not corruption.
   - The #36548 probe (the same prompt at concurrency 12 vs serial) measures **state contamination**, not accuracy. Its pass rule: the c12-vs-serial divergence rate is no higher than the A/A rate plus a preset allowance.
3. **Paired known-answer scoring** on N **distinct** tasks: D and T solve the same tasks.
   - Report the paired difference with a 90 % CI (Newcombe, for paired proportions).
   - Pass only if the CI lower bound is above −δ, with δ fixed before the run (proposal: 5 points).
   - **Honest limit:** with N ≈ 20–30 tasks (all the GPU budget allows), the CI is about ±15–20 points. This part can only catch a gross regression, not prove equivalence. Equivalence evidence then has to come from part 1 (parity at A/A level).
4. **Failure taxonomy** for every discordant pair: format/parse error, wrong grid, truncation, timeout.
5. **Synthetic CPU test of the gate:** a clearly worse D arm must FAIL, and a matched arm must PASS.

## D. What this changes in the README priority list

| Item | Before | After |
|---|---|---|
| E14-0 | workers 16–32; size + header + head/tail check | Adaptive 4 → 8 → 16 (cap 24); `.partial` + atomic rename; per-range hash + destination re-read; safetensors no-hole check; write loop; cgroup memory; state record; off switch; 6 more failure-injection tests |
| E14-P0 | "the template is not the cause" | "Reconstructed divergence 0/170 on tool-call turns". Extend to all turn classes; add the same-text-different-ids test; the after-trim share must be counted. Exact ids: E14-A replay with `return_output_ids_in_sglext` |
| E14-A | Tier A with a large NCCL matrix; #44401 + graphs riding along | Baseline W1 unchanged → +#44401 (source-hash guarded) → +graph bs 1..13; NCCL 3 arms; replay with exact ids |
| E14-G | D ≤ 2 % and McNemar p ≥ 0.01 | Section C |
| Accounts | — | The user confirms the account ownership / team rule (B25) |
