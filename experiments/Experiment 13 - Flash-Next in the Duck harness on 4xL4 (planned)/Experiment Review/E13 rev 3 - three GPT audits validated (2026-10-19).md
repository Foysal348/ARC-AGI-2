# E13 PLAN rev 3: the three GPT audits, validated (2026-10-19)

**Inputs:**
- GPT-A "Rev 3 Deep Validation": conditional GO.
- GPT-B "Deep Plan Audit": conditional GO; the pasted text was truncated after Stage 2.
- GPT-C "Deep Technical Audit": conditional NO-GO.
- A web fact-check of every cited URL (GitHub API + gh + WebFetch; raw dump in the session scratchpad `gh/dump.txt`).
- Local source checks of the 0.5.21 wheel and the Pennyroyal wheel.

**Verdicts:** ✅ accepted (verified), ✏️ accepted with a change, ❌ rejected, ➖ not applicable to us.

## Findings that change the plan

| # | Finding (who) | Check | Verdict |
|---|---|---|---|
| F1 | Classic FA2 works for QSA decode on sm89 (A, B, C) | **PR #36968:** on real L20 (sm89), FA4 4.0.0b34 fails all 8 cases at head dim 256, and **FA2 matches an FP32 reference 16/16**. aisensiy (2026-10-13): 8x RTX 4090, Flash-Next FP8, TP4/EP4, 262k ctx, PLE offload; FA2 `2.8.3.post1` built against torch 2.13+cu130, "server serves fine". The PR is OPEN (fail-early check only, no new kernel). | ✅ **The decode path changes: FA2 primary** (upstream sm89 evidence; zero SGLang source change). P1 (our routing to the Triton packed decode, validated only on ROCm) becomes the measured fallback. |
| F2 | An FA2 wheel exists for torch 2.13 / cu13 / cp312 (A, C) | fogandfrog release: `flash_attn-2.8.3.post1+cu13torch2.13cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`, 232 MB, SHA256 `74984f62…ae84`, "built from the unmodified upstream sdist", archs 80/90/100/120 (sm_80 cubins run on sm_89, same major version: standard CUDA binary compatibility) | ✅ add it to an offline dataset after the user's approval (third-party binary: pin the SHA, check ABI + import + parity on L4) |
| F3 | Franzen patch #3 cannot run verbatim; it asserts the Pennyroyal git pin (A, B, C) | His script asserts `git rev-parse HEAD == d00d88ef…`. **But the 0.5.21 `gptq_moe.py` sha256 = `135e004f…` = his BEFORE_SHA exactly** (local sha256sum; GPT-B found the same). So his two substitutions **and his own BEFORE/AFTER SHA asserts** apply unchanged; only the git-pin line is dropped. | ✏️ D2 wording: "his patch #3 with his SHAs, minus the git pin". GPT-C's placeholder SHA is unnecessary. |
| F4 | `--ple-offload-backend file` is not a safe fallback on L4 (A) | 0.5.21 `qwen4_exp_ple_table.py:321-345`: it needs `cudaDevAttrPageableMemoryAccessUsesHostPageTables` ("unified-memory parts such as GB10") and raises otherwise | ✅ **my rev-3 statement was wrong.** If pinned fails: stop. The FP8-PLE overlay is the only fallback, and it needs user approval. |
| F5 | Pinned PLE: 1 GiB probe ≠ 95-128 GB success (A, B) | 0.5.21 pinned backend = `torch.empty(..., pin_memory=True)` (`qwen4_exp_ple_table.py:225-245`); per rank about 23.8 GB BF16 (VocabParallel), possibly rounded up by the host allocator | ✅ add a pre-load pinned allocation test of the real per-rank size, 4 processes concurrently |
| F6 | JIT `fast_topk<512>` silently drops candidates on concentrated scores (B; A mentions #36807) | **#36807 confirmed:** on a 4090, normal scores 0/20 bad, concentrated scores 20/20 bad; every fix PR is open; the kernel is unchanged in 0.5.21/0.5.22. **Our config uses exactly this path:** indexer budget 2048 / ratio 4 = 512 → `qsa_indexer.py:498-507` (decode) and `kernel.py:23-41` (prefill). **Pennyroyal uses the same JIT fast_topk family with the same `pos < SMEM_INPUT_SIZE` drop**, and Franzen scored 36.6/46.5 with it. | ✏️ keep it verbatim (as Franzen); the micro-preflight measures drop rates vs `torch.topk` on realistic and concentrated logits. A patch only if drops are frequent on realistic logits, and only with approval. |
| F7 | The B2 rung changes two variables (TileLang + NEXTN) and "re-enable NEXTN" is a third launch (A, B, C) | true (PLAN rev 3 §3) | ✅ B2 = exactly one variable, chosen by the failure signature; no re-enable within the budget |
| F8 | Speed of 700 tok/s must not be the only gate (A, B, C) | reasonable | ✏️ two gates: serving (correct, stable, measured concurrency) and competition (EVAL30 score x coverage vs E12 / R5). 700 is a target only. |
| F9 | NEXTN + CUDA graph corruption reports (A, B) | **#37111** (GB10): graph mode gives punctuation loops, eager OK, open. **#37052** (GB10): invalid probabilities, also eager per a commenter, open. Not Ada. | ✅ Gate C: greedy repeat-determinism, invalid-probability check, acceptance logging. NEXTN-off is a B2 option. |
| F10 | Torch indexer fallback (TileLang off) uses large temporary memory (C) | 0.5.21 `qsa/mqa.py:79-108` gathers the full compressed keys in FP32 | ✅ TileLang-off is a diagnostic only; the micro-preflight also measures its memory |
| F11 | Patches #2/#4 absent → memory and cache semantics differ; D7 semantics unverified (B, C) | true | ✅ recorded as known differences; measure intermediate SSM allocation, max_total_num_tokens and cache % |
| F12 | Zero-valid rows in the Triton kernel → 0/0 (B) | 0.5.21 `sparse_attn.py:167-262`: with no valid key, `normalizer = 0` → NaN. Decode rows always contain the current token, so this happens only in CUDA-graph padding rows, whose output is discarded. | ✏️ low risk; test it in the micro-preflight (P1 fallback only) |
| F13 | D3 predicate: resolve the capability once at init (A) | only relevant for P1, now a fallback | ✏️ if P1 is used: compute it once in `__init__` |

## Findings checked and not changing the plan

| # | Finding | Check | Verdict |
|---|---|---|---|
| N1 | Marlin "Invalid thread config" at N=320 (#37089, A) | confirmed for **TP4-sharded** w13 (640/4 x 2 = 320). EP4 keeps whole experts (w13 N=1280, w2 K=640, multiples of 128). Nobody has tested EP4; our inference. The N64 fix #37946 was closed unmerged. | ✅ keep EP4; log the effective MoE shapes at load |
| N2 | `desc_act=True` w2_scales bug (#36822, A) | the Intel AutoRound config has no `desc_act` (sym, g128, `auto_round:auto_gptq`) | ➖ not applicable; still assert `desc_act` false at preflight |
| N3 | #36537 token-0 "!" loop | closed: "the token-0 loop came from the SM120 FA4 varlen fallback in QSA sparse decode"; it appears without tools or speculation | ✅ supports F1: a broken QSA decode alone can produce "!!!!". E3 used our PyTorch shim there (maths re-checked), so E3's cause is still open; F1 replaces that path anyway. |
| N4 | #37633 illegal memory access | fixed upstream (#38346 / #39446); `group_locs.clamp_max` is present in 0.5.21 (web check) | ➖ |
| N5 | #36545 QSA with FP8 KV | fix PR open; 0.5.21 casts FP8 → Q dtype in the chunk kernel (`sparse_attn.py:219-224`) and allocates the decode scratch in `q.dtype`; Franzen ran fp8 KV | ✏️ keep fp8 KV (verbatim); the micro-preflight covers fp8 storage |
| N6 | TileLang "data race" warning | #39393 (4090) saw it in both correct and corrupt runs | ✅ not proof either way; the micro-preflight compares TileLang vs torch on the same inputs |
| N7 | Licence | HF: qwen-community-1.0. ARC Prize: third-party components under "an open source license which allows public sharing". Kaggle rule 5.a.3 exempts pretrained models "with an incompatible license" from re-licensing. **Kaggle staff (2026-10-16, topic 745079): "we cannot offer any guidance or provide interpretation on model licenses".** | ✅ unresolved; LB use is the user's risk decision |
| N8 | GPT-C claim 23 "unverifiable" | `topteam_comments.md` / `lb.json` were not in the zip | ➖ zip gap only; the evidence exists locally |

## Rejected

None of the substantive findings was wrong. GPT-C's "placeholder SHA" (F3) is superseded by the exact SHA match.
