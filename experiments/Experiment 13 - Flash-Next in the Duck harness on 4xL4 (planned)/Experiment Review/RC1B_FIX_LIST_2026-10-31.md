# E13 RC1b: fix list from the three RC1 pre-GPU audits (2026-10-31)

**Input.** Three GPT audits of `E13_RC1_AUDIT_FOR_GPT_2026-10-30.zip` (notebook bf28715d). Each said NO-GO until its HIGH items were fixed.

**Source checks before fixing** (SGLang 0.5.21, `layers/moe/fused_moe_triton/layer.py` and `token_dispatcher/standard.py`):
- `FusedMoE.forward` receives GLOBAL top-k ids.
- `StandardDispatcher.dispatch` maps the ids with `self.local_expert_mapping`, an int32 `[num_experts]` table: the global ids `[r·128, (r+1)·128)` map to local `0..127` and every other id to -1. The table is built on the first EP dispatch.
- `final_hidden_states` is all-reduced inside FusedMoE only when `reduce_results` is set.
- With a fused shared expert, the MoE can run `inplace`.
- **Two defects the audits did not name, both fixed:**
  - RC1 looked for the map on the module, but it lives on the dispatcher, so the GPU run would never have found it.
  - The post-hook fixture would have stored the overwritten input of an in-place MoE.

| # | Audit | Item | Fix (notebook a41c2562 → final e702c098) |
|---|---|---|---|
| 1 | A1 H, A2 H | CPU reference outside the arm deadline (own 900 s; ran inside the GPU arm, twice) | Fixtures are queued (`E13_RC1_PENDING`). The reference runs after ALL GPU arms, with deadline `now + min(15 min, HARD_STOP − now − 4 min)`, and is skipped below 90 s. The deadline is checked per scale tensor, per expert load and per fixture, and a cut-short result is marked `truncated`. `e13_loadprobe.json` is written after each arm and after each reference, so a truncated reference loses no GPU evidence |
| 2 | A1 H | Expert ownership inferred from the id range | Ownership comes only from the captured `dispatcher.local_expert_mapping`. Ids are global; ownership means `map[g] >= 0`. A fixture without a map under EP is not compared (`missing expert_map; EP mapping unverified`). The map round trip is recorded: the owned block must be contiguous and numbered 0.. in order (audit extra A). The comparison is the rank-local sum, or the global sum if `reduce_results` is set. A fused shared slot is computed from its bf16 weights. There is a verdict only when layer 1 validates the reference (rel err < 5 %) |
| 3 | A1 H, A3 H | PLE run continued after a failed `/flush_cache` | Fail closed: no prompt is sent, and the run gets `valid_comparison False` and `error: cache isolation failed` |
| 4 | A3 H | Three different prompts across PLE modes | One token-identical prompt for orig, sync and skip (cold cache each time) |
| 5 | A1 H, A2 H, A3 H | vLLM smoke false PASS | `logprobs_ok` (non-empty and all finite). `chat_ok`: an exact `4` after any `</think>`, asked with `enable_thinking False` and 64 tokens. `streams4_ok`: all 4 streams at exactly 256 tokens. `smoke_correct` = count + logprobs + chat; `throughput_valid` = `smoke_correct` + streams; `ok` = `throughput_valid` |
| 6 | A1 M, A3 H | vLLM arm ignored the host-RAM guard | Fail closed. Both arms also fail closed when the GPU-free check is not exactly True. The install timeout is clamped to the arm deadline, and there is no launch with < 12 min left |
| 7 | A1 M | vLLM gc hook patched quant-method instances | It now wraps the class once (`setattr(cls, ...)`); the layer comes from the argument, so a shared instance stays correct. The instance is never patched |
| 8 | A2 H | `_rc1_tensor` reshape copies a non-contiguous tensor in full | Rows are sliced with `narrow`, and only the ~4M-element chunk is reshaped |
| 9 | A3 M | Unbounded fp32 expert cache (~18.75 MiB per expert) | `OrderedDict` LRU of 24 experts |
| 10 | A1 extra B | Device-wide sync around the layer-2 MoE | `E13_SYNC_RE=layers\.2\.mlp\.experts$`: `torch.cuda.synchronize()` in the pre and post hooks. A failed sync is logged and never breaks a hook |
| 11 | (own) | In-place MoE | The fixture inputs and the input-finiteness check (`inputs_bad`) are taken in the PRE hook |
| 12 | (own) | Size | 400,768 bytes after the fixes, so the unused NANPROBE `e13_probe_smoke` is dropped from RC1 (every RC1 arm uses `ple_modes`). Final: 398,092 bytes |

**Not changed, and why:**
- **The fresh-server confirmation of a sync or skip success (A1):** a follow-up only if that happens.
- **A reasoning or tool-call probe after sync (G1):** the count prompt is the NaN gate.

**Local validation:**
- **`unit_rc1.py` 23/23** (17 old + 6 new). The new cases:
  - the non-contiguous 3D scale is scanned row by row;
  - the dispatcher map is in the scan and the fixture;
  - `reduce_results` is recorded;
  - in an in-place MoE, `inputs_bad` stays False and the fixture keeps the true input;
  - a CPU sync failure is logged, not raised.
- **`unit_vllm_gc.py` OK:** with one quant method shared by two layers, both get gc with their own names, and the instance is not patched.
- **Harness `rc1` 20/20** (13 old + 7 new):
  - the reference is truncated at a 1 s deadline;
  - global ids [3, 9, 10, 14] on EP rank 2 give experts 9 and 10 as local, with rel err < 1e-2;
  - a missing map is not compared;
  - vLLM smoke gives `ok` False for a NaN/Inf logprob, the answer 14, a 4 only inside the thinking, a timed-out stream, or a 255-token stream (good → `ok`);
  - the flush fails closed;
  - the vLLM host-RAM guard skips;
  - the same prompt is used in all modes.
- The verdict on the tiny real GPTQ checkpoint is MARLIN PATH FAULT, as designed.

**Notebook:** `Q27-XHIGH-E13_FLASHNEXT_RC1.ipynb`, sha **e702c098**, 398,092 bytes.
