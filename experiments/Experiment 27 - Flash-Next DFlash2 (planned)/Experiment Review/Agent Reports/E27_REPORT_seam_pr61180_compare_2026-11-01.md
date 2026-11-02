# E27 seam agent (aefbfc6004f4dc34f): PR #61180 comparison, verbatim (SendMessage, 2026-11-01)

PR #61180: same relay semantics as our nvidia/model.py hunks. dflash_seam.py is unchanged; the tests now pass 522/522 (was 520).

I read the PR read-only via gh pr diff (still open, head d6ade47102988d46b463498d01769aa2d3cb8b42). Diff saved at D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/pr61180/pr.diff

Comparison, point by point:
1. Pack / collect: same placement.
   - Upstream taps are collected at the start of forward.
   - Non-last ranks add **pack_local_aux_hidden_states(aux) to the IntermediateTensors.
   - The last rank returns remote_aux + aux.
   - Only difference: the PR calls collect_remote_aux_hidden_states always; we call it only when taps are set, so the no-tap path runs the original statements. Same result, because collect returns [] when the upstream total is 0.
2. Setter through EagleModelMixin._set_aux_hidden_state_layers: same.
   - The PR replaces Qwen3.5's direct assignment with that call.
   - Ours goes: protocol default setter -> our override -> EagleModelMixin's. The result is the same: sorted unique ids, slot base and upstream total.
   - Ours adds: refusal unless DFLASH_SEAM=1, a range check on the ids, the local-id record and the aux_layers log line.
3. Missing upstream slot: same RuntimeError, "Missing aux_hidden_states_1 from PP intermediate tensors". Ours also raises DFLASH_AUX_COUNT_MISMATCH and DFLASH_AUX_WIDTH_MISMATCH, so it is stricter.
4. PP1 / no-aux: same.
   - No taps: the plain hidden tensor, and non-last ranks send only their usual keys.
   - Taps: (hidden, taps).
5. Global order: same. Upstream taps first, then local ones; the id-0 entry tap is taken on the first rank only.

Intended, model-specific differences:
- Tap value: Qwen3Next taps hidden + residual. Ours taps the combined state averaged over the 4 hyper-connection streams, which is what the incoai drafter was trained on.
- HC-SP: ours gathers the taps to the full token axis; the PR does not.

PR test cases:
- Already covered by T3:
  - PP2 global order, including boundary and end ids (T3c / T3e);
  - PP1 with and without taps (T3a / T3b);
  - PP2 without aux keeps the transport keys (T3a).
- Ported, +2 checks: test_qwen3_5_declares_and_configures_aux_hidden_state_pp. The outer's setter is given the unsorted, duplicated ids (46, 6, 26, 16, 36, 6) on stage 1. It yields (6, 16, 26, 36, 46), slot base 2, upstream total 2 and local ids (26, 36, 46). This runs for both the CausalLM-shaped and the ConditionalGeneration-shaped outer.
- Tightened, no new check: test_qwen3_5_missing_upstream_aux_slot_is_not_silently_dropped. The check now requires a RuntimeError containing exactly "Missing aux_hidden_states_1".

Shas:
- dflash_seam.py: c4ea1ecc287036bcadbc98cd33e40309d17f2c58b471e5c876ea7acddba51f78 (unchanged; every file pre / post sha in my earlier report still holds).
- dflash_seam_tests.py: 452bb23163153aa1ffbc6fdda8ae571301919b4e805dfe1a01921eb34dc195be (1745 lines, LF only).

Extra: a repo grep confirms ple_layer_ids=[2] in the Flash-Next configs. None of the taps 6/16/26/36/46 sits next to a PLE layer. DFLASH_TAP_MODE=norm_input would refuse only tap id 2.

---
Coordinator note: the post shas in the earlier seam report table for `nvidia/model.py` (fee65643…) and `dflash/utils.py` (5b3cd014…) do not match the module constants. The module constants verified on Kaggle are `787161118bb4…` and `fb76c074fdc5…` (real kit: seam apply / verify / revert OK). So the earlier report table was stale, and the module is what runs.
