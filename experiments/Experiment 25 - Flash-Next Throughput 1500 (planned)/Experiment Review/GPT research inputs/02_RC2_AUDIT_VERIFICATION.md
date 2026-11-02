# Verification of the GPT RC2 audit (02_RC2_AUDIT_GPT_2026-10-31.md) against the RC2 files

Checked by the lead on 2026-10-31 against `_scratch/e13_rc2_gpu_v1/rc2/logs/server_merlin_fp8.log` and `rc2_results.json`.

| GPT claim | Evidence found | Verdict |
|---|---|---|
| Steady table, KV 513,940 / 269,042, ready 411.1 / 999.1 s | rc2_results.json A1/A3 | VERIFIED |
| 90k@3 = INCONCLUSIVE_PREFILL_IN_WINDOW, not a decode number | census_long 90k@3 status | VERIFIED |
| tokens_per_step = (streams/2) × acceptance (PP2 half-batch accounting) | 6.03 ≈ 2 × 3.017; 12.62 ≈ 4 × 3.154 | VERIFIED (arithmetic) |
| Static scales "6 of 12" is per PP stage, so all 12 QSA layers are scaled | lines 143-218 (PP0 layers 0-25, PP1 25-48) | VERIFIED |
| TF raw per-position arrays not saved, only aggregates | rc2_results.json tf blocks | VERIFIED |
| Mamba page padded 21.01 % | lines 153, 165, 167, 180 (`interface.py:1121`) | VERIFIED |
| KV cache group sizes (4096, 4096, 4096, 8, 4096); side-cache layout on; PLE Mamba group folded; QSA ring packs 1024 requests per block | lines 241, 243 | VERIFIED |
| Cgroup memory preflight 0 MiB available during load | lines 154, 166, 168, 181 (cgroup 177,152 MiB limit reached) | VERIFIED |
| PP1_TP1 load 873 s vs ~125 s | startup lines; its load finished 06:06:38 vs 05:54 for the others | VERIFIED; cause = host memory pressure (strongest lead, not syscall-proven) |
| max_num_scheduled_tokens = 1024 warning | lines 19, 383 | VERIFIED |
| Target FULL_AND_PIECEWISE, draft decode FULL_DECODE_ONLY | lines 245-246, 340-341 | VERIFIED |
| RecoverSSM path taken, 8 spec rows, align | lines 321-327 | VERIFIED (graph replay of the commit not shown) |
| PYNCCL for tp/pp/ep groups | lines 34-38 | VERIFIED |
| JIT during inference (GDN, PLE, RecoverSSM) | lines 441, 551-555: `_fill_num_accepted_kernel`, `_causal_conv1d_update_kernel`, `_gdn_recoverssm_verify_kernel`, `_ple_conv_kernel`, `_ple_conv_writeback_kernel` | VERIFIED (5 kernels) |
| Leaked shared-memory objects at shutdown | line 1825: 2 `psm_*`; line 1823: 4 loky semaphores | VERIFIED |
| Inline system messages change the prefix ("HIGH for Duck") | line 389 | VERIFIED as a log line, but it applies to **`/v1/messages` requests** only (Anthropic-style endpoint). Duck relevance depends on the endpoint and on whether Duck sends non-leading system messages: check in NB5 (CORRECTED severity: conditional) |
| SSM state dtype not reported | no dtype line in the log | VERIFIED (must be printed from the worker) |
| 116k natural answer correctness UNVERIFIED | label not in the bundle | VERIFIED as a gap; the lead's RC2 analysis agent checks it against the local eval solutions |
