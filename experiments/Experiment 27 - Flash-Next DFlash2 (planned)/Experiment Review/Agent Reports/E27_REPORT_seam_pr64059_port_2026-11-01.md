# E27 seam agent (aefbfc6004f4dc34f): PR #64059 port (vLLM #63587), verbatim

The vLLM #63587 fix (a port of PR #64059) is now a 7th file in dflash_seam.py; all 555 tests pass (was 522).

New dflash_seam.py:
- sha256 2df1da39495416b574a98317c9aef9da477354c50e42c18b2550c73807c9eed6, 1649 lines, LF, ASCII.
- NBDF must re-embed it.

New file in PATCHES (part of CORE_RELS, always applied): vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
- pre sha256 b731b09c92b93063d89851a997e1de70e51d512bc40b2a4dd68274a3a999e187 (Merlin ad42e37)
- post sha256 85f57f0ee9bc7018f052f59d0610a91c78d35cd018ef711c6f7ecbef298d47ea
- patch text sha256 34939f66174b3c6c0c7c391493376a7ff474af109b02e9b5ff64cb162bce7417

The other 6 files' pre / post shas are unchanged.

dflash_seam_tests.py: sha256 a46ea2db69fc25d2cfb140fff1b7738fb34db7c299af3c39b05ee6d681dfe64e, 1896 lines.

I read PR #64059 read-only with gh pr diff (open, head 7618884bc5d41d53fb862dca9aa3f389d226921b). Merlin uses the same names: self.attn_groups, self.block_tables.input_block_tables, group.kv_cache_group_id, AttentionGroup.map_to_kernel_block_table (v1/worker/utils.py:415), and supports_decode_metadata_update (set in speculator.py _configure_decode_metadata_update, called from set_attn). Nothing needed renaming.

Hunks (pre-patch lines):
1. @@-34,6, class DFlashSpeculator: class default _e27_kernel_map_refresh = False (this is the marker line).
2. @@-252,6, end of set_attn: calls self._e27_configure_ingraph_meta(), which runs before graph capture and does nothing unless DFLASH_SEAM=1. With DFLASH_SEAM=1:
   - _e27_kernel_map_refresh = the final in-graph flag;
   - DFLASH_NO_INGRAPH_META=1 (default off, the issue's workaround) sets supports_decode_metadata_update = False for this speculator only, so the metadata is rebuilt eagerly every step;
   - prints one E27_DFLASH ingraph_meta line per drafter worker (PP rank 1) with: native_supports_decode_metadata_update, supports_decode_metadata_update, forced_off, kernel_map_refresh, and per-group gid / manager_block / kernel_block.
3. @@-505,6, propose: the PR's branch, as an elif added after the skipped build:
   elif self._e27_kernel_map_refresh:
       for groups in self.attn_groups:
           for group in groups:
               group.map_to_kernel_block_table(self.block_tables.input_block_tables[group.kv_cache_group_id])
   It runs before run_fullgraph.

With every env unset, the flag stays at the class default False, so the elif never fires and set_attn does nothing extra.

Helper and readback changes:
- server_env takes a new no_ingraph_meta=True option.
- BEHAVIOUR CHANGE: readback_label now requires tp_size ingraph_meta lines, each with kernel_map_refresh == the final in-graph flag. Pass no_ingraph_meta=True for the fallback arm, which also requires forced_off True and the in-graph update off. A run where the in-graph update is on but the refresh is off is reported NOT_INTEGRATED.

New tests (T11, 17 checks):
- AST placement checks:
  - the class default exists;
  - set_attn calls super().set_attn first and the configure helper last;
  - the original has no else after the skipped build (the bug), and the patched code has exactly the elif, placed before run_fullgraph;
  - every other statement of propose is unchanged.
- The real _e27_configure_ingraph_meta:
  - env off, with or without DFLASH_NO_INGRAPH_META: nothing changes and nothing is logged;
  - DFLASH_SEAM=1: refresh on and the log line correct;
  - native backend without in-graph update: refresh off;
  - DFLASH_NO_INGRAPH_META=1: in-graph update forced off.
- The real elif body plus the real map_to_kernel_block_table, manager block 4096, kernel block 64:
  - negative control: without the refresh, the kernel table keeps the capture-time ids 1, 2 (stale KV);
  - with the refresh, manager id 3 maps to kernel ids 192..255 and id 5 to 320..383, and other rows follow the current table;
  - packed kernel stride 128: id 3 maps to 384..447;
  - a group without kernel blocks is left alone.
- T9 gained 7 readback / server_env checks.
- T1 / T10 now cover the 7th file (apply / verify / revert, shas, both orders with nb4_ppack).

Check counts by section: T1 76, T2 37, T3 90, T4 17, T5 209, T6 29, T7 26, T8 12, T9 20, T10 22, T11 17.

Merlin source copy still clean.
