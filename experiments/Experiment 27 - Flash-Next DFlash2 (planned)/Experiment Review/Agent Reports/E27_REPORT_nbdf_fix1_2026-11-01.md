# E27 NBDF agent (acfe40bd525f7510b): kit fix 1 report, verbatim

## Both bugs fixed; the notebook is rebuilt and both kit folders are updated (nothing pushed)

Local units: **123/123 PASS**. The units kit, run once locally, gives `@@KIT units ALL OK@@`: nbdf units 121/121 and the seam tests all pass. The 2 export-layout checks need the notebook file, which the kit does not bundle.

The seam is unchanged:
- `dflash_seam.py` is `c4ea1ecc287036bcadbc98cd33e40309d17f2c58b471e5c876ea7acddba51f78`;
- `dflash_seam_tests.py` is `452bb231…`, the same file the units kit bundles.

So no seam re-embed was needed.

### Bug 1: R4 RecursionError (fixed)
**Cause:** my builder wrote `nbdf_code_export.py` as the CELL 7 stub (which execs the compressed body) plus the body inline, and the real kit runs that export. So the body ran twice. On the second pass, `_merlin_fn_server_nb1 = merlin_fn_server` captured NBDF's own wrapper, and `merlin_fn_server` called itself. That is why `host_phase` was null and the R5 fields were empty for all 5 arms. `L.Stream.run` was double-wrapped the same way.

**Fixes:**
- `build_e27_nbdf.py:1523-1524`: the export is now exactly the notebook's code cells, so the body runs once, through the stub. The readable body is written to the new file `nbdf_cell7_body.py`.
- `nbdf_lib.py:134` `wrap_once(holder, name, factory)`: wraps only once; a repeated execution returns the existing wrapper.
- `nbdf_cell7_body.py:61` / `:74`: `_mk_nbdf_run` plus `N.wrap_once(L.Stream, "run", …)`.
- `nbdf_cell7_body.py:309` / `:338`: `_mk_merlin_fn_server` plus `merlin_fn_server = N.wrap_once(globals(), …)`. The launch check moved into `_nbdf_launch_check`.

**New unit tests (T11, 7 checks):**
- `wrap_once` is idempotent.
- The real body's `_mk_merlin_fn_server`, executed **3 times**, is then called for **all 5 arms** (Q, D4, D3, D7, D4b): no recursion, and each call reaches the original exactly once.
- `Stream.run` is wrapped once, and no unguarded rebinding is left in the body.
- The export equals the notebook's code cells, and the body is not inline in it.

**Not covered locally:** a test that runs Merlin's real `host_phase()` needs the installed Merlin tree, which only exists on Kaggle. Real-kit R4 is that test; it runs the notebook's own `merlin_fn_server` for all 5 arms.

### Bug 2: "Failed to infer device type" in R9 / R10 / R12 (fixed with the proven klee100 method)
All of this is in `build_e27_nbdf_cpu_kit.py`:
- **`:208` `STUB`:** a symlink to the libcuda.so stub from the CUDA toolkit, put on `LD_LIBRARY_PATH`, as in `k100kit_main.py:409-427` (`vllm._C_stable_libtorch` needs it to import).
- **`:233` `PLATFORM`:** a preamble copied from `k100kit_real.py:74-148`. Before any vllm config or model import it sets `vllm.platforms._current_platform` to a `NonNvmlCudaPlatform` subclass reporting sm_89 "NVIDIA L4", 23,034 MiB, and not fully connected. It also provides `init_gloo()`: a single-process gloo group (TP1 / PP1) inside `set_current_vllm_config(VllmConfig())`.
- **`:285` `venv_py(..., platform=True)`:** prepends the preamble and sets `CUDA_VISIBLE_DEVICES=""` and `LD_LIBRARY_PATH`. R9, R10, R11 and R12 use it (`:568-573`).
- **R12 `WEIGHTPROBE` (`:374`):** now runs the real seam check:
  1. sets `DFLASH_SEAM=1` on the patched tree, then calls `init_gloo`;
  2. builds a real `ModelConfig` of the drafter dir;
  3. builds `DFlash2Qwen3ForCausalLM` on meta;
  4. calls `m.load_weights` with meta tensors of the real checkpoint's shapes and dtypes.
  - Pass means the seam's `draft_load_check` line is printed and nothing raises.
  - A "got no tensor" raise is recorded as `false_raise_suspect`; that is the evidence for whether `DFLASH_LOAD_CHECK=0` is needed.
- **R9 `KVPROBE` (`:415`):**
  - tries a real `ModelConfig` of the merged W4 checkpoint first (`G["CKPT"]`), with the old stub as fallback;
  - calibrates Q at Q's own pin 3,553,580,524 (must give 213-217 blocks; NB4T measured 215);
  - runs the drafter group at that pin, and at the D pin 3,016,709,612 with 4096-token and 1024-token drafter blocks;
  - every failure now carries a traceback.
- **R10:** the argparse part now runs on the patched platform and records the platform state.

### New shas
| File | sha256 / size |
|---|---|
| `../Q27-XHIGH-E27_NBDF_DFLASH2.ipynb` | `2452381ce886b660e15b6c2c1439d4844f4e568fd7fb566648355c30eb122f57`, 387,396 B (under 400 kB); gpu_ready false, as before |
| `build_e27_nbdf.py` | `7309af3ff8037ec90a569285c0e59f83e31c2e4514065cb6794949c1cf5f88e8` |
| `nbdf_lib.py` | `8c44e8c64238d502a19869377eaaebbe36186a1b5c39cf89a8e71e1533519416` |
| `nbdf_cell7_body.py` (new; = the embedded body) | `39ccba440f214a051a1038bfd21cc0c17649c8ad5cce6fc79afb5eb13bbe87fd` |
| `nbdf_code_export.py` | `b79fc61457ee03f5bc95dedd577bbcdde8e7cef3fcbcd5530378c14dba38bae3` |
| `nbdf_unit_tests.py` | `74421c8e8e9691a914c7ab5e0c6efa3e982296d0ad5261fe1cdf4c6178df61ce` |
| `build_e27_nbdf_cpu_kit.py` | `a500fe09d68a0605be06218091b0f8626eda09cd9c5fda1e20833329e35bcc89` |

The 9 NB4T / E25 files the builder checks are unchanged. All files are LF with 0 CR.

**Kit folders** (updated, not pushed), under `D:/ARC AGI 2/_scratch/push_e25/`:
| Folder | Size | sha256 |
|---|---|---|
| `q27-e27-nbdf-kit-units__foysalemonshanto` | 311,974 B | `6d2c3d46…` (now also bundles the body file) |
| `q27-e27-nbdf-kit-real__foysalemonshanto` | 253,188 B | `2e8434c1…` |
| `q27-xhigh-e27-nbdf__foysalemonshanto` (GPU) | — | updated with the new notebook |

The audit zip `Experiment Review/Audit Zips/E27_NBDF_audit_FLAT_2026-11-01.zip` is refreshed with the fixed files and the body file: 684,068 B, sha `c3871f5c…`.

### Still open, depending on the next real-kit run
- **R9:** I could not run the KV probe locally (no vllm here). With the platform fixed it now gets past import; whether `get_kv_cache_configs` reproduces Q's 215 blocks will only show in the real kit. The tracebacks are there if it does not.
- **R12:** may still hit a CPU-only limit inside layer construction. If so, the error is recorded.
- **After a pass:** rebuild with `--kv-gate-report <nbdf_kit_real_report.json>` so the D arms run on the GPU.
