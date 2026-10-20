# FN-PLEFP8-UPLOAD + FN-SMOKE v3: response to review round 1, with a true / false ledger

I checked every claim against the code, the installed Pennyroyal source, or live Kaggle state before changing
anything. Both validators run from this extracted bundle with no external paths:

```
python build/validate_upload.py --network
python build/validate_v3.py
```

Their outputs are in `build/validate_upload_output.txt` and `build/validate_v3_output.txt`.

## Ledger

| # | reviewer claim | verdict | evidence | action |
|---|---|---|---|---|
| 1 | P0: `"Other (specified in description)"` is not a valid Kaggle **model** license | **INCORRECT** | Live Kaggle API, 2026-10-05: `foysalemonshanto/qwen3-8-flash-next-nvfp4-nvidia/pytorch/nvidia-modelopt-mixed` has `license_name = "Other (specified in description)"`. It was created by the same reference uploader, processed (25 files listed with sizes), and mounted and served in a GPU run on 2026-09-27. | Kept. The label reflects the Qwen Community License honestly; the LICENSE file is uploaded and the description names it. |
| 2 | P0: the final verification cell only prints and never raises; the manifest blob is not verified | **CONFIRMED** | Reference cell 5 printed `missing` / `wrong` and ended normally | Cell 5 now raises `RuntimeError` on any missing file or wrong size, including `KAGGLE_MIRROR_MANIFEST.json`: exact size when this run uploaded it, presence on an idempotent rerun. It prints "VERIFIED" only on success. Eight fake-listing cases are tested (`validate_upload.py` §4). |
| 3 | P0: no RLIMIT_MEMLOCK gate; a 47.7 GiB pinned allocation fails under a small `ulimit -l` | **NOT A BLOCKER for this code path** | Pennyroyal's `_allocate_pinned_table` (`qwen4_exp.py` ~771-797) uses `mmap(MAP_PRIVATE\|MAP_ANONYMOUS)` + `cudaHostRegister`, with no `mlock` or `MAP_LOCKED` anywhere in that path. dfranzen's Kaggle notebook pins the 102 GB BF16 PLE through the same code and never touches ulimit (0 occurrences of ulimit, memlock or mlock). The cited reports concern other runtimes. | Cheap mitigation added anyway: stage 0 records the soft and hard limits and raises soft to hard (the server inherits it). It is informational, not a gate. Tested with a fake `resource` module. |
| 4 | P1: Intel small metadata blind-merged (e.g. `quantization_config.json`) | **CONFIRMED (partly)** | v3-R0 linked every non-safetensors Intel file except a skip list | Better fix than a whitelist: the overlay now carries albucino's small runtime files (tokenizer.json, tokenizer_config.json, chat_template.jinja, generation_config.json, preprocessor_config.json, video_preprocessor_config.json; all byte-identical to Intel's). The merged folder takes **every non-weight file from the overlay**, all SHA256-checked. From Intel only the 15 main shards are taken. `quantization_config.json`, `config.json.autoround.bak`, `processor_config.json`, BF16 PLE and MTP are never linked (tested). The layout equals albucino's own repository. |
| 5 | P1: size + header hash does not catch corruption after the header | **CONFIRMED** | — | Full SHA256 of all 25 shards in stage 1: 8 parallel threads, clamped to the deadline. It also warms the page cache for the load. Tested: a byte flipped after the header fails with the full hash and passes without it. The budget went from 45 to 50 min. |
| 6 | P1: 70 GB RAM gate is tight; re-check before launch | **AGREED** | — | Gate is 75 GB (env `FN_MIN_RAM_GB` overrides). MemAvailable is re-checked right before the first launch; if it is below the gate, `4_launch` fails without starting the server (tested). |
| 7 | P2: validator not portable (hard-coded `D:\` paths) | **CONFIRMED** | — | Bundle layout `notebooks/ reference/ build/ tests/`. All scripts use `__file__`-relative paths, and a fake competition folder is included. Verified by running both validators from a fresh extraction of the zip. |
| 8 | P2: "proven uploader unchanged" not reproducible from the bundle | **CONFIRMED** | — | `reference/NVIDIA_FlashNext_NVFP4_01_upload_on_kaggle.ipynb` is included. `validate_upload.py` rebuilds the notebook from it (identical sha256), asserts that the stream/resume/blob/worker/token-hand-off code is identical, and prints per-cell diff counts. |
| 9 | P2: stage 3 ignores the `--help` exit code; exception stage named `3_torch_cuda` | **CONFIRMED** | — | Success now requires `h.returncode == 0`; the exception path records `3_preflight`. Tested with a fake subprocess: help exit 1, probe failure, missing flag and exception all fail; the good case passes. |
| 10 | P2: stale documentation (110 GB, old launch ladder, 4×95 GB) | **CONFIRMED** | — | Header rewritten for v3. A scan of the notebook finds 0 occurrences of `110`, `95.4`, `4 x 95`, `181 GB`, `3_torch_cuda`, `ulimit -l`. |

## Not changed, by design
- Launch profile (TP4+EP4, Triton GDN, BF16 KV, 8K context, 1 request, language-model-only, no CUDA graphs). This is the
  first boot, and a single variable at a time is the plan.
- Instance metadata keeps `licenseName` (see #1).
