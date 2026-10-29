# E23 P1: Merlin Qwen3.8-27B W4A16 checkpoint as a Kaggle kernel output

Built 2026-10-29. Nothing has been pushed or run on Kaggle. The coordinator audits and pushes.

## What it is

`build_p1_kernel.py` writes `kernel/q27-e23-p1-merlin-ckpt.ipynb` and `kernel/kernel-metadata.json`. The kernel is a
CPU kernel on the foysal account (`foysalemonshanto/q27-e23-p1-merlin-ckpt`): internet ON, GPU off, private,
`machine_shape "None"`, and the same pinned image as the E22 flag kit
(`gcr.io/kaggle-private-byod/python@sha256:37c64f7d...d461`). It has no data sources.

It downloads:

- `halt95/Qwen3.8-27B-W4A16-Merlin` at revision `7a0af39f63dd4fb0733c23d3ec126762074a503f`: 15 files,
  17,049,970,203 bytes (15.88 GiB). They go to `/kaggle/working/Qwen3.8-27B-W4A16-Merlin/`.
- the merlin-engine source at tag `merlin-v3.0.0` (commit `ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5`), as the single
  file `/kaggle/working/merlin-engine-merlin-v3.0.0.tar.gz` (16.2 MB). P2 needs it.

Every file is checked by size and sha256 against hashes embedded at build time. A wrong file is deleted, never left in
place. The kernel writes `p1_report.json` in every case. It writes `q27_e23_p1_manifest.json` only when everything
verified. It prints `ALL OK` / `NOT OK`, and when NOT OK it raises `RuntimeError` so that the version shows as failed.

Rebuild with `python build_p1_kernel.py`. The builder asserts that the embedded Merlin `checkpoint.sha256` agrees with
the embedded list and that the sizes add up to `release/download-sizes.json`, then runs `compile()` on the cell. It also
asserts that the notebook is under 400 kB (it is 25.8 kB).

## Output layout (what the GPU notebook sees)

```
<mount>/q27_e23_p1_manifest.json          sentinel: repo, revision, model_subdir, {file: [size, sha256]}, merlin_src
<mount>/p1_report.json                    full report (per-file size/sha256/ok/method/seconds/MBps, disk, env)
<mount>/merlin-engine-merlin-v3.0.0.tar.gz
<mount>/Qwen3.8-27B-W4A16-Merlin/{model.safetensors, model-mtp.safetensors, kv_scales.safetensors,
    model.safetensors.index.json, config.json, generation_config.json, chat_template.jinja, tokenizer.json,
    tokenizer_config.json, preprocessor_config.json, video_preprocessor_config.json, LICENSE, NOTICE, README.md,
    .gitattributes}
```

That is 18 files and about 17.07 GB in all.

## Attaching it in the GPU notebook

kernel-metadata.json of the GPU notebook: `"kernel_sources": ["foysalemonshanto/q27-e23-p1-merlin-ckpt"]`. This
attaches the latest successful version's output. The kernel is private, so the GPU notebook must run on the foysal
account, or the P1 kernel must be shared with (or made public for) the account that runs it. The weights are
Apache-2.0, so making it public is allowed.

The mount path is not documented. The expected path is `/kaggle/input/notebooks/foysalemonshanto/q27-e23-p1-merlin-ckpt/`,
which follows the pattern our datasets mount under (`/kaggle/input/datasets/<owner>/<slug>/`, from our own logs). Older
and flattened layouts (`/kaggle/input/<slug>/`) also exist, so search for the sentinel instead of hard-coding the path:

```python
import json
from pathlib import Path
P1_OWNER, P1_SLUG, P1_SENTINEL = "foysalemonshanto", "q27-e23-p1-merlin-ckpt", "q27_e23_p1_manifest.json"

def find_p1(root="/kaggle/input"):
    root = Path(root)
    cands = [root / "notebooks" / P1_OWNER / P1_SLUG, root / P1_OWNER / P1_SLUG, root / P1_SLUG]
    hits = [c / P1_SENTINEL for c in cands if (c / P1_SENTINEL).is_file()]
    for pat in ("*/", "*/*/", "*/*/*/", "*/*/*/*/"):      # bounded depth: no full rglob over big wheelhouses
        if hits:
            break
        hits = sorted(root.glob(pat + P1_SENTINEL))
    if not hits:
        raise FileNotFoundError(f"attach kernel_sources {P1_OWNER}/{P1_SLUG} (no {P1_SENTINEL} under {root})")
    man = json.loads(hits[0].read_text())
    md = hits[0].parent / man["model_subdir"]
    for name, (size, _sha) in man["files"].items():  # cheap stat check here; sha256 after staging (below)
        if (md / name).stat().st_size != size:
            raise RuntimeError(f"P1 file {name} has the wrong size")
    return hits[0].parent, md, man      # base dir, model dir, manifest
```

The manifest is written only on ALL OK, so this finder cannot pick up a half-downloaded or failed version.

After staging the weights to `/dev/shm` with parallel readers, hash the staged copy against `man["files"]`. sha256 runs at about 0.5-2 GB/s per core,
so the 16.8 GB file takes roughly 10-35 s; it can run in a thread while the engine starts. Then serve from the staged
dir. Log the filesystem type of the mount at startup (`/proc/mounts` entry or `stat -f -c %T <mount>`) so a slow
GCSFUSE mount shows in the log.

## Research findings (with evidence)

### 1. Kaggle output size, file count, attach, mount, read speed

- **Output cap: 20 GB in `/kaggle/working`.** Kaggle's docs say "up to 20 GBs of output from a Notebook may be saved to
  disk in /kaggle/working" (https://www.kaggle.com/docs/notebooks; the page needs JavaScript, so the quote comes through
  search). A Kaggle staff comment in https://www.kaggle.com/discussions/product-feedback/372506 says /kaggle/working is
  limited to 20 GiB because it is persisted with Save Version. There is about 60 GiB of unpersisted scratch outside it
  (same thread; also https://www.kaggle.com/general/177616). Older posts say 5 GB; they predate the current limit.
- **Per-file size:** no per-file limit is documented anywhere I could find. Our single largest file is 16,816,202,144 B
  (15.66 GiB) and the total is about 17.07 GB (15.89 GiB), under both a 20 GB and a 20 GiB reading. The kernel refuses
  to finish ALL OK if the output reaches 19.5e9 B.
- **File count:** there is a reported 500-item listing limit for outputs
  (https://www.kaggle.com/product-feedback/420538). We write 18 files, and the merlin source stays one tarball (5,141
  members) instead of an extracted tree.
- **Attach:** `kernel_sources` takes `"username/kernel-slug"`
  (https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels_metadata.md). In the editor this is the "Notebook Output
  Files" input (https://www.kaggle.com/docs/notebooks).
- **Mount path:** not documented. kagglehub takes the path from the backend's `mountSlug` and prefixes it with
  `/kaggle/input` (`NotebookOutputKaggleCacheResolver` in kagglehub `kaggle_cache_resolver.py`,
  https://github.com/Kaggle/kagglehub). A Kaggle Q&A answer says dataset paths may drop the username, while models and
  notebooks follow type-specific paths (https://www.kaggle.com/questions-and-answers/684771). Our own logs show
  `/kaggle/input/datasets/<owner>/<slug>/` and `/kaggle/input/models/<owner>/...`. Hence the sentinel finder above.
- **Read speed (discussion 751344,
  https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/751344):**
  - Kaggle staff (Dustin): "I strongly suggest putting Models in Kaggle Models instead of datasets, this helps me know
    what kind of reads to expect on those files so that we can place them on the right type of cache".
  - One user re-uploaded to Kaggle Models, and the save run showed the Model "mounted over NFS".
  - Others logged datasets on `FUSE.GCSFUSE`: 8-13x slower reads, and one model taking more than 7 h to load.
  - Nothing in the thread says how kernel outputs are mounted (NFS or GCSFUSE). This is the main open risk of the
    kernel-output path, and it is why the GPU notebook should stage to /dev/shm with parallel readers and log the
    filesystem type.
- **Kaggle Model version from inside a batch kernel without secrets: not possible.**
  - `kagglehub.model_upload` builds its client from `get_kaggle_credentials()`. That reads only `KAGGLE_API_TOKEN`,
    `KAGGLE_USERNAME`+`KAGGLE_KEY`, the `~/.kaggle` files, or a Colab secret (kagglehub 1.0.3 `config.py` / `models.py`).
  - The in-notebook JWT (`KAGGLE_USER_SECRETS_TOKEN` + `KAGGLE_DATA_PROXY_TOKEN`) is used only for
    `AttachDatasourceUsingJwtRequest`, which attaches inputs (`clients.py`, `kaggle_cache_resolver.py`).
  - Secrets work only in editor runs, so a committed run would upload unauthenticated and fail.
  - The ways to a Kaggle Model are therefore (a) an interactive editor session of a kernel that attaches this output,
    with a Kaggle token secret, calling `kagglehub.model_upload(...)`, or (b) the Kaggle web UI, if it offers notebook
    output as a source; I could not verify that it does for Models (it does for Datasets).
  - For P1, kernel output is the path, as planned.

### 2. Checkpoint identity and hashes

- **Model page:** https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin. The HF API reports `gated: false`,
  `private: false`, license apache-2.0, base `Qwen/Qwen3.8-27B`, library vllm
  (https://huggingface.co/api/models/halt95/Qwen3.8-27B-W4A16-Merlin). No token is needed. The kernel removes
  `HF_TOKEN` from the environment and never prints one.
- **Pinned revision `7a0af39f...`:** commit "Card: serving repository tag v1.3.1", 2026-10-13
  (https://huggingface.co/api/models/halt95/Qwen3.8-27B-W4A16-Merlin/commits/main). `main` is now `5ffeb29a...`
  (2026-10-22). Its weights, configs and tokenizer are byte-identical to the pinned revision; only `README.md`,
  `.gitattributes` and `assets/*.png` differ (tree diff of both revisions).
- **Merlin pins the same revision:** merlin-engine `release/download-sizes.json` at `merlin-v3.0.0` pins exactly this
  revision for `27b`, with 15 files and 17,049,970,203 bytes. The embedded list sums to exactly that.
- **checkpoint.sha256:** fetched from
  https://raw.githubusercontent.com/halt95/merlin-engine/merlin-v3.0.0/models/qwen38-27b/checkpoint.sha256 (11 files).
  It is identical to the copy inside the tag tarball.
  - All 4 LFS files (model, mtp, kv_scales, tokenizer.json) match the HF LFS sha256 oids at the revision
    (https://huggingface.co/api/models/halt95/Qwen3.8-27B-W4A16-Merlin/tree/7a0af39f63dd4fb0733c23d3ec126762074a503f).
  - The 9 small files match the sha256 of my local downloads from the resolve URL at the revision.
  - The 4 files the list does not cover (`.gitattributes`, `LICENSE`, `NOTICE`, `README.md`) are embedded with the
    hashes of those local downloads.
- **Xet storage:** the LFS files are Xet-backed (`xetHash` in the tree). `/resolve/` answers 302 to
  `us.aws.cdn.hf.co/xet-bridge-us/...`, with `X-Linked-Size 16816202144` and `X-Linked-ETag` equal to the sha256. A
  Range request returns 206 (checked with curl).
- **Structure:** the index `weight_map` has 2,460 tensors: 2,387 in model.safetensors, 39 in model-mtp and 34 in
  kv_scales (index metadata `total_size` 17,036,572,664). The kernel checks that every safetensors header's tensor names
  equal the weight_map names for that file and that header length plus data end equals the file size.

### 3. Download method on the Kaggle CPU kernel

- **Library versions:** the pinned image ships huggingface_hub 1.11.0 (our E9 and E12 kernel logs: "Found existing
  installation: huggingface_hub 1.11.0").
- **hf_xet is the default:** from huggingface_hub 1.0, hf_xet is the default transfer backend and hf_transfer is fully
  removed. `HF_HUB_ENABLE_HF_TRANSFER` is ignored, and `HF_XET_HIGH_PERFORMANCE` replaces it
  (https://huggingface.co/docs/huggingface_hub/main/en/concepts/migration, https://huggingface.co/blog/huggingface-hub-v1).
  The kernel logs the huggingface_hub / hf_xet / hf_transfer versions it finds.
- **Primary method:** `hf_hub_download(REPO, name, revision=REV, local_dir=/kaggle/working/<model>)`, one file at a time,
  small files first, with `HF_XET_HIGH_PERFORMANCE=1`. `HF_XET_CHUNK_CACHE_SIZE_BYTES=0` turns off the chunk cache so
  there is no second copy. `HF_HOME` is in /tmp, so nothing else lands in the persisted output. There are 2 attempts
  (the second with `force_download=True`).
- **Fallback:** a stdlib parallel HTTP Range downloader (16 threads, 64 MiB ranges, 6 retries per range), 2 attempts.
  - It resolves the CDN URL once per file and re-resolves on 401/403/410 (an expired signature).
  - Anonymous HF limits are 3,000 resolver calls per 5 min per IP (https://huggingface.co/docs/hub/rate-limits), and
    Kaggle egress IPs are probably shared, so a 429 waits for the `RateLimit` reset time.
  - huggingface_hub 1.2 and later also handle 429 by itself (same page).
- **Disk:** the checkpoint is written straight into `/kaggle/working`, with no staging copy. The kernel fails early if
  free space there is below the needed bytes plus 100 MB. It removes `.cache/huggingface` and partial files from the
  model dir before saving.
- **Time estimate:** no Kaggle to HF throughput figure was found.
  - At 100-300 MB/s (typical for xet on a cloud VM) the 16.8 GB takes about 1-3 min. At 20 MB/s it takes about 14 min.
    sha256 adds about 1 min.
  - Kaggle then persists about 17 GB of output after the run (time unknown).
  - The CPU kernel has a 12 h cap, so there is ample slack.
  - Locally (about 1.2 MB/s home link) the small files plus mtp took 264 s.

### 4. Merlin source

- **Tag:** `merlin-v3.0.0` exists. `git ls-remote` shows annotated tag object
  `72780a84c34035d9ec79bd32fea1ff0a9634631f`, which peels to commit `ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5`
  (https://api.github.com/repos/halt95/merlin-engine/git/refs/tags/merlin-v3.0.0).
- **Release:** "Merlin 3.0.0", published 2026-10-22 (https://github.com/halt95/merlin-engine/releases/tag/merlin-v3.0.0).
  Assets: `vllm-0.32.0+merlin.7002f56537-cp313-cp313-linux_x86_64.whl` (227,044,713 B) and its `.sha256`. Inside the
  tag, `release/merlin-wheel.sha256` says `5c33b3dc...c82d`. P2 fetches the wheel.
- **Download:** `https://codeload.github.com/halt95/merlin-engine/tar.gz/refs/tags/merlin-v3.0.0`, then the
  `github.com/.../archive/...` URL, then `git clone --depth 1 --branch` + `git archive` as the last resort.
- **Checks:**
  - The tarball's pax global header `comment` must equal the commit id. Both codeload and `git archive` write it; this
    is the authority.
  - The tarball sha256 seen on 2026-10-29 (`1cd121b6...2b06`, 16,220,857 B) only gives a warning on mismatch, because
    GitHub archives are not guaranteed byte-stable.
  - These files must exist: `release/install-env.sh`, `release/requirements-0.32.lock`, `release/download-sizes.json`,
    `release/merlin-wheel.sha256`, `merlin_cli.py`, `models/qwen38-27b/{checkpoint.sha256, serve.sh, serve-tp4.sh,
    VERSION}`.
  - `checkpoint.sha256` must equal the embedded copy.
  - `download-sizes.json` 27b must give this revision, 15 files and 17,049,970,203 bytes. `VERSION` = 3.0.0.

## What the kernel does, step by step

1. Removes the HF token and offline variables from the environment. Sets HF_HOME to /tmp, xet high-performance mode,
   chunk cache 0, and timeouts.
2. Logs the environment: Python, huggingface_hub, hf_xet and hf_transfer versions, nproc, RAM, and free disk in
   /kaggle/working and /tmp.
3. Capacity check: free space in /kaggle/working must be at least the checkpoint bytes plus 100 MB, or it fails.
4. Fetches the HF tree at the pinned revision and requires exactly the 15 embedded names, sizes and LFS sha256. Any
   drift fails the run before any download.
5. Downloads each file, smallest first: hf_hub_download x2, then the parallel Range fallback x2. After each attempt it
   checks size and sha256. On a mismatch it deletes the file and retries. It records the method, seconds, MB/s and hash
   seconds for every attempt.
6. Structural checks: safetensors headers against the index weight_map, data end against file size, and a size
   re-check of all files.
7. Merlin source tarball: download, commit-id check, extraction to /tmp, content checks, then deletion of the extracted
   tree.
8. Cleanup: removes `.cache/huggingface` and `*.p1part` from the model dir. Totals the checkpoint bytes, which must be
   17,049,970,203. Total output must be under 19.5e9 B.
9. Writes `q27_e23_p1_manifest.json` (only on ALL OK) and `p1_report.json` (always). Prints one line per file, the
   merlin summary, errors and warnings, and `ALL OK` / `NOT OK`. Raises on NOT OK.

## Local checks run (2026-10-29, Windows, Python 3.14)

- **Build:** `python build_p1_kernel.py` passes all builder asserts. `compile()` of the cell passes, the notebook JSON
  and `kernel-metadata.json` load with `json`, and the metadata fields are correct.
- **Test A (hf_hub_download path):** huggingface_hub 1.31.0 + hf_xet 1.6.0, small files + model-mtp
  (`P1_MAX_FILE_BYTES=300000000`), real HF + GitHub.
  - All 14 files OK (sha256 match). kv_scales and mtp headers equal the weight_map.
  - Merlin commit OK, checkpoint.sha256 matches, download-sizes OK.
  - Result: ALL OK, rc 0, 264.5 s.
- **Test B (stdlib Range fallback path):** `python -I -S`, so huggingface_hub cannot be imported; 1 MiB ranges, so
  tokenizer.json takes 13 ranges.
  - All 13 files OK through `http`.
  - Result: ALL OK, rc 0.
- **Test C (fail loudly on a mismatch):** one embedded hash (config.json) tampered.
  - All 4 attempts reported MISMATCH and the file was deleted.
  - Result: NOT OK, RuntimeError, rc 1, no manifest written.
- **Test D (no network, incidental):** an earlier run hit a local DNS failure. HF tree retries x5, then "fatal" in the
  report. Result: NOT OK, rc 1, no manifest.
- The full 16.8 GB download was not run locally (D: has about 12 GB free). The first full run is the Kaggle run.

## Risks / open items

1. **Read speed of kernel output in the GPU notebook is unknown (NFS or GCSFUSE).** Kaggle staff recommend Kaggle Models
   for weights. If the first GPU session shows a slow mount, convert the output to a Kaggle Model. That has to be done
   in an editor session with a token secret (kagglehub.model_upload), or in the web UI if it allows it; a batch kernel
   cannot do it (see 1.).
2. **No documented per-file limit:** a 15.66 GiB single file in the output has not been tested on Kaggle. The total
   (15.89 GiB) is under the 20 GB cap. If the output save fails, the fallback is a Kaggle Model or Dataset upload of the
   same files.
3. **Shared egress IP and HF anonymous rate limits:** the kernel waits on 429 and retries, but a heavily shared IP could
   slow it. An HF token cannot be used (no secrets in batch runs), and none is needed for a public repo.
4. **`HF_XET_HIGH_PERFORMANCE` stalls:** one community report says it can stall after a first burst. A stall that never
   errors would keep the hf attempt waiting. Remove the variable if the first run shows this; the fallback only takes
   over on errors.
5. **The attach targets the latest successful version.** `kernel_sources` has no version pin. Re-running P1 replaces the
   output, so the hashes still guard against drift.
6. **Private kernel:** a GPU notebook on another account cannot attach it unless it is shared or public.
7. **Mount path:** not documented. Use the sentinel finder, not a hard-coded path.
8. **Merlin glibc 2.38 vs the Ubuntu 22.04 image:** this is P2's item. P1 only ships the source tarball. The release
   wheel (227 MB) is not included, because P2 owns it.
