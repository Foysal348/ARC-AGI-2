# E14-A start failure: the INT4 model did not mount (2026-10-23)

## What happened

| UTC | Account | Kernel | Result |
|---|---|---|---|
| 13:52 → 13:59 | hosen42 | E14A v1 (nb sha a5318dda) | ERROR after 7 min; log `[]`, 0 output files, quota unchanged (36.56). The page error was not read. |
| 14:10 | foysal | DIAG pinned image | COMPLETE; all 3 models mounted (NFS 192.168.3.2), INT4 read 81 MiB/s |
| 14:15 | foysal | DIAG latest image | COMPLETE; all 3 models mounted (NFS 192.168.5.2), INT4 read 159 MiB/s |
| 14:31 | hosen42 | DIAG pinned image | COMPLETE; all 3 models mounted (NFS 192.168.5.2), INT4 read 160 MiB/s |
| 14:34:53 → 14:35:51 | foysal | E14A v1 (same sha) | RUNNING at 14:35:16, ERROR at 14:35:51; log `[]` (2 bytes), 0 files, quota unchanged (26.42) |

The Kaggle page error for the foysal E14A run (pasted by the user):

```
mount data: ERRORED_MOUNTING_DATASET: DataSourceUrl.addDataset(gs://kagglesdsdata/models/765212/1030159 =>
/tmp/fs/kaggle_...-361491395-webtier/ctrfs/kaggle/input/models/foysalemonshanto/qwen3-8-27b-int4-redhat/pytorch/w4a16-g128/1):
retry budget exhausted (30 attempts): rpc error: code = Internal desc = dataset loading failed 3 load attempts0
```

## What this proves

1. **The notebook never ran.** Kaggle failed while mounting the inputs, before the first cell. That is why the log is empty and no quota was used. The API (`kernels_status.failure_message`, `list_kernel_session_output`) does not show this error; only the web page does.
2. **The failing input is the INT4 model:** `qwen3-8-27b-int4-redhat/PyTorch/w4a16-g128` v1, model instance 765212, version 1030159, 19.47 GB. The model instance itself is fine (API: version 1, `totalUncompressedBytes` 19472990394).
3. **The same model mounted fine three times in the 25 minutes before**, in the diag runs. They used the same push script: the only differences are notebook, slug and title. Model ids, datasets, docker image, machine and settings are identical, and the diag runs mounted the same GCS source `models/765212/1030159`.
4. **The E14A ipynb has no Kaggle data metadata.** Its notebook metadata has only `kernelspec` and `language_info`, the same as the diag. So the notebook cannot point Kaggle at a different input.

## Open question

E14A failed 2 of 2 and the diag succeeded 3 of 3. The only difference is the notebook content: 745 kB, with cells 3 and 5 at 250 kB and 287 kB of embedded code, against 2.9 kB for the diag.

- Kaggle's data mount should not depend on notebook content, so a transient failure on the data-loading side is the leading reading.
- In the successful runs, the model was served from an existing NFS cache copy. A host without a cached copy has to load 19.5 GB from GCS, and that load is what failed ("dataset loading failed 3 load attempts").
- The 2-of-2 pattern is still unexplained. A small test separates the two readings:
  - **T1:** re-push E14A unchanged (v2).
    - If it mounts and starts, the failure was transient.
  - **T2, only if T1 fails again:** push the diag notebook padded to about 745 kB.
    - If T2 also fails, notebook size triggers it, and the fix is to move the embedded modules into a dataset.
    - If T2 mounts, something else in E14A is the trigger.

Each test costs about 0.05-0.2 quota-hours, because a mount failure uses none.

## T1 and T2 results (both pushed on foysal at 14:44 UTC)

The two notebooks were pushed 14 s apart.

**T1, E14A v2 (unchanged, sha a5318dda):**
- Status went RUNNING at 14:44:55 and ERROR at 14:45:30.
- The page error was the same INT4 mount error as before: `models/765212/1030159`, session 361493826.
- E14A has now failed 3 of 3, every time on the INT4 mount.

**T2, padded diag (727 kB, E14A source as comments, sha 7f77ab35), COMMIT run:**
- Status was RUNNING from 14:44:55, but the Logs tab stayed empty: 412 s in the user's screenshot.
- The commit never started its code. The user stopped it at about 14:55, after about 11 min (CANCEL_REQUESTED, then CANCEL_ACKNOWLEDGED).

**The same padded notebook, run INTERACTIVELY by the user in the Kaggle editor:**
- Code ran at 14:54:01.
- All inputs mounted: INT4 on NFS 192.168.5.2 with copy dmc9x3mtx4n0, read at 87.9 MiB/s.
- The diag finished in 22.3 s, and the pad cell printed "pad cell ok".
- Correction: I first read this output as the commit's output. It came from the user's interactive session.

## Conclusion: in COMMIT mode, the notebook's size breaks the start

T2 is the decisive control. Its code is identical to the small diag; only about 724 kB of comments were added. The small diag commit started 3 of 3 times. The padded commit did not start in 11 min.

| Notebook | Size (raw) | Mode | Result |
|---|---|---|---|
| diag × 3 | 3.9 kB | commit | started in about 1 min, OK |
| E12 SMOKE1 / EVAL30 / EVAL6 / EVAL6R5 / EVAL12R5_LOW (10-19 to 10-21) | 329-347 kB | commit | ran, outputs in `Experiment Results/Kaggle Runs` |
| E12 rev 6d G4B, sha 1bdadd82 (10-21) | **407 kB** | commit | ran (E12 rev 6d G4B GPU run analysis) |
| E12 rev 6g G4B (10-22, esfersami50), _REBUILD | **444 kB** | commit | **ERROR after about 114 min, empty log, no outputs, quota used**: the same never-started pattern |
| padded diag | 727 kB | commit | not started after 11 min, quota counted while it hung (stopped) |
| E14A × 3 | 745 kB | commit | INT4 mount error (7 min, 1 min, 1 min), no quota |
| padded diag | 727 kB | interactive | OK |

- **The limit:** somewhere between **407 kB (ran) and 444 kB (never ran)**, for commit (batch) runs. As the JSON string inside the save request, those sizes are 424 kB and 464 kB.
- **What this also explains:** the 10-22 rev 6g G4B loss (about 3.8 quota-hours on esfersami50, cause "open" in the README). It is very likely the same limit.
- **Submissions:** the hidden-test rerun is a batch run as well, so a notebook above the limit is a risk there too.
- **Fix:** move the two payloads into a private dataset and load them with a sha check. The payloads are the E14 modules (cell 3, 250 kB) and the Duck package zip (cell 5, 287 kB). The notebook shrinks to about 200 kB.
- **Rule from now on:** keep every Kaggle notebook under 400 kB.
