# E26 K100 QUALITY CPU kits: prerequisite

The kits built by `build_e26_k100q_cpu_kit.py` (`q27-e26-k100q-kit-{units,units-report,e2e-rerun,e2e-commit,real}`) read the
NEW version of the dataset `foysalemonshanto/q27-e25-nb0-tf-ref`. That version is staged, not pushed:
`D:/ARC AGI 2/_scratch/push_e25/ds_e25_nb0_tf_ref_k100q/`. It adds `k100q_payload_emb.json`, `K100Q_PAYLOAD.json`,
`k100q_report.py` (the report lib, no longer embedded) and `nb4_ppack.py` (the Q-defer patch, NB4S's sha 8dadf076) to the
current files. Every current file stays byte-identical (`SHA256SUMS.json` unchanged). The notebook pins the sha of each new
file, so a version that is missing a file, or holds an older copy, fails the kits.

Order:

1. Push the staged dataset version. Wait until Kaggle has finished processing it. `dataset_sources` is unversioned, so a kit
   pushed before then mounts the old version.
2. Push the kits (all five can run in parallel on foysalemonshanto).
3. Push the GPU notebook only after the kits are ALL OK and the user gives the go.

If the old version is mounted, every kit prints `PREREQ NOT MET: ...` and reports `tf_ref_new_version_mounted` as NOT OK.
The e2e and real kits also fail in CELL 1 (`K100Q_PAYLOAD missing or sha mismatch`). In that case, push the dataset version
and rerun the kits. Nothing in the kits needs to change.

Rebuild the kits after every notebook rebuild: `python build_e26_k100q_cpu_kit.py`. The builder reads the notebook sha and
every expected value from `e26_k100q_build_facts.json` and from the notebook.
