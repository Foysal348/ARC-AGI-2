# Experiment 35: Public-best rebuild

Started 2026-11-04 (BDT). User decision: take the best public parts (Domdolus Tolus / Darren Amadeus Martin LB 83.47 engine and harness, notnitsuj, andy2709, mikelou1), rebuild our own pipeline structure on top of them, then optimize further. Base engine = Domdolus stack (stock vLLM 0.31 + their patch cells), not a Merlin 0.32 port.

This experiment is also the main measurement work for the flagship paper: **"Test-time compute economics on fixed hardware: how much ARC-AGI-2 score each engine lever buys per token"** (score ≈ tokens generated × solve yield per token × selection efficiency). See paper 5.

## Steps

1. **Full read (local, no GPU).** Every Domdolus cell (`Build/domdolus_cells/cell00..46.txt`, from `resources/Kaggle Writeups/domdolus_tolus/domdolus-tolus-arc-agi-2.ipynb`) and the other public sources. Output: technique inventory, dependency list (datasets, models, tars, kernel caches), real weight / KV GiB per GPU. Reports in `Experiment Review/Agent Reports/`.
2. **Own copies.** Their public datasets / models copied to our accounts (Kaggle Model .bin wrap) so nothing depends on their uploads.
3. **Reproduce unchanged** on our account: speed and eval score must match theirs. (GPU: needs go.)
4. **Improvement bundle**, each lever measured on the live 4xL4 session first:
   - speculative decoding (they run MTP off): DFlash2 / MTP;
   - our scheduler and attempt policy (Exp 28 A/B/C);
   - concurrency 70 / 96 / 128 against the bigger pool;
   - pool growth arms: expert pruning (320-expert plan), sub-FP8 for the FP8 parts, KV 3-bit.
5. CPU kit, then GPU commit (go needed).

## Status

- 2026-11-04: folder built, cells copied, step 1 agents launched.
- 2026-11-04 (later): step 1 done. Five reports + verbatim final messages in `Experiment Review/Agent Reports/`. Synthesis in `FRAMEWORK.md` (base B0, bundles H and S, arms).
- 2026-11-04 (later): step 2 part 1 done. CPU kernel `foysalemonshanto/e35-domdolus-deps-copy` v1 (private) holds our copy of the utility (site-vllm.tar 8,243,025,920 B sha256 0a8dd7d2…, site-arc.tar 599,439,360 B sha256 7a4a6c91…, pivotools harness repo, manifest) and fast-startup `startup_caches_l4_v2.tar.gz.bin` (19,306,881 B sha256 4c8afb6a…), 8.25 GiB. Model (~92 GB) not copied yet: too big for a kernel output (19 GB); still read from `darrenamadeusmartin/qwen38-flash-next-q2-gsq-vllm` v2.
- Trace-engine handoff for the Gemma 4 teammate: `resources/Handoff Trace Engine/README.md`.
- 2026-11-05 10:50 BDT: step 3 smoke pushed as a GPU commit (user go; live session URL never connected): `hosen42/e35-b0-smoke75` v1 = Domdolus notebook unchanged except `VALIDATION_MODE = True`, `VALIDATION_TIME_MIN = 75`, plus an inserted cell 4 that saves all text traces to `/kaggle/working/traces.tar.gz` every 10 min and after the deadline. Notebook 612,665 B (over our 400 kB rule of thumb; the identical-size Domdolus notebook commits fine). Folder `Experiment Notebooks/E35_B0_SMOKE75/`.
