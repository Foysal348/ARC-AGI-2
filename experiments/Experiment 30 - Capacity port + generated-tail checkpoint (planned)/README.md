# Experiment 30: capacity port + generated-tail checkpoint (2026-11-01)

**Goal:** raise the resident streams N, and cut the generated-tail re-prefill, on klee100 + DFlash2 (Merlin v3.0.0 ad42e37, Kaggle 4×L4). This is the N lever of the 1500 tok/s plan:
- `Experiment 25/Experiment Review/EXPERIMENT_PRIORITY_LIST_K100_FINAL_2026-11-01.md`, items 2-2, 2-8, 2-11 and Mamba padding;
- the throughput identity tok/s = N × AL / (2 × tick).

## Starting facts
- **Pools:** MTP pool 480 blocks, F = 6. DFlash2 SW-1024 pool 540, F ~12 (6 Mamba + 6 sliding-window), so it holds fewer long-context residents than MTP (64k: 18-20 vs 21).
- **Mamba page padding:** about 21 %.
- **Indexer KV fp8:** 480 → 508 blocks (kit R9).
- **Re-prefill:** 72.7 % of fresh prefill is the previous turn's generated tokens. A perfect checkpoint is worth 6.0-8.2 % of the run (`Experiment 25/Experiment Results/prep_0gpu/duck_timeline/`).
- **Generated-tail design:** `Experiment 25/Experiment Review/DESIGN_generated_tail_checkpoint_2026-11-01.md`. Option B: 70 % of the saving, about 4-6 % of the run, scheduler and KV-manager only.

## Work items
| # | Item | Status |
|---|---|---|
| C1 | F reduction / live-block admission | agent (source analysis + CPU pool measurement), `_scratch/capacity_port/` |
| C2 | Mamba page padding | same agent |
| C3 | Draft / target KV-bucket sharing (vLLM #58207 / #58638) | same agent |
| C4 | Generated-tail checkpoint, Option B: `gentail_seam.py` prototype + CPU tests + GPU test spec | agent, `_scratch/gentail_ckpt/` |
| R1 | Upstream resource search (GPT; prompt below, also in `Experiment Review/GPT_RESEARCH_PROMPT_capacity_2026-11-01.md`) | user runs it on GPT |

## Method
- Every patch is a seam-style file (apply / verify / revert, pre / post shas, an env switch whose default leaves Merlin byte-identical).
- Capacity gains are measured on CPU with the real `get_kv_cache_configs` (the kit R9 path) before any GPU run.
- GPU tests use the real-failure-only gates and the matrix in the Exp 25 FINAL list (0.4-0.5).
