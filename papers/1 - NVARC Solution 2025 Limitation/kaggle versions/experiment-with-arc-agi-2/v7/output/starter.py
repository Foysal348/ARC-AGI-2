import os
import time
import json
import torch
import argparse
import shutil
import numpy as np
import torch.multiprocessing as mp

DEFAULT_MODEL_PATH = "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"

# ──────────────────────────────────────────────────────────────
#   Debug mode: only these puzzles run during manual execution.
#   In Kaggle rerun mode, all puzzles are processed automatically.
#   All IDs are from the EVALUATION set (arc-agi_evaluation_challenges.json)
#   to ensure they are found during manual runs.
# ──────────────────────────────────────────────────────────────

DEBUG_PUZZLES = [
    # ── Core 4 (already present) ──
    "0934a4d8",
    "36a08778",
    "981571dc",
    "aa4ec2a5",

    # # ── New additions to test our fixes ──
    # "135a2760",      # Small grid (29x29), simple pattern → tests basic inference pipeline
    # "b5ca7ac4",      # Medium size (22x22), dynamic shape → tests dynamic baseline & dual selection
    # "7b0280bc",      # Large grid (30x30) → tests memory handling & empty batch guard
    # "8b9c3697",      # Dynamic shape change (23x27) → tests invert_mod & jagged row rejection
    # "a395ee82",      # Medium difficulty → tests eval_seed separation effect
    # "65b59efc",      # Large grid + dynamic → tests 1450s timeout & multi-batch processing
    # "7666fa5d",      # Heavy color permutation → tests permutation augmentation & score_aug diversity
    # "1818057f",      # Multiple possible correct solutions → tests dual selection (top_a vs top_b)
    # "2d0172a1",      # Near timeout risk → tests the 1450-second timeout logic
    # "3e6067c3",      # Dynamic + 2 training examples → tests eval_seed vs train_seed separation
    # "6e4f6532",      # Rare color (magenta/maroon) → tests color permutation augmentation

    # # ── NEW: Large + Dynamic (extreme ratio / timeout risk) ──
    # "20a9e565",      # Very large (880 cells), dynamic, ratio 50.0 → tests extreme scale & timeout
    # "4c7dc4dd",      # Max size (900 cells), dynamic, ratio 56.25 → tests memory & jagged row handling
    # "89565ca0",      # Large (557 cells), dynamic, ratio 51.33 → tests high ratio & score_aug diversity
    # "7b3084d4",      # Large (400 cells), dynamic, ratio 25.0 → tests invert_mod & multi-batch
    # "fc7cae8d",      # Large (484 cells), dynamic, ratio 19.36 → tests extreme ratio & empty batch guard

    # # ── NEW: Multi-query puzzles (batch processing & dual selection) ──
    # "1ae2feb7",      # 3 test queries → tests batch processing & dual selection on multi-query
    # "b6f77b65",      # 3 test queries → tests batch processing with multiple candidates
    # "13e47133",      # 2 test queries + rare colors (10 colors) → tests dual selection & color aug

    # # ── NEW: Rare Color + Multi-query (8 picks from EDA) ──
    # "195c6913",      # Size 550, Ratio 1.0, Rare, 2 queries
    # "62593bfd",      # Size 716, Ratio 1.0, Rare, 2 queries (largest)
    # "4e34c42c",      # Size 588, Ratio 9.58, Rare, 2 queries (scaling)
    # "a32d8b75",      # Size 600, Ratio 1.25, Rare, 2 queries
    # "38007db0",      # Size 418, Ratio 3.57, Rare, 2 queries
    # "3a25b0d8",      # Size 484, Ratio 5.32, Rare, 2 queries
    # "5dbc8537",      # Size 312, Ratio 2.22, Rare, 2 queries
    # "d8e07eb2",      # Size 660, Ratio 1.0, Rare, 2 queries

    # # ── NEW: Hardest Tier 1 (4 picks from EDA) ──
    # "6ffbe589",      # Size 400, Ratio 4.0, 8 colors, Dynamic=True → tests dynamic & scaling in Tier 1
    # "58f5dbd5",      # Size 386, Ratio 2.71, 9 colors, Dynamic=True → tests moderate scaling & color diversity
    # "5545f144",      # Size 259, Ratio 4.50, 7 colors, Dynamic=True → tests higher ratio (4.5) in Tier 1
    # "291dc1e1",      # Size 164, Ratio 2.75, 9 colors, Dynamic=True → tests dynamic with many colors
]


def get_puzzle_priority(key, data):
    train = data[key]['train']
    n_train = len(train)

    avg_input_size = np.mean([
        np.prod(np.shape(x['input'])) for x in train
    ])
    size_ratios = [
        max(np.prod(np.shape(x['output'])),
            np.prod(np.shape(x['input']))) /
        max(min(np.prod(np.shape(x['output'])),
                np.prod(np.shape(x['input']))), 1)
        for x in train
    ]
    max_ratio = max(size_ratios)
    avg_ratio = np.mean(size_ratios)

    n_colors = len(set(
        int(c)
        for x in train
        for arr in [x['input'], x['output']]
        for c in np.unique(arr).tolist()
    ))
    is_dynamic = any(
        np.shape(x['input']) != np.shape(x['output'])
        for x in train
    )

    if is_dynamic and max_ratio > 30:
        tier = 3
    elif is_dynamic and max_ratio > 5:
        tier = 2
    elif avg_input_size > 400:
        tier = 2
    else:
        tier = 1

    fine = (
        avg_input_size / 900 * 0.30 +
        (avg_ratio - 1) / 10 * 0.25 +
        n_colors / 10 * 0.20 +
        (0.25 if is_dynamic else 0.0)
    )

    if n_train >= 6 and tier < 3:
        fine *= 0.7

    return (tier, fine)


def local_worker(rank, queue, end_time):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["UNSLOTH_DISABLE_RL_PATCH"] = "1"
    os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

    # Wait for previous worker's ready file with a 300-second timeout
    if rank > 0:
        start_wait = time.time()
        while not os.path.exists(f"/kaggle/worker{rank-1}"):
            time.sleep(5)
            if time.time() - start_wait > 300:
                raise TimeoutError(
                    f"Worker {rank}: Timeout waiting for worker {rank-1} to create sync file."
                )

    from arc_solver import worker

    with open(f"/kaggle/worker{rank}", "w") as f:
        f.write("Ok")

    print(f"[Rank {rank}] start!")
    worker(rank, queue, end_time)
    print(f"[Rank {rank}] done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0,
                        help="Unix timestamp when execution must stop. If 0, compute 12 hours from now.")
    args = parser.parse_args()

    if args.end_time == 0.0:
        end_time = time.time() + 12 * 3600
    else:
        end_time = args.end_time

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    # Fail before spawning workers if the Kaggle model mount is wrong/missing.
    model_path = os.getenv("ARC_MODEL_PATH", DEFAULT_MODEL_PATH)
    model_config = os.path.join(model_path, "config.json")
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"[Starter] model directory does not exist: {model_path}")
    if not os.path.isfile(model_config):
        raise FileNotFoundError(f"[Starter] model config.json not found: {model_config}")
    print(f"[Starter] Model preflight OK: {model_path}")

    if rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    with open(test_path, "r") as f:
        data = json.load(f)

    # Start every run from a clean output/synchronization state.  This prevents
    # stale files from an earlier execution in the same Kaggle session from
    # contaminating selector ranking or causing workers to skip synchronization.
    output_dir = "/kaggle/inference_outputs"
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for worker_rank in range(4):
        sync_file = f"/kaggle/worker{worker_rank}"
        try:
            os.remove(sync_file)
        except FileNotFoundError:
            pass

    queue = mp.Manager().Queue()

    # ── EDA-based puzzle ordering ────────────────────────────
    keys_sorted = sorted(
        data.keys(),
        key=lambda k: get_puzzle_priority(k, data)
    )

    # Log tier breakdown
    tiers = [get_puzzle_priority(k, data)[0] for k in keys_sorted]
    print(f"[Starter] Puzzle tiers — "
          f"Easy: {tiers.count(1)}, "
          f"Hard: {tiers.count(2)}, "
          f"Extreme: {tiers.count(3)}")
    print(f"[Starter] First: {keys_sorted[0]}, "
          f"Last: {keys_sorted[-1]}")

    # In rerun mode: all puzzles (ordered). Manual mode: only DEBUG_PUZZLES.
    scheduled_keys = []
    for key in keys_sorted:
        if not rerun_mode and key not in DEBUG_PUZZLES:
            continue
        scheduled_keys.append(key)
        queue.put(key)

    run_manifest = {
        "rerun_mode": bool(rerun_mode),
        "scheduled_puzzles": scheduled_keys,
        "num_scheduled_puzzles": len(scheduled_keys),
        "model_path": model_path,
    }
    manifest_path = "/kaggle/working/arc_run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(run_manifest, f, indent=2)
    print(f"[Starter] Scheduled puzzles: {len(scheduled_keys)}")
    if not rerun_mode:
        print(f"[Starter] Debug puzzles: {scheduled_keys}")
    print(f"[Starter] Run manifest: {manifest_path}")

    # 4 sentinel values – one for each worker
    for _ in range(4):
        queue.put(None)

    mp.spawn(local_worker, args=(queue, end_time), nprocs=4)
