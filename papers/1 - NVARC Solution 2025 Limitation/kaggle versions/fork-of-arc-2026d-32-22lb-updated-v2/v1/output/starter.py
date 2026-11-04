import os
import time
import json
import torch
import argparse
import traceback
import numpy as np
import torch.multiprocessing as mp


NPROCS = 4

DEBUG_PUZZLES = [
    # no-output / infrastructure or decode-save suspects
    "271d71e2",  # NO OUTPUT, easy_static/rare
    "291dc1e1",  # NO OUTPUT or unstable, hard_tier1/dynamic/rare

    # dynamic / generation bottleneck
    "136b0064",  # UNSOLVED, hard_tier1/dynamic/rare
    "269e22fb",  # UNSOLVED, dynamic + multi-query
    "0934a4d8",  # UNSOLVED, tier3/dynamic/high_ratio/extreme_ratio
    "20a9e565",  # UNSOLVED, tier3/dynamic/multi-query/high_ratio/extreme_ratio
    "89565ca0",  # UNSOLVED, tier3/dynamic/high_ratio/extreme_ratio

    # solved but dual missed -> selection-depth test
    "4c7dc4dd",  # SOLVED, Dual No in 48-run, rank 4; tier3/multi-query/extreme_ratio
    "7b0280bc",  # SOLVED, sometimes Dual No; rank 1-2 depending run
    "2d0172a1",  # SOLVED, Dual No, rank 2/2

    # extra solved but dual missed -> selection-depth test
    "31f7f899",  # SOLVED, Dual No, rank 2/2
    "3dc255db",  # SOLVED, Dual No, rank 3/3
    "8f3a5a89",  # SOLVED, Dual No, rank 2/2
    "800d221b",  # SOLVED, Dual No, rank 2/2

    # solved + dual yes -> guardrail, ensure selection change does not regress
    "6e453dd6",  # SOLVED, Dual Yes
    "7b80bb43",  # SOLVED, Dual Yes
    "1818057f",  # SOLVED, Dual Yes
    "4a21e3da",  # SOLVED, Dual Yes, multi-query

    # unsolved but useful buckets
    "cb2d8a2c",  # UNSOLVED, easy_static + multi-query
    "58490d8a",  # UNSOLVED, dynamic + high_ratio-ish
]

def get_puzzle_priority(key, data):
    train = data[key]["train"]
    n_train = len(train)

    avg_input_size = np.mean([
        np.prod(np.shape(x["input"])) for x in train
    ])

    size_ratios = [
        max(np.prod(np.shape(x["output"])),
            np.prod(np.shape(x["input"]))) /
        max(min(np.prod(np.shape(x["output"])),
                np.prod(np.shape(x["input"]))), 1)
        for x in train
    ]

    max_ratio = max(size_ratios)
    avg_ratio = np.mean(size_ratios)

    n_colors = len(set(
        int(c)
        for x in train
        for arr in [x["input"], x["output"]]
        for c in np.unique(arr).tolist()
    ))

    is_dynamic = any(
        np.shape(x["input"]) != np.shape(x["output"])
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

    # Avoid fragile file-based worker chain. A small stagger reduces startup pressure.
    if rank > 0:
        time.sleep(15 * rank)

    try:
        from arc_solver import worker

        print(f"[Rank {rank}] start!", flush=True)
        worker(rank, queue, end_time)
        print(f"[Rank {rank}] done!", flush=True)

    except Exception as e:
        print(
            f"[Rank {rank}] WORKER_EXCEPTION {type(e).__name__}: {e}",
            flush=True,
        )
        traceback.print_exc()
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--end-time",
        type=float,
        default=0.0,
        help="Unix timestamp when execution must stop. If 0, use 12 hours from now.",
    )
    args = parser.parse_args()

    if args.end_time == 0.0:
        end_time = time.time() + 12 * 3600
    else:
        end_time = args.end_time

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    if rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    with open(test_path, "r") as f:
        data = json.load(f)

    queue = mp.Manager().Queue()

    keys_sorted = sorted(
        data.keys(),
        key=lambda k: get_puzzle_priority(k, data),
    )

    tiers = [get_puzzle_priority(k, data)[0] for k in keys_sorted]
    print(
        f"[Starter] Puzzle tiers - "
        f"Easy: {tiers.count(1)}, "
        f"Hard: {tiers.count(2)}, "
        f"Extreme: {tiers.count(3)}",
        flush=True,
    )
    print(
        f"[Starter] First: {keys_sorted[0]}, Last: {keys_sorted[-1]}",
        flush=True,
    )

    if rerun_mode:
        selected_keys = keys_sorted
    else:
        debug_set = set(DEBUG_PUZZLES)
        selected_keys = [k for k in keys_sorted if k in debug_set]
        missing = [k for k in DEBUG_PUZZLES if k not in data]
        print(f"[Starter] Manual debug mode: {len(selected_keys)} puzzles selected", flush=True)
        print(f"[Starter] Selected: {selected_keys}", flush=True)
        if missing:
            print(f"[Starter] Missing DEBUG_PUZZLES: {missing}", flush=True)

    for key in selected_keys:
        queue.put(key)

    for _ in range(NPROCS):
        queue.put(None)

    mp.spawn(local_worker, args=(queue, end_time), nprocs=NPROCS)
