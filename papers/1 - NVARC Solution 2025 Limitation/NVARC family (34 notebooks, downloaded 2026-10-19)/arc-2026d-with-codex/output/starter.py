import os
import time
import json
import torch
import argparse
import traceback
import numpy as np
import torch.multiprocessing as mp

# ──────────────────────────────────────────────────────────────
#   Debug mode: only these puzzles run during manual execution.
#   In Kaggle rerun mode, all puzzles are processed automatically.
#   All IDs are from the EVALUATION set (arc-agi_evaluation_challenges.json)
#   to ensure they are found during manual runs.
# ──────────────────────────────────────────────────────────────
RUN_FULL_EVAL = True
NPROCS = 4  # number of worker processes

# DEBUG_PUZZLES = [
#     "0934a4d8",  # tier=3, fine=1.7619, tags=tier3,dynamic,rare,high_ratio,extreme_ratio,many_colors,large,dense_non_bg
#     "20a9e565",  # tier=3, fine=1.5125, tags=tier3,dynamic,rare,multi_query,high_ratio,extreme_ratio,many_colors,large,sparse_bg
#     "4c7dc4dd",  # tier=3, fine=1.8181, tags=tier3,dynamic,rare,multi_query,high_ratio,extreme_ratio,large,dense_non_bg,low_train
#     "89565ca0",  # tier=3, fine=1.4844, tags=tier3,dynamic,high_ratio,extreme_ratio,many_colors,large
#     "136b0064",  # tier=1, fine=0.4736, tags=tier1,hard_tier1,dynamic,rare,sparse_bg
#     "291dc1e1",  # tier=1, fine=0.5169, tags=tier1,hard_tier1,dynamic,rare,many_colors,dense_non_bg
#     "16b78196",  # tier=2, fine=0.4400, tags=tier2,rare,large,sparse_bg,low_train
#     "271d71e2",  # tier=1, fine=0.1552, tags=tier1,easy_static,easy,rare,dense_non_bg
#     "28a6681f",  # tier=1, fine=0.1733, tags=tier1,easy,rare
#     "2b83f449",  # tier=1, fine=0.1705, tags=tier1,easy_static,easy,rare,low_train
#     "2c181942",  # tier=2, fine=0.2887, tags=tier2,rare,dense_non_bg
#     "31f7f899",  # tier=1, fine=0.2092, tags=tier1,easy,rare,many_colors,dense_non_bg
#     "35ab12c3",  # tier=1, fine=0.2654, tags=tier1,rare,many_colors,sparse_bg
#     "3dc255db",  # tier=1, fine=0.1631, tags=tier1,easy,rare,sparse_bg
#     "409aa875",  # tier=1, fine=0.1853, tags=tier1,easy_static,easy,rare,dense_non_bg
#     "4c416de3",  # tier=2, fine=0.3368, tags=tier2,rare,many_colors,dense_non_bg
#     "64efde09",  # tier=2, fine=0.3802, tags=tier2,rare,many_colors,large,dense_non_bg,low_train
#     "6e453dd6",  # tier=1, fine=0.1484, tags=tier1,easy_static,easy,rare,dense_non_bg
#     "7491f3cf",  # tier=1, fine=0.2583, tags=tier1,rare,many_colors,dense_non_bg
#     "7b0280bc",  # tier=1, fine=0.3054, tags=tier1,rare,many_colors,dense_non_bg
#     "7b80bb43",  # tier=2, fine=0.2720, tags=tier2,rare,large,low_train
#     "7c66cb00",  # tier=2, fine=0.3800, tags=tier2,rare,large,dense_non_bg
#     "1818057f",  # tier=1, fine=0.1177, tags=tier1,easy_static,easy,dense_non_bg
#     "332f06d7",  # tier=1, fine=0.1380, tags=tier1,easy_static,easy,dense_non_bg
#     "53fb4810",  # tier=1, fine=0.2058, tags=tier1,easy_static,easy,dense_non_bg,low_train
#     "7666fa5d",  # tier=1, fine=0.1653, tags=tier1,easy_static,easy,dense_non_bg,low_train
#     "da515329",  # tier=1, fine=0.1382, tags=tier1,easy_static,easy,sparse_bg
#     "8f3a5a89",  # tier=1, fine=0.1640, tags=tier1,easy_static,easy,rare,dense_non_bg
#     "db695cfb",  # tier=1, fine=0.1465, tags=tier1,easy_static,easy,rare,dense_non_bg
#     "97d7923e",  # tier=1, fine=0.1644, tags=tier1,easy,sparse_bg
#     "de809cff",  # tier=1, fine=0.2533, tags=tier1,dense_non_bg,low_train
#     "c7f57c3e",  # tier=1, fine=0.2280, tags=tier1,rare,dense_non_bg,low_train
#     "cb2d8a2c",  # tier=1, fine=0.1766, tags=tier1,easy_static,easy,multi_query,dense_non_bg
#     "58490d8a",  # tier=2, fine=0.7666, tags=tier2,dynamic,high_ratio,dense_non_bg
#     "dbff022c",  # tier=1, fine=0.2338, tags=tier1,rare,many_colors
#     "5961cc34",  # tier=2, fine=0.2485, tags=tier2,dense_non_bg
#     "80a900e0",  # tier=2, fine=0.2867, tags=tier2,large,low_train
#     "269e22fb",  # tier=1, fine=0.4397, tags=tier1,dynamic,multi_query,dense_non_bg
#     "8b9c3697",  # tier=2, fine=0.2593, tags=tier2,dense_non_bg
#     "e376de54",  # tier=1, fine=0.2253, tags=tier1,rare,dense_non_bg
#     "b99e7126",  # tier=2, fine=0.3803, tags=tier2,large,dense_non_bg
#     "c4d067a0",  # tier=2, fine=0.2763, tags=tier2,large,dense_non_bg
#     "135a2760",  # tier=1, fine=0.2078, tags=tier1,easy,rare,dense_non_bg,low_train
#     "4a21e3da",  # tier=1, fine=0.1680, tags=tier1,easy_static,easy,multi_query,dense_non_bg
#     "71e489b6",  # tier=1, fine=0.1528, tags=tier1,easy_static,easy,multi_query
#     "800d221b",  # tier=1, fine=0.2840, tags=tier1,rare,many_colors,dense_non_bg
#     "dd6b8c4b",  # tier=1, fine=0.1403, tags=tier1,easy_static,easy,rare,multi_query,dense_non_bg
#     "36a08778",  # tier=1, fine=0.0788, tags=tier1,easy_static,easy,rare,multi_query,dense_non_bg
# ]
# DEBUG_PUZZLES = [
#     # Same-rank selection candidates
#     "142ca369",  # rank 7 / 7
#     "3dc255db",  # rank 5 / 5
#     "4c7dc4dd",  # rank 3 / 3
#     "800d221b",  # rank 4 / 4
#     "dfadab01",  # rank 3 / 3

#     # Control cases (different ranks)
#     "62593bfd",  # rank 6 / 8
#     "8b7bacbf",  # rank 9 / 5

#     # # Already rescued by dual selection
#     # "20270e3b",
#     # "2d0172a1",
#     # "2ba387bc",
#     # "31f7f899",
#     ]


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


def local_worker(rank, queue, end_time, import_lock):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["UNSLOTH_DISABLE_RL_PATCH"] = "1"
    os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

    try:
        # Serialize Unsloth import/patching. This avoids multi-process import race.
        with import_lock:
            print(f"[Rank {rank}] importing arc_solver...", flush=True)
            from arc_solver import worker
            print(f"[Rank {rank}] arc_solver import ok", flush=True)

        print(f"[Rank {rank}] start!", flush=True)
        worker(rank, queue, end_time)
        print(f"[Rank {rank}] done!", flush=True)

    except Exception as e:
        print(f"[Rank {rank}] WORKER_EXCEPTION {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--end-time",
        type=float,
        default=0.0,
        help="Unix timestamp when execution must stop. If 0, compute 12 hours from now.",
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

    manager = mp.Manager()
    queue = manager.Queue()
    import_lock = manager.Lock()

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

    selected_keys = []

    for key in keys_sorted:
        if not rerun_mode and not RUN_FULL_EVAL:
            if key not in DEBUG_PUZZLES:
                continue
        selected_keys.append(key)
        queue.put(key)

    print(
        f"[Starter] Queued {len(selected_keys)} puzzles "
        f"(rerun_mode={bool(rerun_mode)}, RUN_FULL_EVAL={RUN_FULL_EVAL})",
        flush=True,
    )

    for _ in range(NPROCS):
        queue.put(None)

    mp.spawn(local_worker, args=(queue, end_time, import_lock), nprocs=NPROCS)
