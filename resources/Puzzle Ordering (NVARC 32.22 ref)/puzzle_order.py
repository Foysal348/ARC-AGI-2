"""Puzzle ordering copied VERBATIM from the NVARC-based reference notebooks
foysalemonshanto/arc-2026d-32-22lb-updated (starter.py) and arc-2026d-32-22lb-updated-v2 (starter.py, run 331102752).
get_puzzle_priority() below is byte-identical to the reference; do not edit it.

Reference usage (starter.py, __main__):
    keys_sorted = sorted(data.keys(), key=lambda k: get_puzzle_priority(k, data))
    -> ascending (tier, fine): easiest first; ties keep the JSON file's key order (Python's sort is stable).
    -> every task goes onto ONE shared queue in that order; 4 workers pull from it until end_time.
"""
import numpy as np


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


def order_task_ids(data):
    """the reference ordering: data = the challenges dict exactly as json.load returned it (file key order)"""
    return sorted(data.keys(), key=lambda k: get_puzzle_priority(k, data))


def order_units(data):
    """our per-test-input units (task#q) in the reference task order; a task's test inputs stay together, q ascending"""
    return [f"{t}#{q}" for t in order_task_ids(data) for q in range(len(data[t]["test"]))]
