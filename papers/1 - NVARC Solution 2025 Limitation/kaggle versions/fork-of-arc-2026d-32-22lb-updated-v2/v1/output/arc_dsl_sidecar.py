import numpy as np


def valid_grid(x):
    x = np.asarray(x)
    return x.ndim == 2 and 0 < x.shape[0] <= 30 and 0 < x.shape[1] <= 30 and np.all((0 <= x) & (x <= 9))


def bbox_nonzero(a):
    a = np.asarray(a)
    ys, xs = np.where(a != 0)
    if len(ys) == 0:
        return None
    return a[ys.min():ys.max()+1, xs.min():xs.max()+1]


def bbox_color(a, color):
    a = np.asarray(a)
    ys, xs = np.where(a == color)
    if len(ys) == 0:
        return None
    return a[ys.min():ys.max()+1, xs.min():xs.max()+1]


def color_map_rule(inp, out):
    if inp.shape != out.shape:
        return None
    mp = {}
    for a, b in zip(inp.ravel(), out.ravel()):
        a = int(a)
        b = int(b)
        if a in mp and mp[a] != b:
            return None
        mp[a] = b
    return mp


def apply_color_map(a, mp):
    res = np.array(a, copy=True)
    for k, v in mp.items():
        res[a == k] = v
    return res


def verify_rule(train, fn):
    for ex in train:
        pred = fn(np.asarray(ex["input"], dtype=int))
        target = np.asarray(ex["output"], dtype=int)
        if not valid_grid(pred):
            return False
        if pred.shape != target.shape:
            return False
        if not np.array_equal(pred, target):
            return False
    return True


def propose_dsl_candidates(task):
    train = task["train"]
    test_in = np.asarray(task["test"][0]["input"], dtype=int)
    candidates = []

    rules = []

    rules.append(("identity", lambda x: np.asarray(x, dtype=int)))

    for k in [1, 2, 3]:
        rules.append((f"rot90_{k}", lambda x, k=k: np.rot90(x, k=k)))
    rules.append(("flipud", lambda x: np.flipud(x)))
    rules.append(("fliplr", lambda x: np.fliplr(x)))
    rules.append(("transpose", lambda x: np.swapaxes(x, 0, 1)))

    rules.append(("bbox_nonzero", lambda x: bbox_nonzero(x)))

    for color in range(1, 10):
        rules.append((f"bbox_color_{color}", lambda x, color=color: bbox_color(x, color)))

    # Consistent direct color map for same-size tasks.
    mp = None
    ok = True
    for ex in train:
        inp = np.asarray(ex["input"], dtype=int)
        out = np.asarray(ex["output"], dtype=int)
        cur = color_map_rule(inp, out)
        if cur is None:
            ok = False
            break
        if mp is None:
            mp = dict(cur)
        else:
            for k, v in cur.items():
                if k in mp and mp[k] != v:
                    ok = False
                    break
                mp[k] = v
        if not ok:
            break
    if ok and mp is not None:
        rules.append(("color_map", lambda x, mp=mp: apply_color_map(x, mp)))

    seen = set()
    for name, fn in rules:
        try:
            if verify_rule(train, fn):
                pred = fn(test_in)
                if valid_grid(pred):
                    key = tuple(map(tuple, np.asarray(pred, dtype=int)))
                    if key not in seen:
                        seen.add(key)
                        candidates.append((name, np.asarray(pred, dtype=int)))
        except Exception:
            pass

    return candidates


def add_dsl_sidecar_candidates(decoder, split_data, only_when_missing=True, max_per_task=2):
    added = {}
    for key in split_data.keys:
        if only_when_missing and key in decoder.decoded_results and decoder.decoded_results[key]:
            continue

        task = split_data.queries[key]
        cands = propose_dsl_candidates(task)
        if not cands:
            continue

        decoder.decoded_results.setdefault(key, {})
        added[key] = []

        for i, (name, sol) in enumerate(cands[:max_per_task]):
            decoder.decoded_results[key][f"{key}.dsl.{name}.{i}"] = {
                "beam_score": 0.0,
                "score_aug": [0.0] * 8,
                "solution": sol,
            }
            added[key].append(name)

    print("DSL sidecar added:", added)
    return added
