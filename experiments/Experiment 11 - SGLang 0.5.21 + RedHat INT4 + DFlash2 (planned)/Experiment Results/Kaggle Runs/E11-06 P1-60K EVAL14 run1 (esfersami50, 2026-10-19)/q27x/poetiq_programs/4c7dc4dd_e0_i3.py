import numpy as np
from collections import Counter, defaultdict

def _border_is_universal(border, bg):
    vals = set(border.flat)
    if len(vals) == 1:
        return vals.pop() == bg
    if len(vals) == 2:
        a, b = vals
        return a == bg or b == bg
    return False

def _find_boxes(g, n, min_boxes=2, require_fill=False, require_interior=False):
    h, w = g.shape
    boxes = []
    for size in range(3, min(h, w) + 1):
        k = size - 2
        if k != n:
            continue
        for r in range(h - size + 1):
            for c in range(w - size + 1):
                top = g[r, c:c+size]
                if len(set(top)) != 1:
                    continue
                bot = g[r+size, c:c+size]
                if len(set(bot)) != 1 or set(bot) != set(top):
                    continue
                left = g[r:r+size+1, c]
                right = g[r:r+size+1, c+size]
                if len(set(left)) != 1 or len(set(right)) != 1:
                    continue
                bc = top[0]
                if left[0] != bc or right[0] != bc:
                    continue
                if require_fill:
                    interior = g[r+1:r+size, c+1:c+size]
                    if (interior == bc).all():
                        continue
                if require_interior:
                    interior = g[r+1:r+size, c+1:c+size]
                    if (interior == bc).any():
                        continue
                boxes.append((r, c, size, bc))

    if len(boxes) < min_boxes:
        return []

    by_bc = defaultdict(list)
    for b in boxes:
        by_bc[b[3]].append(b)

    best = None
    for bc, items in by_bc.items():
        items.sort(key=lambda x: (x[0], x[1]))
        chosen = []
        for b in items:
            if any(abs(b[0]-o[0]) < n and abs(b[1]-o[1]) < n for o in chosen):
                continue
            chosen.append(b)
        if len(chosen) >= min_boxes:
            if best is None or len(chosen) > len(best[1]) or (
                len(chosen) == len(best[1]) and sum(len(v) for v in by_bc.values() if v == items) > 0
            ):
                best = (bc, chosen)
    return best[1] if best else []

def _score_candidate(g, boxes, n, bg):
    h, w = g.shape
    interiors = []
    for r, c, size, bc in boxes:
        interiors.append(g[r+1:r+size, c+1:c+size])

    masks = []
    for it in interiors:
        m = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                v = it[i, j]
                if v != 0 and v != bg:
                    m[i, j] = 1
        masks.append(m)

    x = np.zeros((n, n), dtype=int)
    for m in masks:
        x ^= m
    nz = int(x.sum())
    if nz == 0:
        return None

    out = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if x[i, j] == 0:
                continue
            vals = [it[i, j] for it in interiors if it[i, j] != 0 and it[i, j] != bg]
            if not vals:
                out[i, j] = 2
                continue
            if len(set(vals)) == 1:
                out[i, j] = vals[0]
            else:
                # Prefer the special color 6 if it occurs an odd number of times,
                # otherwise fall back to the most common non-background value.
                c6 = sum(1 for v in vals if v == 6)
                if c6 % 2 == 1:
                    out[i, j] = 6
                else:
                    cnt = Counter(vals)
                    out[i, j] = cnt.most_common(1)[0][0]
    return out

def transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid)
    h, w = g.shape

    for n in range(1, min(h, w)):
        if h - 2*n < 2 and w - 2*n < 2:
            continue
        candidates = []
        for bg in range(10):
            boxes = _find_boxes(g, n, min_boxes=2, require_fill=True, require_interior=True)
            if not boxes:
                continue
            out = _score_candidate(g, boxes, n, bg)
            if out is not None:
                candidates.append((len(boxes), bg, out))
        if candidates:
            candidates.sort(key=lambda x: (-x[0], x[1]))
            return candidates[0][2].astype(np.int64, copy=False)

    return np.zeros((1, 1), dtype=int)