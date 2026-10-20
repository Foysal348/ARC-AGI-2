import numpy as np

def _extract_boxes(g, bg, border, fill, size):
    h, w = g.shape
    boxes = []
    for r in range(h - size - 1):
        for c in range(w - size - 1):
            if g[r, c] == border and all(g[r + i, c + j] == fill
                                         for i in range(1, size) for j in range(1, size)):
                boxes.append((r, c))
    if not boxes:
        return []
    boxes.sort(key=lambda b: (b[0], b[1]))
    res = []
    for r, c in boxes:
        if any(r != o[0] and c == o[1] for o in boxes if o != (r, c)):
            continue
        ok = True
        for k in range(1, size):
            if g[r + k, c] != border or g[r + k, c + size] != border:
                ok = False
                break
            if g[r, c + k] != border and g[r + size, c + k] != border:
                ok = False
                break
        if ok and all(g[r + size, c + j] == border for j in range(size + 1)):
            res.append((r, c))
    return res

def _consensus(g, size):
    best = None
    for bg in range(10):
        bg_mask = g == bg
        if not (0.25 <= bg_mask.mean() <= 0.80):
            continue
        for border in range(10):
            if border == bg:
                continue
            for fill in range(10):
                if fill == bg or fill == border:
                    continue
                boxes = _extract_boxes(g, bg, border, fill, size)
                if not boxes:
                    continue
                interiors = [g[r + 1:r + size, c + 1:c + size] for r, c in boxes]
                for use_box, val in ((0, 1), (1, 2)):
                    for keep, kmin in ((0, 0), (1, 2)):
                        cnt = np.zeros((size, size), dtype=int)
                        for it in interiors:
                            for i in range(size):
                                for j in range(size):
                                    v = it[i, j]
                                    if v == bg or v == border:
                                        continue
                                    if not keep:
                                        cnt[i, j, 0] += 1
                                    elif v == fill:
                                        cnt[i, j, 0] += 1
                                    elif v in (6, 8):
                                        cnt[i, j, 1] += 1
                            if use_box and keep:
                                pass
                        if use_box:
                            for i in range(size):
                                for j in range(size):
                                    cnt[i, j, use_box and 1] += 0
                        cand = np.full((size, size), 0, dtype=int)
                        for i in range(size):
                            for j in range(size):
                                if cnt[i, j, 0] >= kmin:
                                    cand[i, j] = fill
                                elif cnt[i, j, 1] >= kmin:
                                    cand[i, j] = 6
                        nz = int((cand != 0).sum())
                        if 4 <= nz <= size * size * 0.75:
                            key = (nz, -len(boxes), border, bg, border, fill, use_box, keep, size)
                            if best is None or key > best[0]:
                                best = (key, cand.copy())
    return best[1] if best is not None else None

def transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid)
    for size in (5, 4, 6, 3):
        out = _consensus(g, size)
        if out is not None:
            return out
    return np.zeros_like(g)