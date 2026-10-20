import numpy as np
from itertools import combinations


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a grid containing four square framed panels into the XOR of the
    four panel interiors.

    The four frames are detected by a mostly uniform border color.  The interior
    of each frame is extracted, the frame color is cleared inside the frame,
    and the four interiors are XORed in a 2x2 fashion:
        top    = top-left XOR top-right
        bottom = bottom-left XOR bottom-right
        result = top XOR bottom

    Each of top and bottom is horizontally normalized: if the combined half
    panel has no pixel in its leftmost column and its centroid is to the right
    of center, it is flipped horizontally.

    Colors 1-4 are treated as generic foreground and become color 2 in the
    output.  Colors 5-9 are preserved as special colors.
    """
    grid = np.asarray(grid, dtype=int)

    if grid.ndim != 2:
        return grid

    H, W = grid.shape
    if H < 4 or W < 4:
        return np.zeros_like(grid, dtype=int)

    max_val = int(grid.max()) if grid.size else 0
    max_color = max(10, max_val + 1)

    # ------------------------------------------------------------------
    # Candidate frame detection
    # ------------------------------------------------------------------
    def generate_candidates(border_thresh: float, zero_thresh: float):
        candidates = []
        max_L = min(H, W, 12)

        for L in range(4, max_L + 1):
            perimeter_len = 4 * L - 4

            for r in range(H - L + 1):
                for c in range(W - L + 1):
                    top = grid[r, c:c + L]
                    bottom = grid[r + L - 1, c:c + L]
                    left = grid[r + 1:r + L - 1, c]
                    right = grid[r + 1:r + L - 1, c + L - 1]

                    counts = np.zeros(max_color, dtype=int)
                    for arr in (top, bottom, left, right):
                        if arr.size:
                            bc = np.bincount(arr, minlength=max_color)
                            counts += bc[:max_color]

                    if counts.sum() == 0:
                        continue

                    border_color = int(np.argmax(counts))
                    if border_color == 0:
                        continue

                    border_score = counts[border_color] / perimeter_len
                    if border_score < border_thresh:
                        continue

                    interior = grid[r + 1:r + L - 1, c + 1:c + L - 1].copy()
                    interior[interior == border_color] = 0

                    area = interior.size
                    if area == 0:
                        continue

                    zero_frac = float(np.count_nonzero(interior == 0)) / area
                    if zero_frac < zero_thresh:
                        continue

                    if L in (6, 7, 8):
                        l_bonus = 0.10
                    elif L in (5, 9):
                        l_bonus = 0.00
                    else:
                        l_bonus = -0.10

                    candidates.append({
                        "r": r,
                        "c": c,
                        "L": L,
                        "color": border_color,
                        "border_score": float(border_score),
                        "zero_frac": float(zero_frac),
                        "rank": float(border_score + 0.25 * zero_frac + l_bonus),
                    })

        return candidates

    # ------------------------------------------------------------------
    # Choose the best 2x2 set of four frames
    # ------------------------------------------------------------------
    def select_best_frames(candidates):
        best = None
        best_score = -1e9

        by_L = {}
        for cand in candidates:
            by_L.setdefault(cand["L"], []).append(cand)

        for L, lst in by_L.items():
            if len(lst) < 4:
                continue

            lst_sorted = sorted(lst, key=lambda x: x["rank"], reverse=True)

            # Keep a manageable pool, but do not discard one candidate per
            # border color entirely, because the true partial frame may have
            # a slightly lower rank.
            pool = lst_sorted[:50]
            by_color = {}
            for cand in lst_sorted:
                by_color.setdefault(cand["color"], []).append(cand)

            extra = []
            for color, clist in by_color.items():
                extra.extend(clist[:10])

            seen = set()
            pool_unique = []
            for cand in sorted(pool + extra, key=lambda x: x["rank"], reverse=True):
                key = (cand["r"], cand["c"])
                if key not in seen:
                    seen.add(key)
                    pool_unique.append(cand)
                if len(pool_unique) >= 60:
                    break

            pool = pool_unique
            if len(pool) < 4:
                continue

            for combo in combinations(pool, 4):
                centers = [
                    (cand["r"] + L / 2.0, cand["c"] + L / 2.0)
                    for cand in combo
                ]

                rows = sorted(p[0] for p in centers)
                cols = sorted(p[1] for p in centers)

                row_gap = rows[2] - rows[1]
                col_gap = cols[2] - cols[1]

                # The two row clusters and two column clusters must be
                # sufficiently separated, and the chosen split must be the
                # widest gap.
                if row_gap < L or col_gap < L:
                    continue
                if row_gap < rows[1] - rows[0] - 1e-9:
                    continue
                if row_gap < rows[3] - rows[2] - 1e-9:
                    continue
                if col_gap < cols[1] - cols[0] - 1e-9:
                    continue
                if col_gap < cols[3] - cols[2] - 1e-9:
                    continue

                row_mid = (rows[1] + rows[2]) / 2.0
                col_mid = (cols[1] + cols[2]) / 2.0

                quadrant_count = {"tl": 0, "tr": 0, "bl": 0, "br": 0}
                panels = {}

                for cand, (pr, pc) in zip(combo, centers):
                    key = ("t" if pr < row_mid else "b") + (
                        "l" if pc < col_mid else "r"
                    )
                    quadrant_count[key] += 1
                    panels[key] = cand

                if any(v != 1 for v in quadrant_count.values()):
                    continue

                score = sum(cand["border_score"] for cand in combo)
                score += 0.15 * sum(cand["zero_frac"] for cand in combo)

                if L in (6, 7, 8):
                    score += 1.0
                elif L in (5, 9):
                    score += 0.2
                else:
                    score -= 0.5

                score += 0.005 * L
                score += 0.05 * min(cand["border_score"] for cand in combo)

                if len(set(cand["color"] for cand in combo)) == 1:
                    score += 0.05

                if score > best_score:
                    best_score = score
                    best = {
                        "L": L,
                        "panels": panels,
                        "score": score,
                    }

        return best

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------
    def extract_interior(cand):
        L = cand["L"]
        r = cand["r"]
        c = cand["c"]
        interior = grid[r + 1:r + L - 1, c + 1:c + L - 1].copy()
        interior[interior == cand["color"]] = 0
        return interior

    def panel_to_masks(p):
        generic = ((p > 0) & (p <= 4)).astype(np.uint8)
        special = {}

        if p.size:
            max_special = max(9, int(p.max()))
        else:
            max_special = 9

        for v in range(5, max_special + 1):
            m = (p == v).astype(np.uint8)
            if m.any():
                special[v] = m

        return generic, special

    def xor_masks(a, b):
        g = a[0] ^ b[0]
        special = {}

        for v in set(a[1].keys()) | set(b[1].keys()):
            av = a[1].get(v, np.zeros_like(g, dtype=np.uint8))
            bv = b[1].get(v, np.zeros_like(g, dtype=np.uint8))
            x = av ^ bv
            if x.any():
                special[v] = x.astype(np.uint8)

        return g.astype(np.uint8), special

    def total_mask(m):
        t = m[0].copy().astype(np.uint8)
        for arr in m[1].values():
            t |= arr
        return t

    def maybe_flip_horizontal(m):
        t = total_mask(m)
        if not t.any():
            return m

        if t[:, 0].any():
            return m

        _, col_idx = np.where(t)
        centroid = float(col_idx.mean())
        width = m[0].shape[1]

        if centroid > width / 2.0:
            g = m[0][:, ::-1].copy()
            special = {v: arr[:, ::-1].copy() for v, arr in m[1].items()}
            return g, special

        return m

    def output_from_masks(generic, special, shape):
        out = np.zeros(shape, dtype=int)
        out[generic.astype(bool)] = 2

        for v in sorted(special.keys()):
            out[special[v].astype(bool)] = v

        return out

    def output_from_single_interior(interior):
        generic = ((interior > 0) & (interior <= 4))
        out = np.zeros(interior.shape, dtype=int)
        out[generic] = 2

        if interior.size:
            max_special = max(9, int(interior.max()))
            for v in range(5, max_special + 1):
                out[interior == v] = v

        return out

    # ------------------------------------------------------------------
    # Main transformation
    # ------------------------------------------------------------------
    best = select_best_frames(generate_candidates(0.85, 0.10))

    if best is None:
        best = select_best_frames(generate_candidates(0.75, 0.00))

    # Fallback: if no valid 2x2 set is found, output the best single frame.
    if best is None:
        cands = generate_candidates(0.60, 0.00)
        if cands:
            cand = max(cands, key=lambda x: x["rank"])
            return output_from_single_interior(extract_interior(cand))
        return np.zeros((0, 0), dtype=int)

    L = best["L"]
    K = L - 2
    panels = best["panels"]

    tl = panel_to_masks(extract_interior(panels["tl"]))
    tr = panel_to_masks(extract_interior(panels["tr"]))
    bl = panel_to_masks(extract_interior(panels["bl"]))
    br = panel_to_masks(extract_interior(panels["br"]))

    top = xor_masks(tl, tr)
    bottom = xor_masks(bl, br)

    top = maybe_flip_horizontal(top)
    bottom = maybe_flip_horizontal(bottom)

    final_generic, final_special = xor_masks(top, bottom)

    return output_from_masks(final_generic, final_special, (K, K))