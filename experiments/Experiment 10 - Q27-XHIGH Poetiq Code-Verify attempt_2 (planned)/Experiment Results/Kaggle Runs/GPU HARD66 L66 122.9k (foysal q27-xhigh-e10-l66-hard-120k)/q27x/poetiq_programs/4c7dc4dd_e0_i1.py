import numpy as np
import itertools

def transform(grid: np.ndarray) -> np.ndarray:
    """
    The input contains several square framed panels arranged in a 2x2 layout.
    The panels are first detected by their border color.  The interior of each
    panel is extracted, border pixels are cleared, and the four interiors are
    combined by a color-aware XOR:
      - colors 1..4 are treated as one generic "line" layer and output as 2;
      - colors >=5 are treated as special colors and are preserved.

    Before the final XOR, the combined top pair and combined bottom pair are
    normalized: if a half-panel is shifted to the right (no foreground in the
    leftmost column and centroid to the right of center), it is reflected
    horizontally.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2:
        return grid

    H, W = grid.shape
    if H < 3 or W < 3:
        return np.zeros_like(grid)

    # ---------------------------------------------------------------
    # 1. Detect candidate square frames.
    #    The examples use side lengths 6 and 7, but a small range is checked
    #    for robustness.
    # ---------------------------------------------------------------
    candidates = []

    # Prefer the side lengths that occur in the visible examples/challenges.
    for L in (6, 7, 5, 8, 9, 10):
        if L > H or L > W:
            continue

        for r in range(H - L + 1):
            for c in range(W - L + 1):
                # Unique border pixels: top, bottom, and side pixels without corners.
                top = grid[r, c:c + L]
                bottom = grid[r + L - 1, c:c + L]
                left = grid[r + 1:r + L - 1, c]
                right = grid[r + 1:r + L - 1, c + L - 1]

                counts = np.zeros(10, dtype=int)
                for arr in (top, bottom, left, right):
                    if arr.size:
                        # ARC colors are 0..9.  If unexpected larger values occur,
                        # bincount simply expands; we only inspect the first 10 bins.
                        bc = np.bincount(arr, minlength=10)
                        counts[:min(10, len(bc))] += bc[:10]

                if counts.sum() == 0:
                    continue

                border_color = int(np.argmax(counts))
                border_count = int(counts[border_color])
                if border_color == 0:
                    continue

                perimeter = 4 * L - 4
                border_score = border_count / perimeter
                if border_score < 0.60:
                    continue

                interior = grid[r + 1:r + L - 1, c + 1:c + L - 1]
                interior_clear = interior.copy()
                interior_clear[interior_clear == border_color] = 0

                nonempty = bool((interior_clear != 0).any())
                special = bool((interior_clear >= 5).any())

                candidates.append({
                    "score": border_score,
                    "nonempty": nonempty,
                    "special": special,
                    "r": r,
                    "c": c,
                    "L": L,
                    "color": border_color
                })

    if not candidates:
        return np.zeros_like(grid)

    # ---------------------------------------------------------------
    # 2. Choose the best 4 frames forming a 2x2 layout.
    # ---------------------------------------------------------------
    best_combo = None
    best_value = -1.0

    by_L = {}
    for cand in candidates:
        by_L.setdefault(cand["L"], []).append(cand)

    for L, cands in by_L.items():
        # Keep only the strongest candidates for this size.
        cands = sorted(
            cands,
            key=lambda x: (x["score"], x["nonempty"], x["special"]),
            reverse=True
        )[:20]

        for combo in itertools.combinations(cands, 4):
            centers = [(x["r"] + (L - 1) / 2.0, x["c"] + (L - 1) / 2.0) for x in combo]
            rows = sorted(p[0] for p in centers)
            cols = sorted(p[1] for p in centers)

            # Require the two row clusters and two column clusters to be separated.
            if rows[1] - rows[0] < 0.25 * L:
                continue
            if rows[3] - rows[2] < 0.25 * L:
                continue
            if cols[1] - cols[0] < 0.25 * L:
                continue
            if cols[3] - cols[2] < 0.25 * L:
                continue
            if rows[2] - rows[1] < 0.25 * L:
                continue
            if cols[2] - cols[1] < 0.25 * L:
                continue

            row_mid = (rows[1] + rows[2]) / 2.0
            col_mid = (cols[1] + cols[2]) / 2.0

            quad_counts = {"tl": 0, "tr": 0, "bl": 0, "br": 0}
            valid = True
            for pr, pc in centers:
                key = ("t" if pr < row_mid else "b") + ("l" if pc < col_mid else "r")
                quad_counts[key] += 1
                if quad_counts[key] > 1:
                    valid = False
                    break

            if not valid:
                continue

            value = sum(x["score"] for x in combo)
            value += 0.01 * sum(x["nonempty"] for x in combo)
            value += 0.001 * sum(x["special"] for x in combo)

            # Prefer the sizes seen in the examples.
            if L in (6, 7):
                value += 1.0

            # Prefer consistent border color when available.
            if len(set(x["color"] for x in combo)) == 1:
                value += 0.5

            if value > best_value:
                best_value = value
                best_combo = (L, combo)

    if best_combo is None:
        # Fallback: use the single best frame if no 2x2 set can be found.
        cand = max(candidates, key=lambda x: (x["score"], x["nonempty"], x["special"]))
        L = cand["L"]
        r, c = cand["r"], cand["c"]
        interior = grid[r + 1:r + L - 1, c + 1:c + L - 1].copy()
        interior[interior == cand["color"]] = 0

        out = np.zeros_like(interior, dtype=int)
        out[(interior > 0) & (interior <= 4)] = 2
        for v in range(5, 10):
            out[interior == v] = v
        return out

    L, combo = best_combo
    K = L - 2

    # Determine the quadrant midpoint for the selected four frames.
    centers = [(x["r"] + (L - 1) / 2.0, x["c"] + (L - 1) / 2.0) for x in combo]
    rows = sorted(p[0] for p in centers)
    cols = sorted(p[1] for p in centers)
    row_mid = (rows[1] + rows[2]) / 2.0
    col_mid = (cols[1] + cols[2]) / 2.0

    panels = {}
    for x, (pr, pc) in zip(combo, centers):
        r, c = x["r"], x["c"]
        key = ("t" if pr < row_mid else "b") + ("l" if pc < col_mid else "r")
        interior = grid[r + 1:r + L - 1, c + 1:c + L - 1].copy()
        interior[interior == x["color"]] = 0
        panels[key] = interior

    # ---------------------------------------------------------------
    # 3. Color-aware XOR representation.
    #    generic: colors 1..4
    #    special: each color >=5 separately
    # ---------------------------------------------------------------
    def panel_to_masks(p):
        generic = ((p > 0) & (p <= 4)).astype(np.uint8)
        special = {}
        for v in range(5, 10):
            m = (p == v).astype(np.uint8)
            if m.any():
                special[v] = m
        return generic, special

    def xor_masks(a, b):
        g = (a[0] ^ b[0]).astype(np.uint8)
        special = {}
        for v in set(a[1].keys()) | set(b[1].keys()):
            av = a[1].get(v, np.zeros_like(g))
            bv = b[1].get(v, np.zeros_like(g))
            special[v] = (av ^ bv).astype(np.uint8)
        return g, special

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
        if centroid <= m[0].shape[1] / 2.0:
            return m

        g = m[0][:, ::-1].copy()
        special = {v: arr[:, ::-1].copy() for v, arr in m[1].items()}
        return g, special

    tl = panel_to_masks(panels["tl"])
    tr = panel_to_masks(panels["tr"])
    bl = panel_to_masks(panels["bl"])
    br = panel_to_masks(panels["br"])

    top = xor_masks(tl, tr)
    bottom = xor_masks(bl, br)

    top = maybe_flip_horizontal(top)
    bottom = maybe_flip_horizontal(bottom)

    final_generic, final_special = xor_masks(top, bottom)

    # ---------------------------------------------------------------
    # 4. Produce the output grid.
    # ---------------------------------------------------------------
    out = np.zeros((K, K), dtype=int)
    out[final_generic.astype(bool)] = 2

    # If multiple special colors overlap, the larger color wins.
    for v in sorted(final_special.keys()):
        out[final_special[v].astype(bool)] = v

    return out