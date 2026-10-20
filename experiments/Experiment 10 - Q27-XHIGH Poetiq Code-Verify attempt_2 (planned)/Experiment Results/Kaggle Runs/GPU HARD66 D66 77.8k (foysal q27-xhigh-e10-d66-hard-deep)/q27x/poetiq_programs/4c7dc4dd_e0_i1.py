import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the large ARC grid by locating the relevant square framed
    subgrid and decoding its small interior pattern.

    The rule inferred from the examples:
    1. Find square, solid-bordered frames.
    2. Among them, choose the frame whose interior contains the most
       'message' colors:
         - color 2 is the strongest marker,
         - then color 6,
         - then color 8.
       If several candidates remain, prefer the one with more non-background
       cells and a reasonable small interior size.
    3. Decode the chosen interior:
       - If it contains a 6, treat that 6 as an anchor and complete the
         straight row/column segments to other non-background cells.
       - Otherwise, output the negative of the interior: the dominant
         background color becomes 2, and all non-background colors become 0.
    """
    grid = np.asarray(grid)
    if grid.size == 0:
        return np.zeros_like(grid, dtype=int)

    H, W = grid.shape

    # ------------------------------------------------------------------
    # 1. Detect square solid-bordered frames.
    # ------------------------------------------------------------------
    frames = []
    min_size = 5
    max_size = min(H, W)

    for s in range(min_size, max_size + 1):
        for r in range(0, H - s + 1):
            for c in range(0, W - s + 1):
                color = grid[r, c]
                if color == 0:
                    continue

                ok = True

                # top and bottom rows
                if not np.all(grid[r, c:c + s] == color):
                    ok = False
                elif not np.all(grid[r + s - 1, c:c + s] == color):
                    ok = False

                # left and right columns
                if ok and not np.all(grid[r:r + s, c] == color):
                    ok = False
                elif ok and not np.all(grid[r:r + s, c + s - 1] == color):
                    ok = False

                if ok:
                    interior = grid[r + 1:r + s - 1, c + 1:c + s - 1].copy()
                    frames.append({
                        "r": r,
                        "c": c,
                        "s": s,
                        "border": int(color),
                        "interior": interior
                    })

    if not frames:
        return np.zeros_like(grid, dtype=int)

    # ------------------------------------------------------------------
    # 2. Remove small frames nested inside a larger frame of the same color.
    #    This avoids false squares formed inside a large same-color region.
    # ------------------------------------------------------------------
    filtered = []
    for f in frames:
        contained = False
        for g in frames:
            if f is g or g["border"] != f["border"]:
                continue
            if g["s"] <= f["s"]:
                continue
            # f is strictly inside g's interior
            if (f["r"] > g["r"] + 1 and
                f["c"] > g["c"] + 1 and
                f["r"] + f["s"] - 1 < g["r"] + g["s"] - 2 and
                f["c"] + f["s"] - 1 < g["c"] + g["s"] - 2):
                contained = True
                break
        if not contained:
            filtered.append(f)

    frames = filtered

    # ------------------------------------------------------------------
    # 3. Score frames and choose the most likely message frame.
    # ------------------------------------------------------------------
    def most_common_color(arr: np.ndarray) -> int:
        vals, counts = np.unique(arr, return_counts=True)
        return int(vals[np.argmax(counts)])

    def frame_score(f: dict) -> float:
        interior = f["interior"]
        if interior.size == 0:
            return -10000

        bg = most_common_color(interior)
        mask = interior != bg
        non_bg_count = int(np.sum(mask))

        if non_bg_count == 0:
            return -10000

        colors = set(interior[mask].tolist())
        size = interior.shape[0]

        score = 0.0

        # Strong markers, in priority order.
        if 2 in colors:
            score += 1000
        if 6 in colors:
            score += 900
        if 8 in colors:
            score += 600

        # Prefer richer small patterns, but not absurdly large interiors.
        score += 10 * non_bg_count
        score += 5 * len(colors)

        if size in (4, 5, 6):
            score += 30
        elif size <= 8:
            score += 15
        else:
            score -= 40

        return score

    best_frame = max(frames, key=frame_score)
    interior = best_frame["interior"].copy()

    # ------------------------------------------------------------------
    # 4. Decode the chosen interior.
    # ------------------------------------------------------------------
    if interior.size == 0:
        return np.zeros_like(interior, dtype=int)

    bg = most_common_color(interior)
    non_bg = interior != bg
    special_colors = set(interior[non_bg].tolist())

    # Case A: skeleton / anchor case, triggered by color 6.
    if 6 in special_colors:
        out = np.zeros_like(interior, dtype=int)

        anchors = np.argwhere(interior == 6)
        if len(anchors) == 0:
            # Should not happen, but fallback to inversion.
            return (interior == bg).astype(int) * 2

        # Choose the best anchor: preferably a corner of the non-background bbox.
        rmin, rmax, cmin, cmax = (np.argwhere(non_bg).min(axis=0).tolist() +
                                  np.argwhere(non_bg).max(axis=0).tolist())
        rmin, cmin = rmin, cmin
        rmax, cmax = rmax, cmax

        def anchor_score(pos):
            r, c = pos
            sc = 0.0

            # Corner of the bounding box is best.
            if (r == rmin or r == rmax) and (c == cmin or c == cmax):
                sc += 100

            # More aligned non-background cells in same row/column.
            row_cells = np.sum(non_bg[r, :])
            col_cells = np.sum(non_bg[:, c])
            sc += row_cells + col_cells

            # Slight preference for top-left.
            sc -= 0.01 * (r + c)

            return sc

        anchor = max(anchors, key=anchor_score)
        ar, ac = anchor

        out[ar, ac] = 6

        # Fill straight segments from the anchor to every non-background cell
        # that lies in the same row or column.
        for r, c in zip(*np.argwhere(non_bg)):
            if r == ar:
                lo = min(ac, c)
                hi = max(ac, c)
                out[ar, lo:hi + 1] = 2
            elif c == ac:
                lo = min(ar, r)
                hi = max(ar, r)
                out[lo:hi + 1, ac] = 2

        # The anchor remains 6.
        out[ar, ac] = 6
        return out

    # Case B: default case, output the inverted binary pattern.
    # Dominant background becomes 2; everything else becomes 0.
    return (interior == bg).astype(int) * 2