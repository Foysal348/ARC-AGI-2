import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a large ARC grid into a small output pattern.

    Rule inferred from the examples:
    1. The large input contains several small square/rectangular frames
       with solid borders.  The relevant frame is the small one whose
       interior contains the most "special" content:
         - color 2 is very important,
         - color 6 is also important,
         - color 8 can also be part of the relevant interior.
       Small frames are strongly preferred over large background frames.

    2. Extract the interior of the chosen frame.

    3. Decode the interior:
       - If it contains color 6, use the 6 as an anchor.
         Fill straight horizontal/vertical rays from that 6 to other
         non-background cells with 2, and keep the anchor cell as 6.
       - Otherwise, output a binary negative:
         the dominant background color becomes 2, all other colors become 0.
    """
    grid = np.array(grid, dtype=int, copy=True)

    if grid.size == 0:
        return np.zeros((0, 0), dtype=int)

    if grid.ndim != 2:
        try:
            grid = grid.reshape((grid.shape[0], -1))
        except Exception:
            return np.zeros((0, 0), dtype=int)

    H, W = grid.shape
    if H < 3 or W < 3:
        return np.zeros_like(grid, dtype=int)

    # ------------------------------------------------------------
    # Precompute horizontal and vertical same-color run lengths.
    # This makes solid-border frame detection fast.
    # ------------------------------------------------------------
    h_run = np.zeros((H, W), dtype=int)
    for r in range(H):
        for c in range(W - 1, -1, -1):
            if c == W - 1 or grid[r, c] != grid[r, c + 1]:
                h_run[r, c] = 1
            else:
                h_run[r, c] = h_run[r, c + 1] + 1

    v_run = np.zeros((H, W), dtype=int)
    for r in range(H - 1, -1, -1):
        for c in range(W):
            if r == H - 1 or grid[r, c] != grid[r + 1, c]:
                v_run[r, c] = 1
            else:
                v_run[r, c] = v_run[r + 1, c] + 1

    # ------------------------------------------------------------
    # Find all solid-bordered frames up to a given maximum size.
    # Frame size is height/width including border.
    # ------------------------------------------------------------
    def find_frames(max_size: int):
        if max_size < 3:
            return []

        frames = []
        max_size = min(max_size, H, W)

        for r in range(H):
            for c in range(W):
                color = grid[r, c]
                if color == 0:
                    continue

                top_run = h_run[r, c]
                if top_run < 3:
                    continue

                max_w = min(max_size, W - c, top_run)
                max_h = min(max_size, H - r, v_run[r, c])

                for h in range(3, max_h + 1):
                    bottom_r = r + h - 1
                    bottom_run = h_run[bottom_r, c]
                    if bottom_run < 3:
                        continue

                    max_w_for_h = min(max_w, bottom_run)
                    for w in range(3, max_w_for_h + 1):
                        right_c = c + w - 1

                        # Right border must be solid.
                        if v_run[r, right_c] < h:
                            continue

                        # Extra safety: bottom-right corner matches.
                        if grid[bottom_r, right_c] != color:
                            continue

                        interior = grid[r + 1:bottom_r, c + 1:c + w].copy()
                        if interior.size == 0:
                            continue

                        frames.append({
                            "r": r,
                            "c": c,
                            "h": h,
                            "w": w,
                            "border": int(color),
                            "interior": interior,
                        })

        return frames

    # ------------------------------------------------------------
    # Analyze frames and compute useful attributes.
    # ------------------------------------------------------------
    def analyze_frames(frames):
        for f in frames:
            interior = f["interior"]

            vals, counts = np.unique(interior, return_counts=True)
            bg = int(vals[np.argmax(counts)])
            non_bg = interior != bg

            f["bg"] = bg
            f["non_bg_count"] = int(np.sum(non_bg))
            f["colors"] = set(int(x) for x in interior.flat.tolist())
            f["max_dim"] = max(interior.shape)
            f["area"] = int(interior.size)
            f["has2"] = (2 in f["colors"])
            f["has6"] = (6 in f["colors"])

        return frames

    # ------------------------------------------------------------
    # Collect candidates.  Prefer small frames first, then expand
    # only if no useful small frame is found.
    # ------------------------------------------------------------
    candidates = []
    analyzed = []

    for ms in (12, 16, 20, min(H, W)):
        frames = analyze_frames(find_frames(ms))
        analyzed = frames
        candidates = [
            f for f in frames
            if f["non_bg_count"] > 0 or f["has2"] or f["has6"]
        ]
        if candidates:
            break

    if not candidates:
        # Last resort: any non-empty frame with non-background content.
        candidates = [f for f in analyzed if f["non_bg_count"] > 0]

    if not candidates:
        return np.zeros_like(grid, dtype=int)

    # ------------------------------------------------------------
    # Score candidate frames.
    # ------------------------------------------------------------
    color_counts = {}
    for f in candidates:
        for col in f["colors"]:
            if col != 0:
                color_counts[col] = color_counts.get(col, 0) + 1

    def frame_score(f):
        sc = 0
        md = f["max_dim"]

        # Prefer small interiors.
        if md <= 4:
            sc += 1200
        elif md <= 6:
            sc += 1000
        elif md <= 8:
            sc += 800
        elif md <= 10:
            sc += 500
        elif md <= 12:
            sc += 100
        elif md > 16:
            sc -= 5000

        # Non-empty interiors are more likely to be the target.
        if f["non_bg_count"] == 0:
            sc -= 2000
        else:
            sc += min(50, f["non_bg_count"]) * 2

        colors = f["colors"]

        # Content priorities inferred from examples.
        if f["has2"]:
            sc += 1000
        if f["has6"]:
            sc += 900
        if f["has2"] and f["has6"]:
            sc += 500
        if 8 in colors:
            sc += 300

        # Prefer unique interior colors among candidate frames.
        for col in colors:
            if col != 0 and color_counts.get(col, 0) == 1:
                sc += 100
                if col == 2:
                    sc += 200
                elif col == 6:
                    sc += 100
                elif col == 8:
                    sc += 50

        # Slight preference for clean borders not repeated inside.
        if f["border"] not in colors:
            sc += 50

        return sc

    best = max(
        candidates,
        key=lambda f: (
            frame_score(f),
            f["non_bg_count"],
            -f["max_dim"],
            -f["r"],
            -f["c"],
        ),
    )

    interior = best["interior"]

    # ------------------------------------------------------------
    # Decode selected interior.
    # ------------------------------------------------------------
    def decode_simple(interior: np.ndarray) -> np.ndarray:
        """No 6: background -> 2, everything else -> 0."""
        vals, counts = np.unique(interior, return_counts=True)
        bg = int(vals[np.argmax(counts)])
        return (interior == bg).astype(int) * 2

    def decode_with_six(interior: np.ndarray) -> np.ndarray:
        """
        Contains 6: choose a 6 anchor, fill rays to non-background
        cells with 2, and keep anchor as 6.
        """
        h, w = interior.shape

        vals, counts = np.unique(interior, return_counts=True)
        bg = int(vals[np.argmax(counts)])
        non_bg = interior != bg

        sixes = np.argwhere(interior == 6)
        if len(sixes) == 0:
            return decode_simple(interior)

        # If there are no non-background cells besides background itself,
        # just place one 6.
        if not np.any(non_bg):
            ar, ac = sixes[0]
            out = np.zeros((h, w), dtype=int)
            out[ar, ac] = 6
            return out

        non_bg_coords = np.argwhere(non_bg)
        rmin = int(non_bg_coords[:, 0].min())
        rmax = int(non_bg_coords[:, 0].max())
        cmin = int(non_bg_coords[:, 1].min())
        cmax = int(non_bg_coords[:, 1].max())

        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def direction_has_target(r, c, dr, dc):
            rr, cc = r + dr, c + dc
            while 0 <= rr < h and 0 <= cc < w:
                if non_bg[rr, cc]:
                    return True
                rr += dr
                cc += dc
            return False

        def farthest_target(r, c, dr, dc):
            rr, cc = r + dr, c + dc
            last = None
            while 0 <= rr < h and 0 <= cc < w:
                if non_bg[rr, cc]:
                    last = (rr, cc)
                rr += dr
                cc += dc
            return last

        def anchor_score(pos):
            r, c = pos
            sc = 0.0

            dir_count = 0
            for dr, dc in dirs:
                if direction_has_target(r, c, dr, dc):
                    dir_count += 1

            # A corner with two perpendicular rays is strongly preferred.
            sc += dir_count * 1000

            # More aligned non-background cells is better.
            sc += int(np.sum(non_bg[r, :]))
            sc += int(np.sum(non_bg[:, c]))

            # Prefer anchors at corners of the non-background bbox.
            if (r == rmin or r == rmax) and (c == cmin or c == cmax):
                sc += 250

            # Slight deterministic preference for top-left.
            sc -= 0.01 * (r + c)

            return sc

        ar, ac = max(sixes, key=lambda p: anchor_score(p))

        out = np.zeros((h, w), dtype=int)

        for dr, dc in dirs:
            far = farthest_target(ar, ac, dr, dc)
            if far is None:
                continue

            r0, c0 = ar + dr, ac + dc
            r1, c1 = far

            if dr == 0:
                out[ar, min(c0, c1):max(c0, c1) + 1] = 2
            else:
                out[min(r0, r1):max(r0, r1) + 1, ac] = 2

        out[ar, ac] = 6
        return out

    if np.any(interior == 6):
        return decode_with_six(interior)
    else:
        return decode_simple(interior)