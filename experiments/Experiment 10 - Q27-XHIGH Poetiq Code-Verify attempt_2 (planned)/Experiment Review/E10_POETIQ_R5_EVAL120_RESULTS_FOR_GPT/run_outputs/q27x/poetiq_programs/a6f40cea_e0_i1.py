import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Find the main rectangular window in the grid, then reconstruct other
    rectangular frames whose visible parts are outside the window and render
    the hidden portions inside the window.
    """
    grid = np.array(grid, dtype=int, copy=True)
    H, W = grid.shape
    if H == 0 or W == 0:
        return grid.copy()

    # Background color is the most common border color.
    border = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
    bg = int(np.bincount(border, minlength=16).argmax())

    four_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def get_components(mask):
        """4-connected components as lists of (r, c)."""
        seen = np.zeros((H, W), dtype=bool)
        comps = []
        for i in range(H):
            for j in range(W):
                if mask[i, j] and not seen[i, j]:
                    stack = [(i, j)]
                    seen[i, j] = True
                    coords = []
                    while stack:
                        r, c = stack.pop()
                        coords.append((r, c))
                        for dr, dc in four_dirs:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True
                                stack.append((nr, nc))
                    if coords:
                        comps.append(coords)
        return comps

    def find_window():
        """
        Find the main window:
        - complete 1-cell-thick rectangular frame
        - preferably uniform interior
        - largest among valid candidates
        """
        best = None
        best_score = None

        for color in range(16):
            if color == bg:
                continue
            mask = (grid == color)
            if not mask.any():
                continue

            for coords in get_components(mask):
                if len(coords) < 4:
                    continue

                rmin = min(r for r, _ in coords)
                rmax = max(r for r, _ in coords)
                cmin = min(c for _, c in coords)
                cmax = max(c for _, c in coords)

                h = rmax - rmin + 1
                w = cmax - cmin + 1
                if h < 3 or w < 3:
                    continue

                perim_len = 2 * h + 2 * w - 4
                if len(coords) != perim_len:
                    continue

                ok = True
                for r, c in coords:
                    if not (r == rmin or r == rmax or c == cmin or c == cmax):
                        ok = False
                        break
                if not ok:
                    continue

                interior = grid[rmin + 1:rmax, cmin + 1:cmax]
                unique_interior = len(np.unique(interior)) if interior.size else 0
                area = h * w

                # Prefer uniform interiors, then larger area.
                score = (1 if unique_interior == 1 else 0, area, -unique_interior)
                if best_score is None or score > best_score:
                    best_score = score
                    best = (rmin, rmax, cmin, cmax)

        return best

    window = find_window()
    if window is None:
        return grid.copy()

    r0, r1, c0, c1 = window
    ir0, ir1 = r0 + 1, r1 - 1
    ic0, ic1 = c0 + 1, c1 - 1

    if ir1 < ir0 or ic1 < ic0:
        return np.zeros((0, 0), dtype=int)

    interior = grid[ir0:ir1 + 1, ic0:ic1 + 1]
    vals, counts = np.unique(interior, return_counts=True)
    base = int(vals[np.argmax(counts)])

    out = np.full((ir1 - ir0 + 1, ic1 - ic0 + 1), base, dtype=int)

    # Candidate object pixels:
    # - not background
    # - not the window border
    # - not the hidden base color inside the window interior
    cand = (grid != bg)
    cand[r0, c0:c1 + 1] = False
    cand[r1, c0:c1 + 1] = False
    cand[r0:r1 + 1, c0] = False
    cand[r0:r1 + 1, c1] = False
    cand[ir0:ir1 + 1, ic0:ic1 + 1] = (interior != base)

    groups = []

    for coords in get_components(cand):
        arr = np.array(coords, dtype=int)
        rmin = int(arr[:, 0].min())
        rmax = int(arr[:, 0].max())
        cmin = int(arr[:, 1].min())
        cmax = int(arr[:, 1].max())

        h = rmax - rmin + 1
        w = cmax - cmin + 1

        if h <= 0 or w <= 0:
            continue
        if h == 1 and w == 1:
            continue

        colors = set(int(grid[r, c]) for r, c in arr)
        if len(colors) > 2:
            continue

        # Ignore tiny solid blocks that are usually fills/noise, not frame fragments.
        if (
            len(colors) == 1
            and len(arr) == h * w
            and h * w <= 9
            and not (h == 1 or w == 1)
        ):
            continue

        on_perim = (
            (arr[:, 0] == rmin) |
            (arr[:, 0] == rmax) |
            (arr[:, 1] == cmin) |
            (arr[:, 1] == cmax)
        )
        if not on_perim.all():
            continue

        groups.append({
            "coords": arr,
            "colors": colors,
            "bbox": (rmin, rmax, cmin, cmax),
        })

    def can_merge(a, b):
        """Two fragments can be one rectangle if all pixels lie on the union perimeter."""
        if a is b:
            return False
        colors = a["colors"] | b["colors"]
        if len(colors) > 2:
            return False

        coords = np.vstack((a["coords"], b["coords"]))
        rmin = min(a["bbox"][0], b["bbox"][0])
        rmax = max(a["bbox"][1], b["bbox"][1])
        cmin = min(a["bbox"][2], b["bbox"][2])
        cmax = max(a["bbox"][3], b["bbox"][3])

        h = rmax - rmin + 1
        w = cmax - cmin + 1
        if h < 2 or w < 2:
            return False

        on_perim = (
            (coords[:, 0] == rmin) |
            (coords[:, 0] == rmax) |
            (coords[:, 1] == cmin) |
            (coords[:, 1] == cmax)
        )
        return bool(on_perim.all())

    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            if changed:
                break
            for j in range(i + 1, len(groups)):
                if can_merge(groups[i], groups[j]):
                    groups[i]["coords"] = np.vstack((groups[i]["coords"], groups[j]["coords"]))
                    groups[i]["colors"] |= groups[j]["colors"]

                    rmin = min(groups[i]["bbox"][0], groups[j]["bbox"][0])
                    rmax = max(groups[i]["bbox"][1], groups[j]["bbox"][1])
                    cmin = min(groups[i]["bbox"][2], groups[j]["bbox"][2])
                    cmax = max(groups[i]["bbox"][3], groups[j]["bbox"][3])
                    groups[i]["bbox"] = (rmin, rmax, cmin, cmax)

                    del groups[j]
                    changed = True
                    break

    def other_color(col, cset):
        for x in cset:
            if x != col:
                return int(x)
        return int(col)

    def complete_side(length, fixed_anchors, extra_anchors, cset):
        """
        Complete one side of a rectangle.
        fixed_anchors: direct visible pixels on this side.
        extra_anchors: corner colors inferred from adjacent sides.
        Uses the nearest anchor and alternates color by parity distance.
        """
        anchors = []
        for pos, col in fixed_anchors.items():
            if 0 <= pos < length:
                anchors.append((int(pos), int(col), 0))  # priority 0: visible fixed
        for pos, col in extra_anchors:
            if 0 <= pos < length and int(col) != -1:
                anchors.append((int(pos), int(col), 1))  # priority 1: inferred corner

        arr = np.full(length, -1, dtype=int)
        if not anchors:
            return arr

        fixed_map = {int(pos): int(col) for pos, col in fixed_anchors.items() if 0 <= pos < length}

        for pos in range(length):
            if pos in fixed_map:
                arr[pos] = fixed_map[pos]
                continue

            best_key = None
            best_col = -1
            for pos_a, col_a, pri in anchors:
                dist = abs(pos - pos_a)
                cand_col = col_a if (dist % 2 == 0) else other_color(col_a, cset)
                key = (dist, pri, pos_a)
                if best_key is None or key < best_key:
                    best_key = key
                    best_col = cand_col

            arr[pos] = best_col

        return arr

    def render_solid(g):
        rmin, rmax, cmin, cmax = g["bbox"]
        col = next(iter(g["colors"]))

        ir0_l = max(rmin, ir0)
        ir1_l = min(rmax, ir1)
        ic0_l = max(cmin, ic0)
        ic1_l = min(cmax, ic1)

        if ir1_l < ir0_l or ic1_l < ic0_l:
            return

        for r in range(ir0_l, ir1_l + 1):
            for c in range(ic0_l, ic1_l + 1):
                if r == rmin or r == rmax or c == cmin or c == cmax:
                    out[r - ir0, c - ic0] = col

    def render_alternating(g):
        rmin, rmax, cmin, cmax = g["bbox"]
        h = rmax - rmin + 1
        w = cmax - cmin + 1
        cset = g["colors"]

        fixed = {"top": {}, "bottom": {}, "left": {}, "right": {}}

        for r, c in g["coords"]:
            col = int(grid[r, c])
            if r == rmin:
                fixed["top"][c - cmin] = col
            if r == rmax:
                fixed["bottom"][c - cmin] = col
            if c == cmin:
                fixed["left"][r - rmin] = col
            if c == cmax:
                fixed["right"][r - rmin] = col

        lengths = {"top": w, "bottom": w, "left": h, "right": h}
        sides = ["top", "bottom", "left", "right"]
        arrays = {s: np.full(lengths[s], -1, dtype=int) for s in sides}

        # Iterate so corner colors inferred on one side can anchor adjacent sides.
        for _ in range(8):
            new_arrays = {}

            for s in sides:
                extra = []
                if s == "top":
                    if arrays["left"][0] != -1:
                        extra.append((0, arrays["left"][0]))
                    if arrays["right"][0] != -1:
                        extra.append((w - 1, arrays["right"][0]))
                elif s == "bottom":
                    if arrays["left"][h - 1] != -1:
                        extra.append((0, arrays["left"][h - 1]))
                    if arrays["right"][h - 1] != -1:
                        extra.append((w - 1, arrays["right"][h - 1]))
                elif s == "left":
                    if arrays["top"][0] != -1:
                        extra.append((0, arrays["top"][0]))
                    if arrays["bottom"][0] != -1:
                        extra.append((h - 1, arrays["bottom"][0]))
                elif s == "right":
                    if arrays["top"][w - 1] != -1:
                        extra.append((0, arrays["top"][w - 1]))
                    if arrays["bottom"][w - 1] != -1:
                        extra.append((h - 1, arrays["bottom"][w - 1]))

                new_arrays[s] = complete_side(lengths[s], fixed[s], extra, cset)

            if all(np.array_equal(new_arrays[s], arrays[s]) for s in sides):
                break
            arrays = new_arrays

        top_a, bottom_a = arrays["top"], arrays["bottom"]
        left_a, right_a = arrays["left"], arrays["right"]

        ir0_l = max(rmin, ir0)
        ir1_l = min(rmax, ir1)
        ic0_l = max(cmin, ic0)
        ic1_l = min(cmax, ic1)

        if ir1_l < ir0_l or ic1_l < ic0_l:
            return

        for r in range(ir0_l, ir1_l + 1):
            for c in range(ic0_l, ic1_l + 1):
                if not (r == rmin or r == rmax or c == cmin or c == cmax):
                    continue

                col = -1
                if r == rmin:
                    col = top_a[c - cmin]
                if col == -1 and r == rmax:
                    col = bottom_a[c - cmin]
                if col == -1 and c == cmin:
                    col = left_a[r - rmin]
                if col == -1 and c == cmax:
                    col = right_a[r - rmin]

                if col != -1:
                    out[r - ir0, c - ic0] = int(col)

    for g in groups:
        if len(g["colors"]) == 1:
            render_solid(g)
        elif len(g["colors"]) == 2:
            render_alternating(g)

    # If any non-base pixels are already visible inside the window, preserve them.
    for r in range(ir0, ir1 + 1):
        for c in range(ic0, ic1 + 1):
            if grid[r, c] != base:
                out[r - ir0, c - ic0] = int(grid[r, c])

    return out