import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Move inner colored marks outside their surrounding frame.

    - Non-zero pixels are grouped into 8-connected objects.
    - In a multi-color object, the color with the larger bounding box
      is treated as the frame; other colors are inner marks.
    - Inner marks are cleared and replaced by a line of their color
      just outside the frame's bounding box.
    - The line length is the number of rows in the inner mark's bbox.
    - The side is chosen by the frame side with the fewest frame pixels,
      then by available clear space, then by distance from the inner mark.
    """
    grid = np.array(grid, dtype=int, copy=True)

    if grid.ndim != 2:
        return grid

    H, W = grid.shape
    if H == 0 or W == 0:
        return grid

    dirs8 = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    # Find 8-connected non-zero components.
    labels = np.zeros((H, W), dtype=int)
    components = []

    for i in range(H):
        for j in range(W):
            if grid[i, j] != 0 and labels[i, j] == 0:
                label = len(components) + 1
                labels[i, j] = label
                stack = [(i, j)]
                coords = []

                while stack:
                    r, c = stack.pop()
                    coords.append((r, c))

                    for dr, dc in dirs8:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if labels[nr, nc] == 0 and grid[nr, nc] != 0:
                                labels[nr, nc] = label
                                stack.append((nr, nc))

                components.append(coords)

    # First pass: determine which pixels will be cleared.
    clearable = np.zeros((H, W), dtype=bool)
    objects = []

    for coords in components:
        color_counts = {}
        for r, c in coords:
            col = int(grid[r, c])
            color_counts[col] = color_counts.get(col, 0) + 1

        if len(color_counts) < 2:
            continue

        # Choose frame color: larger bbox area, then larger count.
        best_col = None
        best_key = None
        for col in color_counts:
            rs = [r for r, c in coords if int(grid[r, c]) == col]
            cs = [c for r, c in coords if int(grid[r, c]) == col]
            area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
            key = (area, color_counts[col], col)

            if best_key is None or key > best_key:
                best_key = key
                best_col = col

        frame_col = best_col
        inner_cols = [col for col in color_counts if col != frame_col]

        for r, c in coords:
            if int(grid[r, c]) != frame_col:
                clearable[r, c] = True

        objects.append((coords, frame_col, inner_cols))

    clear_cells = set()
    place_ops = []
    side_order = {"top": 0, "left": 1, "right": 2, "bottom": 3}

    def is_clear(r, c):
        return (
            0 <= r < H and 0 <= c < W and
            (grid[r, c] == 0 or clearable[r, c])
        )

    # Second pass: choose placement side and line cells.
    for coords, frame_col, inner_cols in objects:
        frame = [(r, c) for r, c in coords if int(grid[r, c]) == frame_col]
        if not frame:
            continue

        rmin = min(r for r, c in frame)
        rmax = max(r for r, c in frame)
        cmin = min(c for r, c in frame)
        cmax = max(c for r, c in frame)

        sides = {"top": [], "bottom": [], "left": [], "right": []}
        for r, c in frame:
            if r == rmin:
                sides["top"].append((r, c))
            if r == rmax:
                sides["bottom"].append((r, c))
            if c == cmin:
                sides["left"].append((r, c))
            if c == cmax:
                sides["right"].append((r, c))

        for col in inner_cols:
            inner = [(r, c) for r, c in coords if int(grid[r, c]) == col]
            if not inner:
                continue

            irmin = min(r for r, c in inner)
            irmax = max(r for r, c in inner)
            icmin = min(c for r, c in inner)
            icmax = max(c for r, c in inner)

            line_len = irmax - irmin + 1

            center_r = int(np.floor(sum(r for r, c in inner) / len(inner) + 0.5))
            center_c = int(np.floor(sum(c for r, c in inner) / len(inner) + 0.5))

            candidates = []

            for side in ("top", "bottom", "left", "right"):
                side_count = len(sides[side])

                if side in ("top", "bottom"):
                    if len(sides[side]) == 1:
                        coord = sides[side][0][1]
                    else:
                        coord = center_c
                        coord = max(cmin, min(cmax, coord))

                    if side == "top":
                        rr = rmin - 1
                        cc = coord
                        clear_space = 0
                        while is_clear(rr, cc):
                            clear_space += 1
                            rr -= 1

                        seg = []
                        for k in range(line_len):
                            rr = rmin - k - 1
                            if is_clear(rr, coord):
                                seg.append((rr, coord))
                            else:
                                break

                    else:  # bottom
                        rr = rmax + 1
                        cc = coord
                        clear_space = 0
                        while is_clear(rr, cc):
                            clear_space += 1
                            rr += 1

                        seg = []
                        for k in range(line_len):
                            rr = rmax + k + 1
                            if is_clear(rr, coord):
                                seg.append((rr, coord))
                            else:
                                break

                else:  # left or right
                    if len(sides[side]) == 1:
                        coord = sides[side][0][0]
                    else:
                        coord = center_r
                        coord = max(rmin, min(rmax, coord))

                    if side == "left":
                        rr = coord
                        cc = cmin - 1
                        clear_space = 0
                        while is_clear(rr, cc):
                            clear_space += 1
                            cc -= 1

                        seg = []
                        for k in range(line_len):
                            cc = cmin - k - 1
                            if is_clear(coord, cc):
                                seg.append((coord, cc))
                            else:
                                break

                    else:  # right
                        rr = coord
                        cc = cmax + 1
                        clear_space = 0
                        while is_clear(rr, cc):
                            clear_space += 1
                            cc += 1

                        seg = []
                        for k in range(line_len):
                            cc = cmax + k + 1
                            if is_clear(coord, cc):
                                seg.append((coord, cc))
                            else:
                                break

                if side == "top":
                    dist = max(0, irmin - rmin)
                elif side == "bottom":
                    dist = max(0, rmax - irmax)
                elif side == "left":
                    dist = max(0, icmin - cmin)
                else:  # right
                    dist = max(0, cmax - icmax)

                feasible = (len(seg) == line_len)
                score = (side_count, -clear_space, -dist, side_order[side])
                candidates.append((score, seg, feasible, clear_space))

            feasible_candidates = [cand for cand in candidates if cand[2]]

            if feasible_candidates:
                chosen_score, seg, _, _ = min(
                    feasible_candidates,
                    key=lambda x: x[0]
                )
            else:
                # Fallback if no side fully fits: place as much as possible.
                chosen_score, seg, _, _ = min(
                    candidates,
                    key=lambda x: (x[0][0], -len(x[1]), -x[3], x[0][2], x[0][3])
                )

            clear_cells.update(inner)
            for r, c in seg:
                place_ops.append((r, c, col))

    # Apply all changes.
    for r, c in clear_cells:
        grid[r, c] = 0

    for r, c, col in place_ops:
        if 0 <= r < H and 0 <= c < W:
            grid[r, c] = col

    return grid