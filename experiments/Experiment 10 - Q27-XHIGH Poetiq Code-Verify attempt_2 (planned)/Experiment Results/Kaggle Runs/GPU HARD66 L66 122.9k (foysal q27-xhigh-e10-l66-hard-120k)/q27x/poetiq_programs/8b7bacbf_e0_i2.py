import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill the largest compatible enclosed background regions with the singleton
    fill color.

    A hole is a 4-connected background component. Its boundary may contain one
    frame color, and optionally the key color adjacent to the fill marker.
    Border-bounded holes are accepted only when their border segment is bounded
    by allowed colors. For each frame color, the largest compatible holes are
    filled.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2:
        return grid.copy()

    H, W = grid.shape
    out = grid.copy()
    if H == 0 or W == 0:
        return out

    # ------------------------------------------------------------------
    # Background color
    # ------------------------------------------------------------------
    vals, counts = np.unique(grid, return_counts=True)
    counts = counts.astype(int)
    max_count = int(counts.max())

    if int(np.sum(counts == max_count)) > 1:
        if 0 in vals:
            bg = 0
        else:
            bg = int(vals[np.argmin(vals)])
    else:
        bg = int(vals[np.argmax(counts)])

    color_count = {int(c): int(n) for c, n in zip(vals, counts)}

    # ------------------------------------------------------------------
    # Fill color: unique non-background pixel.
    # If multiple, prefer border, then bottom, then right, then position.
    # ------------------------------------------------------------------
    singletons = [int(c) for c in vals if int(c) != bg and color_count[int(c)] == 1]

    if singletons:
        singleton_pos = {}
        for c in singletons:
            idx = np.argwhere(grid == c)
            if idx.size:
                singleton_pos[c] = (int(idx[0, 0]), int(idx[0, 1]))

        def singleton_score(c):
            r, cc = singleton_pos[c]
            border = (r == 0 or r == H - 1 or cc == 0 or cc == W - 1)
            return (border, r == H - 1, cc == W - 1, r, cc, c)

        fill = max(singletons, key=singleton_score)
    else:
        nonbg = [int(c) for c in vals if int(c) != bg]
        if not nonbg:
            return out
        fill = min(nonbg, key=lambda c: (color_count[c], c))

    # ------------------------------------------------------------------
    # Key-color candidates: colors adjacent to the fill marker.
    # ------------------------------------------------------------------
    fill_positions = np.argwhere(grid == fill)
    dirs4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    dirs8 = dirs4 + [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    key_candidates = set()

    for r, c in fill_positions:
        for dr, dc in dirs4:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < H and 0 <= nc < W:
                v = int(grid[nr, nc])
                if v != bg and v != fill:
                    key_candidates.add(v)

    if not key_candidates:
        for r, c in fill_positions:
            for dr, dc in dirs8:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < H and 0 <= nc < W:
                    v = int(grid[nr, nc])
                    if v != bg and v != fill:
                        key_candidates.add(v)

    if not key_candidates:
        nonbg = [int(c) for c in vals if int(c) != bg and int(c) != fill]
        if not nonbg:
            return out
        key_candidates = {max(nonbg, key=lambda c: (color_count[c], -c))}

    key_candidates = sorted(key_candidates)

    # ------------------------------------------------------------------
    # Helpers for edge-bounded holes.
    # ------------------------------------------------------------------
    def is_contiguous(values):
        values = sorted(values)
        for i in range(len(values) - 1):
            if values[i + 1] != values[i] + 1:
                return False
        return True

    def side_ok(values, side, allowed_colors):
        if not values:
            return False
        values = sorted(values)
        if not is_contiguous(values):
            return False

        a, b = values[0], values[-1]

        if side == "top" or side == "bottom":
            row = 0 if side == "top" else H - 1
            if a > 0 and int(grid[row, a - 1]) not in allowed_colors:
                return False
            if b < W - 1 and int(grid[row, b + 1]) not in allowed_colors:
                return False
        else:
            col = 0 if side == "left" else W - 1
            if a > 0 and int(grid[a - 1, col]) not in allowed_colors:
                return False
            if b < H - 1 and int(grid[b + 1, col]) not in allowed_colors:
                return False

        return True

    def edge_ok(cells, allowed_colors):
        top_cols = []
        bottom_cols = []
        left_rows = []
        right_rows = []

        for r, c in cells:
            if r == 0:
                top_cols.append(c)
            elif r == H - 1:
                bottom_cols.append(c)
            if c == 0:
                left_rows.append(r)
            elif c == W - 1:
                right_rows.append(r)

        sides = []
        if top_cols:
            sides.append("top")
        if bottom_cols:
            sides.append("bottom")
        if left_rows:
            sides.append("left")
        if right_rows:
            sides.append("right")

        n_sides = len(sides)
        if n_sides == 0:
            return True
        if n_sides > 2:
            return False

        if n_sides == 1:
            side = sides[0]
            if side == "top":
                return side_ok(top_cols, side, allowed_colors)
            if side == "bottom":
                return side_ok(bottom_cols, side, allowed_colors)
            if side == "left":
                return side_ok(left_rows, side, allowed_colors)
            return side_ok(right_rows, side, allowed_colors)

        # Two adjacent sides are allowed only as a bounded corner room.
        if top_cols and left_rows and 0 in top_cols and 0 in left_rows:
            return side_ok(top_cols, "top", allowed_colors) and side_ok(left_rows, "left", allowed_colors)

        if top_cols and right_rows and (W - 1) in top_cols and 0 in right_rows:
            return side_ok(top_cols, "top", allowed_colors) and side_ok(right_rows, "right", allowed_colors)

        if bottom_cols and left_rows and 0 in bottom_cols and (H - 1) in left_rows:
            return side_ok(bottom_cols, "bottom", allowed_colors) and side_ok(left_rows, "left", allowed_colors)

        if bottom_cols and right_rows and (W - 1) in bottom_cols and (H - 1) in right_rows:
            return side_ok(bottom_cols, "bottom", allowed_colors) and side_ok(right_rows, "right", allowed_colors)

        return False

    # ------------------------------------------------------------------
    # Find background components / holes.
    # ------------------------------------------------------------------
    bg_mask = (grid == bg)
    comp_id = np.zeros((H, W), dtype=int)
    holes = []  # each: (area, boundary_colors, boundary_pixels_by_color, cells)
    label = 0

    for i in range(H):
        for j in range(W):
            if not bg_mask[i, j] or comp_id[i, j] != 0:
                continue

            label += 1
            stack = [(i, j)]
            comp_id[i, j] = label
            cells = []

            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in dirs4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if bg_mask[nr, nc] and comp_id[nr, nc] == 0:
                            comp_id[nr, nc] = label
                            stack.append((nr, nc))

            # Boundary colors: at most one frame color + possibly key color.
            boundary_colors = set()
            for r, c in cells:
                for dr, dc in dirs4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        v = int(grid[nr, nc])
                        if v != bg:
                            boundary_colors.add(v)
                            if len(boundary_colors) > 2:
                                break
                if len(boundary_colors) > 2:
                    break

            if len(boundary_colors) == 0 or len(boundary_colors) > 2:
                continue

            touches_border = any(
                r == 0 or r == H - 1 or c == 0 or c == W - 1
                for r, c in cells
            )

            if touches_border and not edge_ok(cells, boundary_colors):
                continue

            boundary_pixels = {int(c): set() for c in boundary_colors}
            for r, c in cells:
                for dr, dc in dirs4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        v = int(grid[nr, nc])
                        if v != bg:
                            boundary_pixels[v].add((nr, nc))

            holes.append((len(cells), boundary_colors, boundary_pixels, cells))

    if not holes:
        return out

    # ------------------------------------------------------------------
    # Compatibility with a key color.
    # ------------------------------------------------------------------
    def colors_touch(hole, c1, c2):
        _, _, boundary_pixels, _ = hole
        for r, c in boundary_pixels.get(c1, ()):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and int(grid[nr, nc]) == c2:
                        return True
        return False

    def compatible_frame_color(hole, key):
        _, boundary_colors, _, _ = hole

        if key in boundary_colors:
            if len(boundary_colors) == 1:
                return key
            others = [c for c in boundary_colors if c != key]
            if len(others) != 1:
                return None
            frame_color = others[0]
            if colors_touch(hole, frame_color, key):
                return frame_color
        else:
            if len(boundary_colors) == 1:
                frame_color = next(iter(boundary_colors))
                if colors_touch(hole, frame_color, key):
                    return frame_color

        return None

    # Choose the key color that explains the most compatible holes.
    key_scores = {}
    for key in key_candidates:
        score = 0
        for hole in holes:
            if compatible_frame_color(hole, key) is not None:
                score += 1
        key_scores[key] = score

    if max(key_scores.values()) > 0:
        key = max(key_scores, key=lambda k: (key_scores[k], color_count.get(k, 0), k))
    else:
        key = max(key_candidates, key=lambda k: (color_count.get(k, 0), k))

    # Group compatible holes by frame color.
    groups = {}
    for hole in holes:
        frame_color = compatible_frame_color(hole, key)
        if frame_color is None:
            continue
        groups.setdefault(frame_color, []).append(hole)

    if not groups:
        return out

    # Fill the largest compatible hole for each frame color.
    fill_mask = np.zeros((H, W), dtype=bool)

    for frame_color, hole_list in groups.items():
        max_area = max(hole[0] for hole in hole_list)
        for area, _, _, cells in hole_list:
            if area == max_area:
                for r, c in cells:
                    fill_mask[r, c] = True

    out[fill_mask] = fill
    return out