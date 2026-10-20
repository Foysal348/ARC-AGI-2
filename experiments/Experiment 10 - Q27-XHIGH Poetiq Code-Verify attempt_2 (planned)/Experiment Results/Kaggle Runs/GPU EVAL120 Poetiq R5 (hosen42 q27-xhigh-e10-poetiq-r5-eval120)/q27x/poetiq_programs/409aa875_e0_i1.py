import numpy as np
from collections import Counter


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform small 3-pixel arrows/corners into fixed-distance shadows.

    Rule:
      1. Treat the majority color as background.
      2. Find every connected 3-pixel foreground object.
      3. For each valid object:
         - 2x2 corner: the "point" is the cell orthogonally connected to both
           other cells; it points diagonally away from the missing corner.
         - 2x3 or 3x2 chevron: the "point" is the single centered cell; it
           points straight away from the other two cells.
      4. Put a shadow 5 cells away from the point in that direction.
      5. Shadow color is 9, unless two shadows share a cell, in which case
         that cell is 1.
      6. If a shadow lands on an existing foreground object, recolor the whole
         object to the shadow color.
      7. All other cells remain unchanged.
    """
    arr = np.array(grid, dtype=int, copy=True)

    if arr.ndim != 2 or arr.size == 0:
        return arr

    h, w = arr.shape

    # Background is the most common color.
    vals, counts = np.unique(arr, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    out = arr.copy()

    def sign(x: int) -> int:
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    def classify(cells):
        """
        Return ((point_row, point_col), (dr, dc)) for a valid 3-cell pointer,
        or None if the cells do not form a recognized pointer.
        """
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]

        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        height = max_r - min_r + 1
        width = max_c - min_c + 1

        # 2x2 corner: one cell touches the other two orthogonally.
        if height == 2 and width == 2:
            elbow = None
            for r, c in cells:
                orth_count = 0
                for r2, c2 in cells:
                    if (r, c) != (r2, c2) and abs(r - r2) + abs(c - c2) == 1:
                        orth_count += 1
                if orth_count == 2:
                    elbow = (r, c)
                    break

            if elbow is None:
                return None

            er, ec = elbow
            arms = [cell for cell in cells if cell != elbow]

            # Direction is away from the other two cells.
            dr = 2 * er - arms[0][0] - arms[1][0]
            dc = 2 * ec - arms[0][1] - arms[1][1]

            if abs(dr) != 1 or abs(dc) != 1:
                return None

            return elbow, (sign(dr), sign(dc))

        # 2x3 horizontal chevron.
        if height == 2 and width == 3:
            row_count = Counter(r for r, _ in cells)
            single_rows = [r for r, cnt in row_count.items() if cnt == 1]
            double_rows = [r for r, cnt in row_count.items() if cnt == 2]

            if len(single_rows) != 1 or len(double_rows) != 1:
                return None

            point_row = single_rows[0]
            arm_row = double_rows[0]

            point_cols = [c for r, c in cells if r == point_row]
            arm_cols = sorted(c for r, c in cells if r == arm_row)

            if len(point_cols) != 1 or len(arm_cols) != 2:
                return None
            if arm_cols[1] - arm_cols[0] != 2:
                return None

            point_col = point_cols[0]
            if point_col != (arm_cols[0] + arm_cols[1]) // 2:
                return None

            point = (point_row, point_col)
            direction = (-1, 0) if point_row < arm_row else (1, 0)
            return point, direction

        # 3x2 vertical chevron.
        if height == 3 and width == 2:
            col_count = Counter(c for _, c in cells)
            single_cols = [c for c, cnt in col_count.items() if cnt == 1]
            double_cols = [c for c, cnt in col_count.items() if cnt == 2]

            if len(single_cols) != 1 or len(double_cols) != 1:
                return None

            point_col = single_cols[0]
            arm_col = double_cols[0]

            point_rows = [r for r, c in cells if c == point_col]
            arm_rows = sorted(r for r, c in cells if c == arm_col)

            if len(point_rows) != 1 or len(arm_rows) != 2:
                return None
            if arm_rows[1] - arm_rows[0] != 2:
                return None

            point_row = point_rows[0]
            if point_row != (arm_rows[0] + arm_rows[1]) // 2:
                return None

            point = (point_row, point_col)
            direction = (0, -1) if point_col < arm_col else (0, 1)
            return point, direction

        return None

    # Find all foreground connected components, using 8-connectivity.
    comp_cells = []
    cell_to_comp = {}
    targets = []

    non_bg_colors = [int(c) for c in vals if c != bg]
    dirs8 = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    for color in non_bg_colors:
        mask = (arr == color)
        visited = np.zeros((h, w), dtype=bool)

        for r0 in range(h):
            for c0 in range(w):
                if not mask[r0, c0] or visited[r0, c0]:
                    continue

                cells = []
                stack = [(r0, c0)]
                visited[r0, c0] = True

                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))

                    for dr, dc in dirs8:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if mask[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                stack.append((nr, nc))

                cid = len(comp_cells)
                comp_cells.append(cells)
                for r, c in cells:
                    cell_to_comp[(r, c)] = cid

                # Only 3-cell objects are pointers.
                if len(cells) == 3:
                    classified = classify(cells)
                    if classified is not None:
                        (pr, pc), (dr, dc) = classified
                        tr, tc = pr + 5 * dr, pc + 5 * dc
                        if 0 <= tr < h and 0 <= tc < w:
                            targets.append((tr, tc))

    if not targets:
        return out

    target_counts = Counter(targets)

    # Determine which objects should be recolored because a shadow hits them.
    comp_recolor = {}
    for tr, tc in targets:
        shadow_color = 1 if target_counts[(tr, tc)] >= 2 else 9
        hit_cid = cell_to_comp.get((tr, tc))
        if hit_cid is not None:
            # If any collision shadow hits the object, prefer color 1.
            if shadow_color == 1 or hit_cid not in comp_recolor:
                comp_recolor[hit_cid] = shadow_color

    # Recolor hit objects.
    for cid, color in comp_recolor.items():
        for r, c in comp_cells[cid]:
            out[r, c] = color

    # Place shadow cells.
    for tr, tc in targets:
        shadow_color = 1 if target_counts[(tr, tc)] >= 2 else 9
        out[tr, tc] = shadow_color

    return out