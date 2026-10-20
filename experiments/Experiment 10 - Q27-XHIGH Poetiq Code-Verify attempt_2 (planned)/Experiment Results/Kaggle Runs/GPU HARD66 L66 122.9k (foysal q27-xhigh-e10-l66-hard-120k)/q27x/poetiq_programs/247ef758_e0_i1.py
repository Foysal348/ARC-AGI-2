import numpy as np
from collections import deque, defaultdict


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid according to the observed rule:

    - A vertical separator column divides the left object panel and the right
      bordered frame.
    - Non-background colors on the frame border define row ports and column
      ports.
    - For each color that has both row ports and column ports, copy every
      connected object of that color from the left panel to the right frame,
      centered at every (row port, column port) intersection.
    - Used left objects are removed.
    - If stamps overlap, the smaller color value remains.
    """
    arr = np.array(grid, dtype=int, copy=True)
    H, W = arr.shape

    if H < 2 or W < 2:
        return arr

    # ------------------------------------------------------------------
    # Find the separator column.
    #
    # It is a constant non-zero column immediately to the left of a full
    # nonzero frame border.  We prefer a candidate whose right neighbor has
    # more than one nonzero color, because the frame border contains marks.
    # ------------------------------------------------------------------
    sep = None
    fallback_sep = None

    for c in range(W - 1):
        col = arr[:, c]

        # Separator must be a single non-zero color in every row.
        if col[0] == 0 or not np.all(col == col[0]):
            continue

        right_start = c + 1
        if right_start >= W:
            continue

        top = arr[0, right_start:]
        bottom = arr[-1, right_start:]
        left_border = arr[:, right_start]
        right_border = arr[:, -1]

        # The right frame border must be fully non-zero.
        if (
            top.size == 0
            or not np.all(top != 0)
            or not np.all(bottom != 0)
            or not np.all(left_border != 0)
            or not np.all(right_border != 0)
        ):
            continue

        if fallback_sep is None:
            fallback_sep = c

        # Prefer the true frame: its left border should contain marks,
        # i.e. more than one non-zero color.
        unique_left_border = np.unique(left_border[left_border != 0])
        if len(unique_left_border) > 1:
            sep = c
            break

    if sep is None:
        sep = fallback_sep

    if sep is None:
        return arr

    left = arr[:, :sep].copy()
    right = arr[:, sep + 1:].copy()
    H, Wr = right.shape

    out = arr.copy()
    out_left = left.copy()
    out_right = right.copy()

    # The interior of the right frame is replaced by the stamped objects.
    if H > 2 and Wr > 2:
        out_right[1:-1, 1:-1] = 0

    # ------------------------------------------------------------------
    # Determine the frame's base/background border color.
    # ------------------------------------------------------------------
    def frame_base_color(panel: np.ndarray) -> int:
        h, w = panel.shape
        if h == 0 or w == 0:
            return 0

        mask = np.zeros((h, w), dtype=bool)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True

        border_colors = panel[mask]
        border_colors = border_colors[border_colors != 0]

        if border_colors.size == 0:
            return 0

        corners = np.array(
            [
                panel[0, 0],
                panel[0, w - 1],
                panel[h - 1, 0],
                panel[h - 1, w - 1],
            ]
        )

        # If all corners agree, that is almost certainly the frame base color.
        unique_corners = np.unique(corners)
        if len(unique_corners) == 1 and unique_corners[0] != 0:
            return int(unique_corners[0])

        vals, counts = np.unique(border_colors, return_counts=True)
        max_count = counts.max()
        candidates = vals[counts == max_count]

        if len(candidates) == 1:
            return int(candidates[0])

        # Tie-breakers: prefer a color appearing in at least three corners,
        # then a color appearing on all four border sides.
        for val in candidates:
            if int(np.sum(corners == val)) >= 3:
                return int(val)

        for val in candidates:
            if (
                np.any(panel[0, :] == val)
                and np.any(panel[-1, :] == val)
                and np.any(panel[:, 0] == val)
                and np.any(panel[:, -1] == val)
            ):
                return int(val)

        return int(candidates[0])

    base = frame_base_color(right)

    # ------------------------------------------------------------------
    # Collect non-base port rows and columns.
    # ------------------------------------------------------------------
    row_ports = defaultdict(set)
    col_ports = defaultdict(set)

    for r in range(H):
        for c in (0, Wr - 1):
            val = int(right[r, c])
            if val != 0 and val != base:
                row_ports[val].add(r)

    for c in range(Wr):
        for r in (0, H - 1):
            val = int(right[r, c])
            if val != 0 and val != base:
                col_ports[val].add(c)

    # ------------------------------------------------------------------
    # Extract connected components from the left panel.
    # 8-connectivity is used so diagonal objects remain one object.
    # ------------------------------------------------------------------
    visited = np.zeros(left.shape, dtype=bool)
    components = []

    for r in range(left.shape[0]):
        for c in range(left.shape[1]):
            val = int(left[r, c])
            if val == 0 or visited[r, c]:
                continue

            q = deque([(r, c)])
            visited[r, c] = True
            cells = []

            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))

                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue

                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < left.shape[0]
                            and 0 <= nc < left.shape[1]
                            and not visited[nr, nc]
                            and int(left[nr, nc]) == val
                        ):
                            visited[nr, nc] = True
                            q.append((nr, nc))

            components.append((val, cells))

    # ------------------------------------------------------------------
    # Stamp used objects into the right frame and remove them from left.
    # ------------------------------------------------------------------
    for color, cells in components:
        rset = row_ports.get(color, set())
        cset = col_ports.get(color, set())

        # Only colors with both row and column ports can be placed.
        if not rset or not cset:
            continue

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]

        min_r = min(rows)
        max_r = max(rows)
        min_c = min(cols)
        max_c = max(cols)

        h = max_r - min_r + 1
        w = max_c - min_c + 1

        # Relative coordinates inside the object's bounding box.
        rel_cells = [(r - min_r, c - min_c) for r, c in cells]

        # Center of the bounding box.
        center_r = h // 2
        center_c = w // 2

        placed = False

        for tr in sorted(rset):
            for tc in sorted(cset):
                target_cells = []
                valid = True

                for dr, dc in rel_cells:
                    rr = tr - center_r + dr
                    cc = tc - center_c + dc

                    # Keep objects entirely inside the frame interior.
                    if rr < 1 or rr >= H - 1 or cc < 1 or cc >= Wr - 1:
                        valid = False
                        break

                    target_cells.append((rr, cc))

                if not valid:
                    continue

                placed = True

                for rr, cc in target_cells:
                    current = int(out_right[rr, cc])
                    # Smaller color value wins on overlap.
                    if current == 0 or color < current:
                        out_right[rr, cc] = color

        # Remove this object from the left panel only if it was placed.
        if placed:
            for r, c in cells:
                out_left[r, c] = 0

    # ------------------------------------------------------------------
    # Reassemble output.
    # ------------------------------------------------------------------
    out[:, :sep] = out_left
    out[:, sep + 1:] = out_right

    return out