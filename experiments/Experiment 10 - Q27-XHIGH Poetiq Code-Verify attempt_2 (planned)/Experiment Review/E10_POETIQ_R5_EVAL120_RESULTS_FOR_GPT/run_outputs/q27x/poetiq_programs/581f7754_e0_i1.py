import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform grid according to the observed rule:

    1. The most frequent color is treated as background.
    2. Connected non-background components are extracted.
    3. A single-pixel component on the boundary is an anchor.
       - If anchors are on left/right edges, objects align to their row.
       - If anchors are on top/bottom edges, objects align to their column.
    4. Every component containing exactly one anchor-color pixel is moved
       so that that pixel lies on the anchor row/column.
    5. If a component is a straight one-cell-wide line and the anchor pixel
       is at an endpoint, and the movement is along the line, the line is
       stretched/compressed rather than rigidly translated. This handles the
       vertical 4/6 and 1/8 cases.
    """
    from collections import Counter

    if grid is None or grid.ndim != 2:
        return grid

    H, W = grid.shape
    if H == 0 or W == 0:
        return grid

    vals, counts = np.unique(grid, return_counts=True)
    background = int(vals[np.argmax(counts)])

    def in_bounds(r, c):
        return 0 <= r < H and 0 <= c < W

    # Label 4-connected non-background components.
    comp_id = np.full((H, W), -1, dtype=int)
    raw_components = []

    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) != background and comp_id[r, c] == -1:
                cid = len(raw_components)
                stack = [(r, c)]
                comp_id[r, c] = cid
                cells = []

                while stack:
                    cr, cc = stack.pop()
                    cells.append((cr, cc))

                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if (
                            in_bounds(nr, nc)
                            and int(grid[nr, nc]) != background
                            and comp_id[nr, nc] == -1
                        ):
                            comp_id[nr, nc] = cid
                            stack.append((nr, nc))

                raw_components.append(cells)

    # Build component metadata.
    components = []
    for cells in raw_components:
        color_count = {}
        for r, c in cells:
            col = int(grid[r, c])
            color_count[col] = color_count.get(col, 0) + 1

        heads = {}
        for col in color_count:
            if color_count[col] == 1:
                for r, c in cells:
                    if int(grid[r, c]) == col:
                        heads[col] = (r, c)
                        break

        is_marker = {}
        if len(cells) == 1:
            r, c = cells[0]
            if r == 0 or r == H - 1 or c == 0 or c == W - 1:
                is_marker[int(grid[r, c])] = True

        components.append(
            {
                "cells": cells,
                "counts": color_count,
                "heads": heads,
                "is_marker": is_marker,
                "old_set": set(cells),
            }
        )

    # Find boundary single-pixel anchors.
    markers = {}
    for comp in components:
        for col in comp["is_marker"]:
            markers.setdefault(col, []).append(comp["heads"][col])

    def compute_target(marks):
        vertical_edge = [m for m in marks if m[1] == 0 or m[1] == W - 1]
        horizontal_edge = [m for m in marks if m[0] == 0 or m[0] == H - 1]

        if vertical_edge:
            axis = "row"
            coords = [m[0] for m in vertical_edge]
        elif horizontal_edge:
            axis = "col"
            coords = [m[1] for m in horizontal_edge]
        else:
            return None

        cnt = Counter(coords)
        best = max(cnt.values())
        candidates = sorted(x for x, v in cnt.items() if v == best)
        target = candidates[len(candidates) // 2]
        return axis, target

    color_targets = {}
    for col, marks in markers.items():
        t = compute_target(marks)
        if t is not None:
            color_targets[col] = t

    def rigid_move(comp, head_color, head, target):
        dr = target[0] - head[0]
        dc = target[1] - head[1]

        place = []
        for r, c in comp["cells"]:
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                return None
            place.append((nr, nc, int(grid[r, c])))

        return {"clear": comp["old_set"], "place": place}

    def line_move(comp, head_color, head, target):
        cells = comp["cells"]
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}

        # Horizontal straight line.
        if len(rows) == 1:
            r = next(iter(rows))
            min_c = min(cols)
            max_c = max(cols)

            if len(cols) != max_c - min_c + 1:
                return None

            old_c = head[1]
            new_c = target[1]

            if target[0] != r or new_c == old_c:
                return None

            if old_c == min_c:
                far_c = max_c
            elif old_c == max_c:
                far_c = min_c
            else:
                return None

            # If the target crosses the far endpoint, use rigid movement.
            if far_c != new_c and (new_c - far_c) * (old_c - far_c) < 0:
                return None

            new_len = abs(new_c - far_c)
            seq = []
            if far_c < old_c:
                for c in range(far_c, old_c):
                    seq.append(int(grid[r, c]))
            else:
                for c in range(far_c, old_c, -1):
                    seq.append(int(grid[r, c]))

            place = []
            if new_len:
                orig_len = len(seq)
                if orig_len == 0:
                    return None
                if new_len <= orig_len:
                    colors = seq[:new_len]
                else:
                    colors = seq + [seq[-1]] * (new_len - orig_len)

                if far_c < new_c:
                    for i in range(new_len):
                        place.append((r, far_c + i, colors[i]))
                else:
                    for i in range(new_len):
                        place.append((r, far_c - i, colors[i]))

            place.append((r, new_c, int(head_color)))

            for rr, cc, _ in place:
                if not in_bounds(rr, cc):
                    return None

            return {"clear": comp["old_set"], "place": place}

        # Vertical straight line.
        if len(cols) == 1:
            c = next(iter(cols))
            min_r = min(rows)
            max_r = max(rows)

            if len(rows) != max_r - min_r + 1:
                return None

            old_r = head[0]
            new_r = target[0]

            if target[1] != c or new_r == old_r:
                return None

            if old_r == min_r:
                far_r = max_r
            elif old_r == max_r:
                far_r = min_r
            else:
                return None

            # If the target crosses the far endpoint, use rigid movement.
            if far_r != new_r and (new_r - far_r) * (old_r - far_r) < 0:
                return None

            new_len = abs(new_r - far_r)
            seq = []
            if far_r < old_r:
                for rr in range(far_r, old_r):
                    seq.append(int(grid[rr, c]))
            else:
                for rr in range(far_r, old_r, -1):
                    seq.append(int(grid[rr, c]))

            place = []
            if new_len:
                orig_len = len(seq)
                if orig_len == 0:
                    return None
                if new_len <= orig_len:
                    colors = seq[:new_len]
                else:
                    colors = seq + [seq[-1]] * (new_len - orig_len)

                if far_r < new_r:
                    for i in range(new_len):
                        place.append((far_r + i, c, colors[i]))
                else:
                    for i in range(new_len):
                        place.append((far_r - i, c, colors[i]))

            place.append((new_r, c, int(head_color)))

            for rr, cc, _ in place:
                if not in_bounds(rr, cc):
                    return None

            return {"clear": comp["old_set"], "place": place}

        return None

    moves = []

    for comp in components:
        candidates = []

        for col, (axis, target) in color_targets.items():
            if col not in comp["heads"]:
                continue
            if comp["is_marker"].get(col, False):
                continue

            head = comp["heads"][col]
            if axis == "row":
                target_pos = (int(target), head[1])
            else:
                target_pos = (head[0], int(target))

            dist = abs(target_pos[0] - head[0]) + abs(target_pos[1] - head[1])
            if dist == 0:
                continue

            candidates.append((dist, col, target_pos))

        if not candidates:
            continue

        # If a component could be controlled by more than one anchor color,
        # choose the one requiring the smallest movement.
        candidates.sort(key=lambda x: (x[0], x[1]))
        _, col, target_pos = candidates[0]
        head = comp["heads"][col]

        if len(comp["cells"]) == 1:
            if not in_bounds(target_pos[0], target_pos[1]):
                continue
            move = {
                "clear": comp["old_set"],
                "place": [(target_pos[0], target_pos[1], int(col))],
            }
        else:
            move = line_move(comp, col, head, target_pos)
            if move is None:
                move = rigid_move(comp, col, head, target_pos)

        if move is not None:
            moves.append(move)

    out = grid.copy()
    final_values = {}
    clear_set = set()

    for move in moves:
        clear_set.update(move["clear"])
        for coord, color in move["place"]:
            final_values[coord] = color

    for coord in clear_set | final_values.keys():
        r, c = coord
        if in_bounds(r, c):
            if coord in final_values:
                out[r, c] = final_values[coord]
            else:
                out[r, c] = background

    return out