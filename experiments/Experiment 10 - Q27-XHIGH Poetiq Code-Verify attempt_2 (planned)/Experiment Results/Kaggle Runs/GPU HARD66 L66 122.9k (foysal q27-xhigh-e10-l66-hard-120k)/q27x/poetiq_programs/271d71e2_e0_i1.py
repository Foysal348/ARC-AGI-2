import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform each movable 0/5/7 block next to a 9 segment.

    A block moves toward the next parallel 9 segment in the direction of the
    touching 9 segment. It moves by:

        min(distance_to_far_9, number_of_5_cells)

    While moving, that many 5 cells are converted to 7. The converted 5 cells
    are chosen in a direction-dependent scan order:
        right: columns right-to-left, top-to-bottom
        left:  columns left-to-right, bottom-to-top
        up:    rows top-to-bottom, left-to-right
        down:  rows bottom-to-top, right-to-left
    """
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape

    def find_far_h(row, c1, c2, step):
        """Find next horizontal 9 segment with same column range."""
        r = row + step
        while 0 <= r < h:
            if np.all(grid[r, c1:c2 + 1] == 9):
                return r
            r += step
        return None

    def find_far_v(col, r1, r2, step):
        """Find next vertical 9 segment with same row range."""
        c = col + step
        while 0 <= c < w:
            if np.all(grid[r1:r2 + 1, c] == 9):
                return c
            c += step
        return None

    def order_key(cell, direction):
        r, c = cell
        if direction == "R":
            return (-c, r)
        if direction == "L":
            return (c, -r)
        if direction == "U":
            return (r, c)
        if direction == "D":
            return (-r, -c)
        return (r, c)

    # Cells that belong to movable blocks: 0, 5, or 7.
    target = (grid == 0) | (grid == 5) | (grid == 7)

    # Find connected components of target cells.
    visited = np.zeros((h, w), dtype=bool)
    components = []

    for i in range(h):
        for j in range(w):
            if target[i, j] and not visited[i, j]:
                stack = [(i, j)]
                visited[i, j] = True
                cells = []

                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))

                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and target[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))

                r1 = min(r for r, _ in cells)
                r2 = max(r for r, _ in cells)
                c1 = min(c for _, c in cells)
                c2 = max(c for _, c in cells)
                count5 = sum(1 for r, c in cells if grid[r, c] == 5)

                components.append((cells, (r1, r2, c1, c2), count5))

    groups = []

    for cells, bbox, count5 in components:
        if count5 == 0:
            continue

        r1, r2, c1, c2 = bbox
        candidates = []

        # Touching a horizontal 9 segment above: move up.
        if r1 > 0 and np.all(grid[r1 - 1, c1:c2 + 1] == 9):
            far = find_far_h(r1 - 1, c1, c2, -1)
            if far is not None:
                dmax = abs(far - (r1 - 1))
                if dmax > 0:
                    candidates.append((dmax, "U", "H", r1 - 1, far, c1, c2))

        # Touching a horizontal 9 segment below: move down.
        if r2 + 1 < h and np.all(grid[r2 + 1, c1:c2 + 1] == 9):
            far = find_far_h(r2 + 1, c1, c2, 1)
            if far is not None:
                dmax = abs(far - (r2 + 1))
                if dmax > 0:
                    candidates.append((dmax, "D", "H", r2 + 1, far, c1, c2))

        # Touching a vertical 9 segment left: move left.
        if c1 > 0 and np.all(grid[r1:r2 + 1, c1 - 1] == 9):
            far = find_far_v(c1 - 1, r1, r2, -1)
            if far is not None:
                dmax = abs(far - (c1 - 1))
                if dmax > 0:
                    candidates.append((dmax, "L", "V", c1 - 1, far, r1, r2))

        # Touching a vertical 9 segment right: move right.
        if c2 + 1 < w and np.all(grid[r1:r2 + 1, c2 + 1] == 9):
            far = find_far_v(c2 + 1, r1, r2, 1)
            if far is not None:
                dmax = abs(far - (c2 + 1))
                if dmax > 0:
                    candidates.append((dmax, "R", "V", c2 + 1, far, r1, r2))

        if not candidates:
            continue

        # If multiple candidates exist, use the nearest far 9 segment.
        candidates.sort(key=lambda x: x[0])
        _, direction, orientation, wall_coord, far_coord, a, b = candidates[0]

        dmax = abs(far_coord - wall_coord)
        dist = min(dmax, count5)
        if dist == 0:
            continue

        if direction == "U":
            dr, dc = -1, 0
        elif direction == "D":
            dr, dc = 1, 0
        elif direction == "L":
            dr, dc = 0, -1
        else:  # direction == "R"
            dr, dc = 0, 1

        five_cells = [(r, c) for r, c in cells if grid[r, c] == 5]
        five_cells.sort(key=lambda cell: order_key(cell, direction))
        convert_set = set(five_cells[:dist])

        groups.append({
            "cells": cells,
            "direction": direction,
            "dr": dr,
            "dc": dc,
            "orientation": orientation,
            "wall_coord": wall_coord,
            "far_coord": far_coord,
            "start": a,
            "end": b,
            "dist": dist,
            "convert_set": convert_set,
        })

    if not groups:
        return grid.copy()

    # Clear old block cells and old moving wall cells.
    clear = np.zeros((h, w), dtype=bool)

    for g in groups:
        for r, c in g["cells"]:
            clear[r, c] = True

        if g["orientation"] == "H":
            for c in range(g["start"], g["end"] + 1):
                clear[g["wall_coord"], c] = True
        else:
            for r in range(g["start"], g["end"] + 1):
                clear[r, g["wall_coord"]] = True

    out = grid.copy()
    out[clear] = 6

    # Write moved block cells.
    for g in groups:
        d = g["dist"]
        dr = g["dr"]
        dc = g["dc"]
        convert_set = g["convert_set"]

        for r, c in g["cells"]:
            val = grid[r, c]
            if val == 5 and (r, c) in convert_set:
                val = 7

            nr = r + d * dr
            nc = c + d * dc
            out[nr, nc] = val

    # Write moved 9 walls, and restore/confirm far 9 walls.
    for g in groups:
        d = g["dist"]

        if g["orientation"] == "H":
            new_coord = g["wall_coord"] + d * g["dr"]
            for c in range(g["start"], g["end"] + 1):
                out[new_coord, c] = 9
                out[g["far_coord"], c] = 9
        else:
            new_coord = g["wall_coord"] + d * g["dc"]
            for r in range(g["start"], g["end"] + 1):
                out[r, new_coord] = 9
                out[r, g["far_coord"]] = 9

    return out