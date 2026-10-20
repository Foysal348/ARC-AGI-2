import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Extract all 4x4 colored objects from the input grid.

    Each object is classified as:
      - solid: its 4x4 bounding box is completely occupied by the object
      - hollow: its 4x4 bounding box is not completely occupied by the object

    The output is a 2-column grid of 4x4 blocks:
      - left column: hollow objects
      - right column: solid objects

    Objects in each column are ordered by their original top edge
    in the input grid, with left-to-right as a tie-breaker.
    Empty slots are filled with 0.
    """
    grid = np.asarray(grid, dtype=int)

    if grid.ndim != 2:
        return np.zeros((4, 8), dtype=int)

    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects = []

    # Find connected monochromatic components.
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0 or visited[r, c]:
                continue

            color = grid[r, c]
            stack = [(r, c)]
            visited[r, c] = True

            min_r = max_r = r
            min_c = max_c = c
            cells = []

            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))

                if cr < min_r:
                    min_r = cr
                if cr > max_r:
                    max_r = cr
                if cc < min_c:
                    min_c = cc
                if cc > max_c:
                    max_c = cc

                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not visited[nr, nc]
                        and grid[nr, nc] == color
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            # We only care about the 4x4 objects used in this task.
            if max_r - min_r + 1 != 4 or max_c - min_c + 1 != 4:
                continue

            block = grid[min_r:min_r + 4, min_c:min_c + 4].copy()

            # Determine whether this 4x4 object fills its whole bounding box.
            mask = np.zeros((4, 4), dtype=bool)
            for rr, cc in cells:
                mask[rr - min_r, cc - min_c] = True

            is_solid = bool(np.all(mask))
            objects.append((min_r, min_c, block, is_solid))

    hollow = [obj for obj in objects if not obj[3]]
    solid = [obj for obj in objects if obj[3]]

    # Preserve original vertical order within each class.
    hollow.sort(key=lambda x: (x[0], x[1]))
    solid.sort(key=lambda x: (x[0], x[1]))

    rows = max(len(hollow), len(solid))
    if rows == 0:
        return np.zeros((4, 8), dtype=int)

    out = np.zeros((4 * rows, 8), dtype=int)

    for i, (_, _, block, _) in enumerate(hollow):
        out[4 * i:4 * i + 4, 0:4] = block

    for i, (_, _, block, _) in enumerate(solid):
        out[4 * i:4 * i + 4, 4:8] = block

    return out