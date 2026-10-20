import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform each rectangular room of non-7 cells.

    A room is erased (changed to 7) iff:
      - it contains exactly one 1, and
      - the next non-zero digit after that 1 in row-major order is not 5.
    """
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape
    out = grid.copy()
    visited = np.zeros((h, w), dtype=bool)

    for r in range(h):
        for c in range(w):
            if visited[r, c] or out[r, c] == 7:
                continue

            # Flood fill this room of non-7 cells using 4-connectivity.
            room = []
            stack = [(r, c)]
            visited[r, c] = True

            while stack:
                cr, cc = stack.pop()
                room.append((cr, cc))

                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if not visited[nr, nc] and out[nr, nc] != 7:
                            visited[nr, nc] = True
                            stack.append((nr, nc))

            # Sort non-zero cells in row-major order.
            nonzero_cells = sorted(
                (rr, cc, out[rr, cc])
                for rr, cc in room
                if out[rr, cc] != 0
            )

            ones = [x for _, _, x in nonzero_cells if x == 1]

            erase = False
            if len(ones) == 1:
                idx = next(i for i, val in enumerate(nonzero_cells) if val == 1)
                if idx + 1 < len(nonzero_cells):
                    nxt_val = nonzero_cells[idx + 1][2]
                    if nxt_val != 5:
                        erase = True

            if erase:
                for rr, cc in room:
                    out[rr, cc] = 7

    return out