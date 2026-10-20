import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Move reachable 9s into the 3x3 target square.

    Rules inferred from the examples:
    1. The target is the 3x3 block centered on the unique 2, whose border is 3.
    2. Color 6 is a wall; all other colors are passable for the purpose of
       reaching the target.
    3. Perform a 4-directional BFS from all target cells.
    4. Any 9 in the reachable region can enter the target square.
    5. The target square has 9 cells, so at most 9 9s are consumed.
       If more reachable 9s exist, consume the closest ones by BFS distance
       (ties broken by row, then column).
    6. Consumed 9s become 7, and the target cells are filled with 9 in
       row-major order from the top-left cell.
    """
    out = np.array(grid, dtype=int, copy=True)

    if out.ndim != 2:
        return out

    h, w = out.shape
    if h == 0 or w == 0:
        return out

    # Locate the 3x3 target block. Prefer a 2 surrounded by eight 3s.
    target_center = None
    for r, c in np.argwhere(out == 2):
        if 0 <= r - 1 and r + 1 < h and 0 <= c - 1 and c + 1 < w:
            ok = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if out[r + dr, c + dc] != 3:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                target_center = (r, c)
                break

    # Fallback if no perfect target is found.
    if target_center is None:
        twos = np.argwhere(out == 2)
        if len(twos) == 0:
            return out
        target_center = tuple(twos[0])

    r, c = target_center

    # Target cells in row-major order.
    target_cells = []
    target_set = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                target_cells.append((rr, cc))
                target_set.add((rr, cc))

    if not target_cells:
        return out

    # BFS from the target block, treating 6 as blocked.
    dist = np.full((h, w), -1, dtype=int)
    q = deque()

    for rr, cc in target_cells:
        if out[rr, cc] != 6:
            dist[rr, cc] = 0
            q.append((rr, cc))

    while q:
        rr, cc = q.popleft()
        nd = dist[rr, cc] + 1

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = rr + dr, cc + dc
            if (
                0 <= nr < h
                and 0 <= nc < w
                and dist[nr, nc] == -1
                and out[nr, nc] != 6
            ):
                dist[nr, nc] = nd
                q.append((nr, nc))

    # Collect reachable 9s outside the target block.
    tokens = []
    for i in range(h):
        for j in range(w):
            if (
                out[i, j] == 9
                and (i, j) not in target_set
                and dist[i, j] != -1
            ):
                tokens.append((dist[i, j], i, j))

    # Closest first; tie-break by row, then column.
    tokens.sort()

    # Consume at most as many as there are target cells.
    k = min(len(tokens), len(target_cells))

    # Remove consumed 9s from their original positions.
    for _, rr, cc in tokens[:k]:
        out[rr, cc] = 7

    # Fill the target block in row-major order.
    for i in range(k):
        out[target_cells[i]] = 9

    return out