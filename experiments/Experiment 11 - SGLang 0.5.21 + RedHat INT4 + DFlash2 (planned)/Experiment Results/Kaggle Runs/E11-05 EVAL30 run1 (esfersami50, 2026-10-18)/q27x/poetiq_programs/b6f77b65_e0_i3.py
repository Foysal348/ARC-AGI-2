import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule inferred from the examples:

    1. The cell at (0, 0) is a command marker. Its color is removed from
       everywhere else in the grid, but the marker itself is kept.

    2. The amount to move objects down is determined by the command color:
           1, 2, 3 -> 1
           4, 5, 6 -> 2
           7, 8    -> 3
           9       -> 0

    3. After removing the command color (except the marker), treat each
       4-connected non-zero component as a rigid object.  Each such object
       moves downward by the command-derived shift, except that:
           - it cannot move if doing so would push any part of it outside
             the grid,
           - it cannot move if it already touches the bottom border.

       This matches the examples where bottom-touching objects stay fixed
       while higher objects slide down.

    The implementation is deliberately conservative: it first removes the
    command color, labels 4-connected components, computes the shift, and
    places shifted copies when safe.  If a component cannot be shifted
    without clipping or because it touches the bottom, it remains in place.
    """
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape
    work = grid.astype(int).copy()

    # Preserve command marker.
    cmd = int(work[0, 0])

    # Remove command color everywhere except top-left.
    work[work == cmd] = 0
    work[0, 0] = cmd

    # Shift mapping inferred from examples.
    shift = {
        1: 1,
        2: 1,
        3: 1,
        4: 2,
        5: 2,
        6: 2,
        7: 3,
        8: 3,
        9: 0,
    }.get(cmd, 0)

    if shift == 0:
        return work

    # Label 4-connected components of non-zero cells in the cleaned grid.
    visited = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            if work[r, c] != 0 and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                comp = [(r, c)]

                while q:
                    cr, cc = q.popleft()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if work[nr, nc] != 0 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                                comp.append((nr, nc))

                components.append(comp)

    result = np.zeros_like(work)
    result[0, 0] = cmd

    for comp in components:
        # Do not move the command marker component.
        if (0, 0) in comp:
            for r, c in comp:
                result[r, c] = work[r, c]
            continue

        # Determine if component can be shifted downward.
        shifted_cells = []
        can_shift = True
        touches_bottom = any(r == h - 1 for r, _ in comp)

        if not touches_bottom:
            for r, c in comp:
                nr = r + shift
                if nr >= h:
                    can_shift = False
                    break
                shifted_cells.append((nr, c))
        else:
            can_shift = False

        if can_shift and len(shifted_cells) == len(comp):
            for (nr, nc), (orig_r, orig_c) in zip(shifted_cells, comp):
                result[nr, nc] = work[orig_r, orig_c]
        else:
            # Leave in original position.
            for r, c in comp:
                result[r, c] = work[r, c]

    return result