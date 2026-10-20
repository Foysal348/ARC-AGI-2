import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill with 7 every 4-connected non-7 region that contains all key colors
    and does not contain the marker color 3.

    The key colors are the non-zero, non-3, non-7 pixels in the 4x4 block
    whose top-left corner is the top-left corner of the 3-marker.
    Components containing 3 are always preserved.
    """
    arr = np.asarray(grid, dtype=int)
    if arr.ndim != 2:
        return arr.copy()

    out = arr.copy()
    h, w = out.shape
    if h == 0 or w == 0:
        return out

    WALL = 7
    MARKER = 3

    # Locate the 4x4 key box from the marker's top-left corner.
    marker_mask = (out == MARKER)
    if np.any(marker_mask):
        row_has = np.any(marker_mask, axis=1)
        col_has = np.any(marker_mask, axis=0)
        r0 = int(np.argmax(row_has))
        c0 = int(np.argmax(col_has))
    else:
        r0, c0 = 0, 0

    r1 = min(h, r0 + 4)
    c1 = min(w, c0 + 4)

    key_colors = set()
    for r in range(r0, r1):
        for c in range(c0, c1):
            v = int(out[r, c])
            if v not in (WALL, MARKER, 0):
                key_colors.add(v)

    # If there are no key colors, nothing should be filled.
    if not key_colors:
        return out

    # Label 4-connected components of non-wall cells.
    open_mask = (out != WALL)
    comp_id = np.zeros((h, w), dtype=int)
    comp_has_marker = [False]
    comp_colors = [set()]
    next_id = 0
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for i in range(h):
        for j in range(w):
            if open_mask[i, j] and comp_id[i, j] == 0:
                next_id += 1
                cid = next_id
                stack = [(i, j)]
                comp_id[i, j] = cid

                has_marker = False
                colors = set()

                while stack:
                    r, c = stack.pop()
                    v = int(out[r, c])

                    if v == MARKER:
                        has_marker = True
                    elif v != 0:
                        colors.add(v)

                    for dr, dc in dirs:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if open_mask[nr, nc] and comp_id[nr, nc] == 0:
                                comp_id[nr, nc] = cid
                                stack.append((nr, nc))

                comp_has_marker.append(has_marker)
                comp_colors.append(colors)

    # Mark components that should be erased.
    fill_component = np.zeros(next_id + 1, dtype=bool)
    for cid in range(1, next_id + 1):
        if not comp_has_marker[cid] and key_colors.issubset(comp_colors[cid]):
            fill_component[cid] = True

    out[fill_component[comp_id]] = WALL
    return out