import numpy as np
from collections import deque

def _flood_box(grid, start, bg, fg):
    """Return all cells of the background component containing start."""
    h, w = grid.shape
    r0, c0 = start
    seen = np.zeros((h, w), dtype=bool)
    q = deque([(r0, c0)])
    seen[r0, c0] = True
    cells = []

    while q:
        r, c = q.popleft()
        cells.append((r, c))
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == bg:
                seen[nr, nc] = True
                q.append((nr, nc))

    return cells


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Simplify a rectangular wall/room drawing.

    The non-background color is treated as wall.  The input is reduced to the
    principal enclosed room structure: the largest fully enclosed background
    region is represented by a rectangular wall border.  Inside it, enclosed
    background regions are represented recursively as smaller wall borders.
    Isolated foreground cells inside a room are kept as wall dots.
    """
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) == 1:
        return np.zeros_like(grid)

    # Background is the majority color.
    bg = int(colors[np.argmax(counts)])
    # Foreground/wall is the other single color in these tasks.
    fg = int(colors[np.argmin(counts)] if counts[np.argmin(counts)] < counts[np.argmax(counts)] else colors[0])
    if fg == bg:
        # Fallback for unexpected inputs.
        fg = int(colors[(np.argmax(counts) + 1) % len(colors)])

    # Mark background cells connected to the outer border.
    outside = np.zeros((h, w), dtype=bool)
    q = deque()

    for i in range(h):
        for j in (0, w - 1):
            if grid[i, j] == bg and not outside[i, j]:
                outside[i, j] = True
                q.append((i, j))

    for j in range(w):
        for i in (0, h - 1):
            if grid[i, j] == bg and not outside[i, j]:
                outside[i, j] = True
                q.append((i, j))

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not outside[nr, nc] and grid[nr, nc] == bg:
                outside[nr, nc] = True
                q.append((nr, nc))

    # Label enclosed background components.
    enclosed = np.zeros((h, w), dtype=bool)
    components = []
    for i in range(h):
        for j in range(w):
            if grid[i, j] == bg and not outside[i, j] and not enclosed[i, j]:
                cells = _flood_box(grid, (i, j), bg, fg)
                for r, c in cells:
                    enclosed[r, c] = True
                components.append(cells)

    if not components:
        # No enclosed room: return a minimal foreground dot if possible.
        out = np.full((1, 1), bg, dtype=grid.dtype)
        out[0, 0] = fg
        return out

    # Choose the largest enclosed room by area.
    main_cells = max(components, key=len)
    rows = [r for r, _ in main_cells]
    cols = [c for _, c in main_cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    # Output room interior plus a one-pixel wall border.
    inner_h = r1 - r0 + 1
    inner_w = c1 - c0 + 1
    out = np.full((inner_h + 2, inner_w + 2), bg, dtype=grid.dtype)

    # Outer wall border.
    out[0, :] = fg
    out[-1, :] = fg
    out[:, 0] = fg
    out[:, -1] = fg

    # Recursively add nested rooms and isolated wall dots inside the main room.
    # A nested component is considered if it lies strictly inside the main room
    # and is fully enclosed by the original walls.
    nested = []
    for cells in components:
        rr = [r for r, _ in cells]
        cc = [c for _, c in cells]
        if r0 < min(rr) and max(rr) < r1 and c0 < min(cc) and max(cc) < c1:
            nested.append(cells)

    # Draw isolated foreground pixels inside the main room that are not part of
    # larger nested rooms.
    fg_positions = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if grid[r, c] == fg:
                # Ignore foreground that is immediately adjacent to the room
                # boundary in the simplified drawing; keep interior dots.
                fg_positions.append((r - r0 + 1, c - c0 + 1))

    # Add smaller enclosed background regions as nested rectangular borders.
    # Only keep sufficiently small/central nested rooms to match the intended
    # simplified schematic style.
    nested_bboxes = []
    for cells in nested:
        rr = [r for r, _ in cells]
        cc = [c for _, c in cells]
        nr0, nr1 = min(rr), max(rr)
        nc0, nc1 = min(cc), max(cc)
        nh = nr1 - nr0 + 1
        nw = nc1 - nc0 + 1
        if nh >= 1 and nw >= 1:
            nested_bboxes.append((nr0 - r0 + 1, nr1 - r0 + 1, nc0 - c0 + 1, nc1 - c0 + 1))

    # For the observed task family, at most one principal nested room is needed.
    # If present, draw its wall border.
    if nested_bboxes:
        # Prefer the largest nested room, but prefer a compact/central one if
        # multiple are present.
        def score(bbox):
            tr, br, tc, bc = bbox
            area = (br - tr + 1) * (bc - tc + 1)
            # Center bias relative to interior.
            cr = (tr + br) / 2
            cc = (tc + bc) / 2
            center_dist = abs(cr - (inner_h + 1) / 2) + abs(cc - (inner_w + 1) / 2)
            return (area, -center_dist)

        tr, br, tc, bc = max(nested_bboxes, key=score)
        out[tr:br + 1, tc:tc + 1] = bg
        out[tr, tc:bc + 1] = fg
        out[br, tc:bc + 1] = fg
        out[tr:br + 1, tc] = fg
        out[tr:br + 1, bc] = fg

    # Add isolated wall dots.
    for r, c in fg_positions:
        # Do not overwrite nested room borders except true isolated dots.
        # In these tasks, isolated dots are not on the border of a nested room.
        if out[r, c] == bg:
            out[r, c] = fg

    # If there is a nested room and its center contains a foreground pixel in
    # the original room, ensure the center dot is present.
    if nested_bboxes:
        tr, br, tc, bc = max(nested_bboxes, key=lambda b: ((b[1] - b[0] + 1) * (b[3] - b[2] + 1)))
        mr = (tr + br) // 2
        mc = (tc + bc) // 2
        # Original center coordinate:
        orig_r = mr - 1
        orig_c = mc - 1
        if 0 <= orig_r < inner_h and 0 <= orig_c < inner_w and grid[r0 + orig_r, c0 + orig_c] == fg:
            out[mr, mc] = fg

    return out