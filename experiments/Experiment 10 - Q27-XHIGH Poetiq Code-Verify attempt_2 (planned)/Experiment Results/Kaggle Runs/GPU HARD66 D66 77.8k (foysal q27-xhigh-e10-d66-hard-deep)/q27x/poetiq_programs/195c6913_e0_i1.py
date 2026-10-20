import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden thin "river/skeleton" path from a thick colored band.

    The rule inferred from the examples:
    1. Identify the thick band color.  This is the color adjacent to the
       left-edge start pixels that are not the main background color.
    2. Identify the repeating pattern colors from the 2x2 blocks in the
       top-left corner, read left-to-right.
    3. Identify the endpoint color from the isolated 2x2 block near the
       bottom-right corner.
    4. Start from the left-edge start pixels and trace monotone (up/right)
       centerline branches through the thick band.  These branches form the
       thin path that is painted with the repeating pattern.
    5. Color the traced path by distance from the start, modulo the pattern
       length.  Mark true dead-ends with the endpoint color, except for the
       original left-edge start pixels.

    The implementation is conservative and tailored to the geometry present
    in the examples/challenges: the path is a set of axis-aligned monotone
    branches moving only upward and rightward from the left-edge starts.
    """
    if grid.size == 0:
        return grid

    H, W = grid.shape
    out = grid.astype(int, copy=True)

    # -------------------------------------------------------------------------
    # Helper: find connected components (4-neighbor) of a color.
    # -------------------------------------------------------------------------
    def components_of_color(color):
        seen = np.zeros((H, W), dtype=bool)
        comps = []
        for i in range(H):
            for j in range(W):
                if not seen[i, j] and grid[i, j] == color:
                    q = deque([(i, j)])
                    seen[i, j] = True
                    comp = []
                    while q:
                        r, c = q.popleft()
                        comp.append((r, c))
                        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and grid[nr, nc] == color:
                                seen[nr, nc] = True
                                q.append((nr, nc))
                    comps.append(comp)
        return comps

    # -------------------------------------------------------------------------
    # Find all distinct colors and choose main background / thick band colors.
    # -------------------------------------------------------------------------
    colors = np.unique(grid)
    color_counts = {int(c): int(np.sum(grid == c)) for c in colors}

    # The background color is usually the most frequent color, but in some
    # examples the thick band itself is the most frequent.  We identify starts
    # first to separate them.
    # Left-edge start pixels: non-zero cells in column 0 that are not part of
    # a 2x2 block and are not the most frequent color.
    starts = []
    most_common_color = max(color_counts, key=color_counts.get)
    for r in range(H):
        c = int(grid[r, 0])
        if c == 0:
            continue
        # skip if this cell belongs to a 2x2 block
        if r + 1 < H and grid[r+1, 0] == c and r - 1 >= 0 and grid[r-1, 0] == c:
            continue
        # skip if it is the global most common color
        if c == most_common_color:
            continue
        starts.append((r, 0))

    if not starts:
        # Fallback: if no obvious start pixels, leave input unchanged.
        return out

    # For each start, the thick band color is the neighbor color to the right
    # if possible, else up/down.
    band_candidates = []
    for r, c in starts:
        for nr, nc in ((r, c+1), (r-1, c), (r+1, c)):
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] != grid[r, c]:
                band_candidates.append(int(grid[nr, nc]))
                break
    if not band_candidates:
        return out
    band_color = max(band_candidates, key=lambda x: color_counts.get(x, 0))

    # -------------------------------------------------------------------------
    # Extract pattern colors from top-left 2x2 blocks.
    # We scan the first ~10 rows and columns for solid 2x2 blocks that are not
    # the band/background colors and are not the left-edge starts.
    # -------------------------------------------------------------------------
    pattern = []
    used = np.zeros((H, W), dtype=bool)

    # Mark all start pixels as used.
    for r, c in starts:
        used[r, c] = True

    # Find 2x2 blocks in top-left region.
    for r in range(0, min(H, 10)):
        for c in range(0, min(W, 12)):
            col = int(grid[r, c])
            if col in (band_color,):
                continue
            if used[r, c]:
                continue
            if r+1 < H and c+1 < W and \
               grid[r, c] == grid[r, c+1] == grid[r+1, c] == grid[r+1, c+1] == col:
                # Accept if not already represented
                if col not in pattern:
                    pattern.append(col)
                for rr in (r, r+1):
                    for cc in (c, c+1):
                        used[rr, cc] = True

    # If no pattern found, use all non-background, non-band colors from top-left.
    if not pattern:
        for r in range(0, min(H, 10)):
            for c in range(0, min(W, 12)):
                col = int(grid[r, c])
                if col != band_color and col != most_common_color and col not in pattern:
                    pattern.append(col)

    if not pattern:
        return out

    # -------------------------------------------------------------------------
    # Identify endpoint color: a 2x2 block of a unique color in the bottom
    # right region, not in pattern and not band/background.
    # -------------------------------------------------------------------------
    endpoint_color = None
    for r in range(max(0, H-8), H):
        for c in range(max(0, W-8), W):
            col = int(grid[r, c])
            if col in pattern or col == band_color or col == most_common_color:
                continue
            if r+1 < H and c+1 < W and \
               grid[r, c] == grid[r, c+1] == grid[r+1, c] == grid[r+1, c+1] == col:
                endpoint_color = col
                break
        if endpoint_color is not None:
            break

    if endpoint_color is None:
        # Fallback: least frequent non-pattern/non-band color
        for col in sorted(color_counts, key=lambda x: color_counts[x]):
            if col not in pattern and col != band_color and col != most_common_color:
                endpoint_color = col
                break

    # Remove the top-left pattern blocks and endpoint block from output.
    # They are replaced by the local background color (usually the most common
    # color or the color of their immediate surroundings).
    def erase_2x2(r, c):
        if r+1 < H and c+1 < W:
            # Use the color of the cell just above-left if possible, else most common.
            fill = int(most_common_color)
            if r-1 >= 0 and c-1 >= 0 and grid[r-1, c-1] != grid[r, c]:
                fill = int(grid[r-1, c-1])
            for rr in (r, r+1):
                for cc in (c, c+1):
                    out[rr, cc] = fill

    for r in range(0, min(H, 10)):
        for c in range(0, min(W, 12)):
            col = int(grid[r, c])
            if col in pattern:
                if r+1 < H and c+1 < W and \
                   grid[r, c] == grid[r, c+1] == grid[r+1, c] == grid[r+1, c+1] == col:
                    erase_2x2(r, c)

    if endpoint_color is not None:
        for r in range(max(0, H-8), H):
            for c in range(max(0, W-8), W):
                col = int(grid[r, c])
                if col == endpoint_color and r+1 < H and c+1 < W and \
                   grid[r, c] == grid[r, c+1] == grid[r+1, c] == grid[r+1, c+1] == col:
                    erase_2x2(r, c)
                    break
            else:
                continue
            break

    # -------------------------------------------------------------------------
    # Trace monotone up/right branches from each start through the band.
    # -------------------------------------------------------------------------
    path_cells = set(starts)

    def in_band(r, c):
        return 0 <= r < H and 0 <= c < W and (out[r, c] == band_color or grid[r, c] == band_color)

    def is_start(r, c):
        return (r, c) in starts

    # For each start, explore a small directed graph allowing moves:
    #   right (0, +1), up (-1, 0)
    # We keep only cells that are on "central" branches:
    # a move is allowed if the destination is band color (or an allowed endpoint
    # wall cell just beyond a dead end).  To avoid filling entire wide areas,
    # we restrict branching to moves that stay on a straight line for at least
    # 3 cells before turning, unless the move ends at a boundary.
    for sr, sc in starts:
        stack = [(sr, sc, 0, 1)]  # (r, c, dist, direction), direction 1=right, 2=up
        seen_branch = set()

        while stack:
            r, c, d, dirn = stack.pop()
            path_cells.add((r, c))

            # If current cell is a dead end (not a start), mark endpoint color.
            # Dead end: no right or up band neighbor.
            right_ok = 0 <= r < H and 0 <= c+1 < W and in_band(r, c+1)
            up_ok = 0 <= r-1 < H and 0 <= c < W and in_band(r-1, c)
            if not right_ok and not up_ok and not is_start(r, c):
                # Mark the cell just beyond if possible, else current cell.
                # In examples endpoints are often placed one step outside the
                # band in the direction opposite to the incoming path.
                if dirn == 1:  # came from left, endpoint to the right
                    er, ec = r, c+1
                    if 0 <= ec < W:
                        out[er, ec] = endpoint_color
                elif dirn == 2:  # came from below, endpoint above
                    er, ec = r-1, c
                    if 0 <= er < H:
                        out[er, ec] = endpoint_color
                else:
                    out[r, c] = endpoint_color

            # Continue in the same direction if possible.
            if dirn == 1:
                if right_ok:
                    stack.append((r, c+1, d+1, 1))
                # Turn up only if we have traveled at least 3 cells right
                # in this branch and the up cell is band.
                if up_ok and d >= 3:
                    stack.append((r-1, c, d+1, 2))
            elif dirn == 2:
                if up_ok:
                    stack.append((r-1, c, d+1, 2))
                # Turn right only if we have traveled at least 3 cells up.
                if right_ok and d >= 3:
                    stack.append((r, c+1, d+1, 1))

    # -------------------------------------------------------------------------
    # Color the traced path by distance from its start.  We need distances for
    # each path cell from the start that generated it.  Re-run a directed BFS
    # from each start to assign the minimal directed distance.
    # -------------------------------------------------------------------------
    dist = {}
    for sr, sc in starts:
        dq = deque()
        dq.append((sr, sc, 0))
        best = {}
        while dq:
            r, c, d = dq.popleft()
            if (r, c) in best and best[(r, c)] <= d:
                continue
            best[(r, c)] = d
            if 0 <= r < H and 0 <= c+1 < W and in_band(r, c+1):
                dq.append((r, c+1, d+1))
            if 0 <= r-1 < H and 0 <= c < W and in_band(r-1, c):
                dq.append((r-1, c, d+1))
        for cell, dd in best.items():
            if cell not in dist or dd < dist[cell]:
                dist[cell] = dd

    L = len(pattern)
    for (r, c) in path_cells:
        if is_start(r, c):
            # Keep the original start color (which is already in the pattern).
            continue
        d = dist.get((r, c), 0)
        out[r, c] = pattern[d % L]

    # Restore left-edge start pixels to their original input colors.
    for r, c in starts:
        out[r, c] = int(grid[r, c])

    return out