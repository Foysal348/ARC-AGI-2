import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid according to the 3x3 icon color-chain rule.

    Icons are 3x3 patterns with a distinct center color and a border color.
    The border may be a full 3x3 ring or a plus shape, and clipped border icons
    are allowed near grid edges.

    For each icon, if its center color equals another icon's frame color,
    create a directed link to that icon.  Chains are followed from starts
    (icons whose frame color is not any icon's center color).  Each pair of
    consecutive icons in a chain is compressed: the first is kept and its
    center is replaced by the second's original center; the second is erased.
    Unpaired icons are erased.

    All 5-cells are preserved unchanged.
    """
    grid = np.asarray(grid, dtype=int)

    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape

    # Output starts empty; special color 5 is preserved exactly.
    out = np.zeros((H, W), dtype=int)
    out[grid == 5] = 5

    dirs = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    cardinal = {(-1, 0), (1, 0), (0, -1), (0, 1)}

    # ------------------------------------------------------------------
    # Detect icons.
    # Each icon is stored as (center_row, center_col, frame_color, center_color).
    # ------------------------------------------------------------------
    objects = []

    for r in range(H):
        for c in range(W):
            center_val = int(grid[r, c])

            # Background and special color are not icon centers.
            if center_val == 0 or center_val == 5:
                continue

            neigh = []
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    neigh.append((dr, dc, int(grid[nr, nc])))

            if not neigh:
                continue

            # All non-background, non-special visible border cells must be one color.
            border_colors = set()
            for _, _, v in neigh:
                if v != 0 and v != 5:
                    border_colors.add(v)

            if len(border_colors) != 1:
                continue

            frame = border_colors.pop()

            # Center must differ from the border.
            if frame == center_val:
                continue

            frame_count = sum(1 for _, _, v in neigh if v == frame)

            # Full ring: every visible neighbor is the frame color.
            is_full = all(v == frame for _, _, v in neigh)

            # Plus shape: every visible cardinal neighbor is frame,
            # every visible corner neighbor is empty (0) or special 5.
            is_plus = True
            for dr, dc, v in neigh:
                if (dr, dc) in cardinal:
                    if v != frame:
                        is_plus = False
                        break
                else:
                    if v not in (0, 5):
                        is_plus = False
                        break

            if is_full:
                # At least a clipped corner full icon has 3 visible cells.
                if frame_count < 3:
                    continue
            elif is_plus:
                # A plus icon at a grid corner may have only 2 visible arms.
                if frame_count < 2:
                    continue
            else:
                continue

            objects.append((r, c, frame, center_val))

    N = len(objects)
    if N == 0:
        return out

    # ------------------------------------------------------------------
    # Build directed color links:
    # icon i -> icon j if icon i's center color equals icon j's frame color.
    # Frame colors are expected to be unique; if not, use the first in row-major order.
    # ------------------------------------------------------------------
    frame_to_idx = {}
    for i, (_, _, frame, _) in enumerate(objects):
        if frame not in frame_to_idx:
            frame_to_idx[frame] = i

    out_map = [-1] * N
    for i, (_, _, _, center) in enumerate(objects):
        j = frame_to_idx.get(center, -1)
        if j == i:
            out_map[i] = -1
        else:
            out_map[i] = j

    indeg = [0] * N
    for j in out_map:
        if j != -1:
            indeg[j] += 1

    visited = [False] * N
    kept = [False] * N
    new_center = [0] * N

    # ------------------------------------------------------------------
    # Process chains.
    # Starts are icons with no incoming link.  Row-major order is used for
    # deterministic handling if chains merge.
    # ------------------------------------------------------------------
    def process_chain(start):
        cur = start
        while cur != -1 and not visited[cur]:
            nxt = out_map[cur]

            if nxt != -1 and not visited[nxt]:
                # Pair current with next:
                # keep current, erase next, move current's center to next's old center.
                kept[cur] = True
                new_center[cur] = objects[nxt][3]

                visited[cur] = True
                visited[nxt] = True

                # Continue from the object after the erased one.
                cur = out_map[nxt]
            else:
                # No usable next icon; this icon is unpaired/remaining, so erase it.
                visited[cur] = True
                kept[cur] = False
                break

    for i in range(N):
        if indeg[i] == 0 and not visited[i]:
            process_chain(i)

    # Handle any remaining cycles or components deterministically.
    for i in range(N):
        if not visited[i]:
            process_chain(i)

    # ------------------------------------------------------------------
    # Reconstruct output:
    # - 5 cells already copied
    # - kept icons redrawn with their updated center color
    # - removed icons left as zero
    # ------------------------------------------------------------------
    for i, (r, c, frame, _) in enumerate(objects):
        if kept[i]:
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if grid[nr, nc] == frame:
                        out[nr, nc] = frame

            out[r, c] = int(new_center[i])

    return out