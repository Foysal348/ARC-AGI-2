import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Remove noisy extra pixels from colored objects.

    The rule used here is a conservative denoising operation:
      * For every non-background color, work with its 8-connected components.
      * Iteratively remove pixels that are local "bumps" or over-thickened
        artifacts:
          - a single-pixel protrusion with very few same-color neighbours,
          - part of a 2x2 block that is not well supported by the rest of the
            component,
          - an overhang at a corner that is only weakly connected.
      * The operation is repeated until stable.
    """
    if grid.size == 0:
        return grid

    g = grid.astype(np.int32, copy=True)
    h, w = g.shape

    # Background is the border colour, or 0 if border is not uniform.
    border = g[0, 0]
    if not (np.all(g[0, :] == border) and
            np.all(g[-1, :] == border) and
            np.all(g[:, 0] == border) and
            np.all(g[:, -1] == border)):
        # Fallback: most common colour on the border
        vals, counts = np.unique(np.concatenate([
            g[0, :].ravel(), g[-1, :].ravel(),
            g[:, 0].ravel(), g[:, -1].ravel()
        ]), return_counts=True)
        border = int(vals[np.argmax(counts)])

    colors = [int(c) for c in np.unique(g) if c != border]

    def neighbours8(r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    yield nr, nc

    def same_count(r, c, color):
        return sum(1 for nr, nc in neighbours8(r, c) if g[nr, nc] == color)

    def component_mask(color):
        """Return 8-connected components of a colour as a list of masks."""
        unvisited = (g == color)
        components = []
        for i in range(h):
            for j in range(w):
                if unvisited[i, j]:
                    mask = np.zeros((h, w), dtype=bool)
                    q = deque([(i, j)])
                    unvisited[i, j] = False
                    mask[i, j] = True
                    while q:
                        r, c = q.popleft()
                        for nr, nc in neighbours8(r, c):
                            if unvisited[nr, nc] and g[nr, nc] == color:
                                unvisited[nr, nc] = False
                                mask[nr, nc] = True
                                q.append((nr, nc))
                    components.append(mask)
        return components

    def support_outside_block(r, c, color, block):
        """
        Support outside a 2x2 block:
        count same-coloured 8-neighbours not in the block.
        """
        cnt = 0
        for nr, nc in neighbours8(r, c):
            if g[nr, nc] == color and (nr, nc) not in block:
                cnt += 1
        return cnt

    def has_quadrant_bump(r, c, color):
        """
        Detect a local convex bump: all same-colour neighbours are in a small
        angular region (corner / overhang), and the pixel has very low
        cardinal support.
        """
        orth = 0
        diag = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and g[nr, nc] == color:
                orth += 1
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and g[nr, nc] == color:
                diag += 1

        # Single orthogonal protrusion
        if orth == 1 and diag <= 1:
            return True

        # T-shaped / cap-like bump above/below a short line
        if orth == 1 and diag == 3:
            return True

        # Corner overhang with only two orthogonal and one/two diagonal
        if orth == 2 and diag in (1, 2):
            # Check if the two orthogonal directions are perpendicular
            dirs = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and g[nr, nc] == color:
                    dirs.append((dr, dc))
            if len(dirs) == 2:
                (dr1, dc1), (dr2, dc2) = dirs
                if dr1 * dr2 + dc1 * dc2 == 0:
                    return True
        return False

    def prune_color(color):
        for mask in component_mask(color):
            changed = True
            while changed:
                changed = False
                rs, cs = np.where(mask)
                for r, c in zip(rs, cs):
                    if g[r, c] != color:
                        continue

                    # 1) remove weak single-pixel bumps
                    if has_quadrant_bump(r, c, color):
                        total = same_count(r, c, color)
                        # keep endpoints that are part of a clear diagonal bridge
                        orth = sum(1 for dr, dc in ((-1,0),(1,0),(0,-1),(0,1))
                                   if 0 <= r+dr < h and 0 <= c+dc < w
                                   and g[r+dr, c+dc] == color)
                        if total <= 2 and orth <= 1:
                            g[r, c] = border
                            mask[r, c] = False
                            changed = True
                            continue

                    # 2) thin 2x2 blocks that are not well supported
                    # check the four 2x2 blocks where (r,c) can be a corner
                    for dr, dc in ((-1, -1), (-1, 0), (0, -1), (0, 0)):
                        r0, c0 = r + dr, c + dc
                        if not (0 <= r0 < h - 1 and 0 <= c0 < w - 1):
                            continue
                        block = [(r0, c0), (r0, c0 + 1),
                                 (r0 + 1, c0), (r0 + 1, c0 + 1)]
                        if all(g[x, y] == color for x, y in block):
                            sup = support_outside_block(r, c, color, set(block))
                            # If this pixel has almost no support outside the
                            # 2x2 block, it is likely an extra thickening.
                            if sup == 0:
                                g[r, c] = border
                                mask[r, c] = False
                                changed = True
                                break

    # Apply pruning to each non-background colour.
    for color in colors:
        prune_color(color)

    return g.astype(grid.dtype, copy=False)