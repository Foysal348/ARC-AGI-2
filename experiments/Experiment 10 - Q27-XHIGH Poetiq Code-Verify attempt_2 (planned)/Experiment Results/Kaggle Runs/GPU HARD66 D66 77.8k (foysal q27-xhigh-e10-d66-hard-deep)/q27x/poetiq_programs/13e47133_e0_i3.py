import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform wall-separated rectangular rooms into concentric rectangle fills.

    The grid contains:
      * a background color,
      * a wall color forming line segments that divide the grid into rooms,
      * one or more seed pixels in some rooms.

    The rule used here:
      1. Identify the background color (most frequent) and the wall color
         (most frequent non-background color).
      2. Treat wall cells as separators.  For each 4-connected non-wall
         component, determine the rectangular span of the component.
      3. For each component, find a diagonal seed sequence starting from one
         of its four corners.  The sequence is read along the diagonal inward:
           - include leading background pixels if the corner itself is background;
           - include following non-background pixels;
           - stop before the first background pixel after a non-background
             pixel has already been seen.
         If several corner diagonals contain seeds, prefer the corner whose
         diagonal has more seed colors, then longer diagonal, then top-left,
         top-right, bottom-left, bottom-right.
      4. Fill the component with concentric rectangular rings.  The ring index
         of a cell is its Chebyshev distance from the component boundary,
         i.e. min(row offset, column offset, height-1-row offset,
         width-1-column offset).
      5. The color of ring i is sequence[i % len(sequence)].
      6. Components with no usable diagonal seed remain background.
      7. Wall cells are preserved unchanged.

    This implements the visible examples' behavior: seed diagonals define a
    short color pattern, and each wall-bounded room is painted by repeating
    that pattern over concentric rectangular rings.
    """
    grid = np.array(grid, dtype=int, copy=True)
    if grid.size == 0:
        return grid

    h, w = grid.shape

    # ------------------------------------------------------------------
    # 1. Background and wall colors.
    # ------------------------------------------------------------------
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[np.argmax(counts)])

    non_bg_counts = {
        int(c): int(n)
        for c, n in zip(colors, counts)
        if int(c) != background
    }

    if not non_bg_counts:
        return grid

    wall = int(max(non_bg_counts, key=lambda c: non_bg_counts[c]))

    # ------------------------------------------------------------------
    # 2. 4-connected non-wall components.
    # ------------------------------------------------------------------
    nonwall = grid != wall
    comp_id = -np.ones((h, w), dtype=int)
    components = []

    for r in range(h):
        for c in range(w):
            if nonwall[r, c] and comp_id[r, c] == -1:
                cid = len(components)
                cells = []
                q = deque([(r, c)])
                comp_id[r, c] = cid
                cells.append((r, c))

                while q:
                    cr, cc = q.popleft()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < h and 0 <= nc < w and
                            nonwall[nr, nc] and comp_id[nr, nc] == -1
                        ):
                            comp_id[nr, nc] = cid
                            cells.append((nr, nc))
                            q.append((nr, nc))

                components.append(cells)

    out = np.full((h, w), background, dtype=int)
    out[grid == wall] = grid[grid == wall]

    # ------------------------------------------------------------------
    # Helpers for diagonal seed extraction.
    # ------------------------------------------------------------------
    def in_component(r, c, cid):
        return 0 <= r < h and 0 <= c < w and comp_id[r, c] == cid

    def read_diagonal(r, c, dr, dc, cid):
        """
        Read a diagonal seed sequence inward from a corner.

        Includes:
          * leading background pixels (if the corner is background),
          * non-background pixels immediately after,
        and stops before the first background pixel after a non-background
        pixel has been seen.
        """
        seq = []
        seen_non_bg = False
        rr, cc = r, c

        while in_component(rr, cc, cid):
            v = int(grid[rr, cc])

            if v != background:
                seen_non_bg = True
                seq.append(v)
            else:
                if seen_non_bg:
                    break
                # Leading background may be part of the sequence, e.g.
                # corner background followed by a seed.
                seq.append(v)

            rr += dr
            cc += dc

        # Need at least one non-background seed to be meaningful.
        if any(v != background for v in seq):
            return seq
        return None

    def corner_candidates(cells, cid):
        """
        Candidate diagonal starts are component cells that have two
        perpendicular 'outside' neighbours (wall or grid edge).
        """
        candidates = []

        for r, c in cells:
            up_out = r == 0 or not nonwall[r - 1, c]
            down_out = r == h - 1 or not nonwall[r + 1, c]
            left_out = c == 0 or not nonwall[r, c - 1]
            right_out = c == w - 1 or not nonwall[r, c + 1]

            # Top-left corner: go down-right.
            if up_out and left_out:
                candidates.append((r, c, 1, 1, 0))
            # Top-right corner: go down-left.
            if up_out and right_out:
                candidates.append((r, c, 1, -1, 1))
            # Bottom-left corner: go up-right.
            if down_out and left_out:
                candidates.append((r, c, -1, 1, 2))
            # Bottom-right corner: go up-left.
            if down_out and right_out:
                candidates.append((r, c, -1, -1, 3))

        return candidates

    # ------------------------------------------------------------------
    # Fill each component.
    # ------------------------------------------------------------------
    for cid, cells in enumerate(components):
        if not cells:
            continue

        # If there is no actual seed in this component, leave background.
        has_seed = any(grid[r, c] != background for r, c in cells)
        if not has_seed:
            for r, c in cells:
                out[r, c] = background
            continue

        # Find the best diagonal seed sequence.
        best_seq = None
        best_key = None

        for r, c, dr, dc, corner_order in corner_candidates(cells, cid):
            seq = read_diagonal(r, c, dr, dc, cid)
            if seq is None:
                continue

            n_nonbg = sum(1 for v in seq if v != background)
            # Prefer more seed colors, then longer sequence,
            # then stable corner order.
            key = (-n_nonbg, -len(seq), corner_order, r, c)
            if best_key is None or key < best_key:
                best_key = key
                best_seq = seq

        if best_seq is None:
            for r, c in cells:
                out[r, c] = background
            continue

        # Bounding rectangle of this component.
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        comp_h = r1 - r0 + 1
        comp_w = c1 - c0 + 1

        # Fill using concentric rectangular rings inside the bounding box.
        # Only cells actually belonging to the component are changed.
        L = len(best_seq)
        for r, c in cells:
            rr = r - r0
            cc = c - c0
            layer = min(rr, cc, comp_h - 1 - rr, comp_w - 1 - cc)
            out[r, c] = int(best_seq[layer % L])

    # Ensure walls are exactly unchanged.
    out[grid == wall] = grid[grid == wall]

    return out