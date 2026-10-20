import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform each wall-bounded region by concentric rectangular rings.

    The input contains:
      * a background color,
      * a wall/frame color that divides the grid into connected regions,
      * one or more "seed" pixels in some regions.

    For every 4-connected non-wall region:
      * if it has no non-background seed pixels, it is left as background;
      * otherwise a diagonal seed sequence is read from the best corner of the
        region.  The sequence starts at the corner pixel (even if that pixel
        is background), then follows the diagonal inward, appending non-
        background pixels, and stops just before the first background pixel
        after a non-background pixel has already been seen.
      * each pixel in the region is assigned to a ring layer: 0 for pixels on
        the region boundary, 1 for the next inner ring, etc.  This is computed
        by a 4-neighbor distance transform from the region boundary.
      * the color for a pixel is sequence[layer % len(sequence)].

    Wall pixels are preserved unchanged.
    """
    grid = np.array(grid, dtype=int, copy=True)
    if grid.size == 0:
        return grid

    h, w = grid.shape

    # ------------------------------------------------------------------
    # Identify background and wall colors.
    # Background is the most frequent color.  Wall is the most frequent
    # non-background color; in these tasks it forms the frames.
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

    nonwall = grid != wall

    # ------------------------------------------------------------------
    # Label 4-connected non-wall components.
    # ------------------------------------------------------------------
    comp_id = -1 * np.ones((h, w), dtype=int)
    comp_cells = []

    for r in range(h):
        for c in range(w):
            if nonwall[r, c] and comp_id[r, c] == -1:
                cid = len(comp_cells)
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

                comp_cells.append(cells)

    # ------------------------------------------------------------------
    # Helper: read a diagonal seed sequence from a corner.
    #
    # A corner is a component cell that has two perpendicular "outside"
    # neighbours (wall or grid edge).  The diagonal is read inward.
    #
    # The sequence includes:
    #   * the corner pixel, whether background or not;
    #   * any following non-background pixels;
    # and stops before the first background pixel after the first
    # non-background pixel has been seen.
    # ------------------------------------------------------------------
    def in_component(r, c, cid):
        return 0 <= r < h and 0 <= c < w and comp_id[r, c] == cid

    def is_outside(r, c):
        return not (0 <= r < h and 0 <= c < w and nonwall[r, c])

    def read_sequence(r, c, dr, dc, cid):
        seq = []
        seen_non_bg = False

        rr, cc = r, c
        while in_component(rr, cc, cid):
            v = int(grid[rr, cc])
            seq.append(v)

            if v != background:
                seen_non_bg = True
            else:
                if seen_non_bg:
                    break

            rr += dr
            cc += dc

        if any(v != background for v in seq):
            return seq
        return None

    # ------------------------------------------------------------------
    # For each component, find the best corner diagonal.
    # ------------------------------------------------------------------
    out = np.full((h, w), background, dtype=int)

    # Wall pixels are kept exactly.
    out[nonwall == 0] = grid[nonwall == 0]

    for cid, cells in enumerate(comp_cells):
        # If there is no actual seed, leave the whole component as background.
        if not any(grid[r, c] != background for r, c in cells):
            for r, c in cells:
                out[r, c] = background
            continue

        comp_set = set(cells)
        best_seq = None
        best_key = None

        # Candidate corners: any component cell with two perpendicular
        # outside neighbours.
        for r, c in cells:
            up_out = is_outside(r - 1, c)
            down_out = is_outside(r + 1, c)
            left_out = is_outside(r, c - 1)
            right_out = is_outside(r, c + 1)

            candidates = []

            # Top-left corner: interior direction is down-right.
            if up_out and left_out:
                candidates.append((r, c, 1, 1))

            # Top-right corner: interior direction is down-left.
            if up_out and right_out:
                candidates.append((r, c, 1, -1))

            # Bottom-left corner: interior direction is up-right.
            if down_out and left_out:
                candidates.append((r, c, -1, 1))

            # Bottom-right corner: interior direction is up-left.
            if down_out and right_out:
                candidates.append((r, c, -1, -1))

            for sr, sc, dr, dc in candidates:
                seq = read_sequence(sr, sc, dr, dc, cid)
                if seq is None:
                    continue

                # Score:
                #   primary: more non-background seed colors is better
                #   secondary: longer sequence is better
                #   tertiary: prefer earlier/top-leftish corners for stability
                n_nonbg = sum(1 for v in seq if v != background)
                # Corner priority: top-left, top-right, bottom-left, bottom-right
                corner_order = 0
                if dr == 1 and dc == 1:
                    corner_order = 0
                elif dr == 1 and dc == -1:
                    corner_order = 1
                elif dr == -1 and dc == 1:
                    corner_order = 2
                elif dr == -1 and dc == -1:
                    corner_order = 3

                key = (-n_nonbg, -len(seq), corner_order, sr, sc)
                if best_key is None or key < best_key:
                    best_key = key
                    best_seq = seq

        if best_seq is None:
            # No usable diagonal; keep background.
            for r, c in cells:
                out[r, c] = background
            continue

        # ------------------------------------------------------------------
        # Compute ring layers for this component:
        #   boundary cells = 0
        #   next inward cells = 1, etc.
        #
        # This is a 4-neighbor distance transform from the component boundary.
        # ------------------------------------------------------------------
        layer = -1 * np.ones((h, w), dtype=int)
        q = deque()

        for r, c in cells:
            if (
                r == 0 or r == h - 1 or
                c == 0 or c == w - 1 or
                not nonwall[r - 1, c] or
                not nonwall[r + 1, c] or
                not nonwall[r, c - 1] or
                not nonwall[r, c + 1]
            ):
                layer[r, c] = 0
                q.append((r, c))

        while q:
            r, c = q.popleft()
            nd = layer[r, c] + 1
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and comp_id[nr, nc] == cid and layer[nr, nc] == -1:
                    layer[nr, nc] = nd
                    q.append((nr, nc))

        # Fill the component using the cyclic seed sequence.
        L = len(best_seq)
        for r, c in cells:
            # If a cell somehow has no layer (should not happen), use 0.
            d = max(0, layer[r, c])
            out[r, c] = int(best_seq[d % L])

    # Preserve wall pixels exactly.
    out[grid == wall] = grid[grid == wall]

    return out