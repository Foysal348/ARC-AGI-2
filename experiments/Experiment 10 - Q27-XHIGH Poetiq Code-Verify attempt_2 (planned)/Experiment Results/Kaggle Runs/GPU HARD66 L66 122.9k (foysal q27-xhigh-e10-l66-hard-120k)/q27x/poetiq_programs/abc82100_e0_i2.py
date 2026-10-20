import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule
    ==============
    The 8-pixels form an obstacle / central object.  The remaining coloured
    pixels are grouped into local clusters (same colour, close together).

    Each cluster is assigned to one of the four sides of the grid that it is
    closest to.  The colour used for a side is taken from the opposite side.
    The cluster is then projected outward toward its side:

      * If the cluster is a thin line (one row or one column), only the
        immediate outward / lateral neighbours are painted.
      * Otherwise the cluster itself is painted and rays are cast in the
        three outward directions for that side.  For diagonal rays every
        painted cell keeps the cluster's checkerboard parity; for the
        orthogonal ray the same parity is also preserved.

    All 8-pixels are removed in the output.
    """

    grid = np.array(grid, dtype=int)
    if grid.ndim != 2:
        return grid

    H, W = grid.shape
    out = np.zeros((H, W), dtype=int)

    # No obstacle: keep non-zero pixels unchanged.
    if not (grid == 8).any():
        return grid.copy()

    eight = (grid == 8)
    eight_centroid = np.argwhere(eight).mean(axis=0)
    cr, cc = float(eight_centroid[0]), float(eight_centroid[1])

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def dilate(mask: np.ndarray) -> np.ndarray:
        """8-neighbour dilation, used only to group nearby same-colour pixels."""
        d = mask.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                d |= np.roll(np.roll(mask, dr, axis=0), dc, axis=1)
        return d

    def label_8(mask: np.ndarray):
        """Label 8-connected components in a binary mask."""
        labels = np.zeros(mask.shape, dtype=int)
        n = 0
        Hm, Wm = mask.shape
        for r in range(Hm):
            for c in range(Wm):
                if not mask[r, c] or labels[r, c]:
                    continue
                n += 1
                stack = [(r, c)]
                labels[r, c] = n
                while stack:
                    rr, rc = stack.pop()
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = rr + dr, rc + dc
                            if 0 <= nr < Hm and 0 <= nc < Wm:
                                if mask[nr, nc] and not labels[nr, nc]:
                                    labels[nr, nc] = n
                                    stack.append((nr, nc))
        return labels, n

    def choose_side(r, c):
        """
        Choose the side closest to the cluster centroid.
        Ties are broken using the direction from the 8-centre.
        """
        distances = {
            "top": r,
            "bottom": H - 1 - r,
            "left": c,
            "right": W - 1 - c,
        }
        best = min(distances.values())
        candidates = [s for s, d in distances.items() if d == best]

        if len(candidates) == 1:
            return candidates[0]

        dr = r - cr
        dc = c - cc

        if abs(dr) >= abs(dc):
            pref = ["top", "bottom"] if dr <= 0 else ["bottom", "top"]
        else:
            pref = ["left", "right"] if dc <= 0 else ["right", "left"]

        for s in pref:
            if s in candidates:
                return s

        return candidates[0]

    def is_line(cells):
        """A cluster is line-like if all its cells share one row or one column."""
        if not cells:
            return True
        rows = {r for r, c, _ in cells}
        cols = {c for r, c, _ in cells}
        return len(rows) == 1 or len(cols) == 1

    # Directions for one-step projection of line clusters.
    line_dirs = {
        "top": [(-1, 0), (0, -1), (0, 1)],
        "bottom": [(1, 0), (0, -1), (0, 1)],
        "left": [(0, -1), (-1, 0), (1, 0)],
        "right": [(0, 1), (-1, 0), (1, 0)],
    }

    # Directions for ray projection of 2-D clusters.
    ray_dirs = {
        "top": [(-1, -1), (-1, 0), (-1, 1)],
        "bottom": [(1, -1), (1, 0), (1, 1)],
        "left": [(-1, -1), (0, -1), (1, -1)],
        "right": [(-1, 1), (0, 1), (1, 1)],
    }

    # ------------------------------------------------------------------
    # Build non-8 clusters, grouped by colour, using a light dilation
    # ------------------------------------------------------------------
    colours = [c for c in np.unique(grid) if c != 0 and c != 8]
    clusters_by_side = {"top": [], "bottom": [], "left": [], "right": []}
    side_colour_votes = {
        "top": {},
        "bottom": {},
        "left": {},
        "right": {},
    }

    for colour in colours:
        mask = (grid == colour)
        if not mask.any():
            continue

        grown = dilate(mask)
        lab, nlab = label_8(grown)

        for lab_id in range(1, nlab + 1):
            cells = []
            for r, c in zip(*np.where((lab == lab_id) & mask)):
                cells.append((int(r), int(c), int(colour)))

            if not cells:
                continue

            centroid = np.mean([(r, c) for r, c, _ in cells], axis=0)
            side = choose_side(centroid[0], centroid[1])

            clusters_by_side[side].append(cells)

            cnt = side_colour_votes[side].get(colour, 0) + len(cells)
            side_colour_votes[side][colour] = cnt

    # ------------------------------------------------------------------
    # Determine the paint colour for each side: opposite side majority colour
    # ------------------------------------------------------------------
    def majority(colour_counts):
        if not colour_counts:
            return None
        return max(colour_counts.items(), key=lambda kv: kv[1])[0]

    side_colour = {
        side: majority(side_colour_votes[side])
        for side in side_colour_votes
    }

    paint_colour = {
        "top": side_colour["bottom"],
        "bottom": side_colour["top"],
        "left": side_colour["right"],
        "right": side_colour["left"],
    }

    # ------------------------------------------------------------------
    # Project each cluster outward using the colour from the opposite side
    # ------------------------------------------------------------------
    for side, clusters in clusters_by_side.items():
        pcolour = paint_colour[side]
        if pcolour is None:
            continue

        for cells in clusters:
            if not cells:
                continue

            # If a line cluster, only paint immediate outward neighbours.
            if is_line(cells):
                for r, c, _ in cells:
                    for dr, dc in line_dirs[side]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and not eight[nr, nc]:
                            out[nr, nc] = pcolour
                continue

            # 2-D cluster: paint the cluster itself and cast outward rays.
            parities = [((r + c) & 1) for r, c, _ in cells]
            parity = max(set(parities), key=parities.count)

            for r, c, _ in cells:
                if ((r + c) & 1) == parity:
                    out[r, c] = pcolour

                for dr, dc in ray_dirs[side]:
                    rr, cc_ = r, c
                    while True:
                        nr, nc = rr + dr, cc_ + dc
                        if not (0 <= nr < H and 0 <= nc < W):
                            break
                        if eight[nr, nc]:
                            break

                        if ((nr + nc) & 1) == parity:
                            out[nr, nc] = pcolour

                        rr, cc_ = nr, nc

    return out