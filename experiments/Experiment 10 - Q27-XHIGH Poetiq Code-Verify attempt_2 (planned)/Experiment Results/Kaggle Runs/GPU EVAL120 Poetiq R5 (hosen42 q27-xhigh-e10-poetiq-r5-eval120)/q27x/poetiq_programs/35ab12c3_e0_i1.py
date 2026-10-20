import numpy as np
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Complete partial colored shapes.

    Rule implemented:
    1. For every color that has at least two pixels, connect its pixels with
       straight 8-neighbor line segments whenever they are aligned horizontally,
       vertically, or diagonally. This reconstructs the main visible strokes.
    2. If a color appears as a single isolated pixel next to a completed stroke
       of another color, treat the isolated pixel as the seed of a parallel
       stroke: shift the entire completed stroke of the neighboring color by the
       adjacency vector and draw it in the isolated pixel's color.
       Cells already occupied by the original stroke or by another color are
       not overwritten.
    """

    if grid.size == 0:
        return grid

    h, w = grid.shape
    out = grid.copy()

    points_by_color = defaultdict(list)
    for r in range(h):
        for c in range(w):
            v = int(grid[r, c])
            if v != 0:
                points_by_color[v].append((r, c))

    # Make deterministic point order
    for color in points_by_color:
        points_by_color[color].sort()

    def chebyshev(p, q):
        return max(abs(p[0] - q[0]), abs(p[1] - q[1]))

    def is_aligned(p, q):
        dr = p[0] - q[0]
        dc = p[1] - q[1]
        return dr == 0 or dc == 0 or abs(dr) == abs(dc)

    def draw_8_path(r0, c0, r1, c1, color, mask, allow_override_other=False):
        """
        Draw a straightest 8-neighbor path from (r0,c0) to (r1,c1).
        Prefers diagonal moves while both row and column still need to change.
        """
        if r0 < 0 or r1 < 0 or c0 < 0 or c1 < 0 or r0 >= h or r1 >= h or c0 >= w or c1 >= w:
            return

        r, c = r0, c0
        first = True
        while (r != r1) or (c != c1):
            if first:
                first = False
                rr, cc = r0, c0
            else:
                rr, cc = r, c

            if 0 <= rr < h and 0 <= cc < w:
                if out[rr, cc] == 0 or out[rr, cc] == color or allow_override_other:
                    out[rr, cc] = color
                mask.add((rr, cc))

            sr = (r1 > r) - (r1 < r)
            sc = (c1 > c) - (c1 < c)

            if sr != 0 and sc != 0:
                r += sr
                c += sc
            elif sr != 0 and (sc == 0 or abs(r1 - r) >= abs(c1 - c)):
                r += sr
            elif sc != 0:
                c += sc
            else:
                break

        # Final point
        if 0 <= r < h and 0 <= c < w:
            if out[r, c] == 0 or out[r, c] == color or allow_override_other:
                out[r, c] = color
            mask.add((r, c))

    # Primary masks: all cells belonging to completed strokes for colors with >=2 pixels.
    primary_masks = {}

    # Step 1: complete same-color strokes.
    for color, pts in points_by_color.items():
        if len(pts) < 2:
            continue

        mask = set(pts)
        primary_masks[color] = mask

        # Draw every aligned pair. This naturally reconstructs hull-like outlines
        # such as rectangles, diamonds, and V-shaped strokes in the examples.
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                if is_aligned(pts[i], pts[j]):
                    draw_8_path(pts[i][0], pts[i][1],
                                pts[j][0], pts[j][1],
                                color, mask)

    # Step 2: infer parallel strokes from isolated adjacent pixels.
    for color, pts in points_by_color.items():
        if len(pts) != 1:
            continue

        r, c = pts[0]
        best_primary_color = None
        best_vector = None
        best_score = -1

        # Look for an 8-neighbor pixel of another color that has a completed stroke.
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    pcolor = int(grid[nr, nc])
                    if pcolor == 0 or pcolor == color:
                        continue
                    if pcolor not in primary_masks:
                        continue
                    # Vector from the primary stroke's neighbor to the seed.
                    v = (r - nr, c - nc)
                    # Prefer larger primary strokes, then smaller color for determinism.
                    score = (len(primary_masks[pcolor]), -pcolor)
                    if score > best_score:
                        best_score = score
                        best_primary_color = pcolor
                        best_vector = v

        if best_primary_color is None:
            continue

        pmask = primary_masks[best_primary_color]
        vr, vc = best_vector

        for pr, pc in pmask:
            nr, nc = pr + vr, pc + vc
            if 0 <= nr < h and 0 <= nc < w:
                # Do not overwrite the original primary stroke or other colors.
                if (nr, nc) in pmask:
                    continue
                if out[nr, nc] == 0 or out[nr, nc] == color:
                    out[nr, nc] = color

    return out