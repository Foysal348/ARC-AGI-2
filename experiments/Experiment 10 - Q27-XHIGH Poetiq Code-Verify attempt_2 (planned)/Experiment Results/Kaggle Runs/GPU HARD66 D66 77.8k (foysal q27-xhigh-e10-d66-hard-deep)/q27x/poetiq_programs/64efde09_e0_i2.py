import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by projecting seed colors outward from rectangular frames.

    Rule implemented:
    1. The background is the most frequent color (8 in the given examples).
    2. Frames are 4-connected non-background components that form a full
       2 x N or N x 2 rectangle.
    3. For each frame, the emitting side is the long side with the fewest
       4s; ties are broken by the stronger dominant side color, then by
       seed alignment if needed.
    4. Single-pixel non-background components are seeds.
    5. A seed gives a coordinate to a frame only if it is aligned with the
       frame's emitting side and the gap between them is clear.
       - For left/right frame sides, the coordinate is the 1-indexed row.
       - For top/bottom frame sides, the coordinate is the 1-indexed column.
    6. Let L be the maximum long-side length among all frames.
       If a color knows only one coordinate, the other is reflected:
           other = L + 1 - known
    7. From each frame's emitting side, draw a ray in the outward direction
       for each known color coordinate. Rays stop at the first non-background
       cell in the current output.
    """
    if grid is None:
        return grid

    grid = np.array(grid, dtype=int, copy=True)
    if grid.ndim != 2 or grid.size == 0:
        return grid

    H, W = grid.shape

    # Background color: most frequent cell value.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    out = grid.copy()

    # Find connected components of non-background cells.
    visited = np.zeros((H, W), dtype=bool)
    frames = []
    seeds = []
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for r in range(H):
        for c in range(W):
            if visited[r, c] or grid[r, c] == bg:
                continue

            q = deque([(r, c)])
            visited[r, c] = True
            cells = []

            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))

                for dr, dc in directions:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < H
                        and 0 <= nc < W
                        and not visited[nr, nc]
                        and grid[nr, nc] != bg
                    ):
                        visited[nr, nc] = True
                        q.append((nr, nc))

            if len(cells) == 1:
                sr, sc = cells[0]
                seeds.append((sr, sc, int(grid[sr, sc])))
            else:
                rmin = min(rr for rr, cc in cells)
                rmax = max(rr for rr, cc in cells)
                cmin = min(cc for rr, cc in cells)
                cmax = max(cc for rr, cc in cells)
                h = rmax - rmin + 1
                w = cmax - cmin + 1

                # Full 2-cell-thick rectangle.
                if min(h, w) == 2 and len(cells) == h * w:
                    frames.append((rmin, cmin, rmax, cmax))

    if not frames or not seeds:
        return out

    # Maximum long side length, used for coordinate reflection.
    L = max(max(rmax - rmin + 1, cmax - cmin + 1) for rmin, cmin, rmax, cmax in frames)

    def side_cells(frame, side):
        """Return the cells along one side of a frame."""
        rmin, cmin, rmax, cmax = frame
        if side == "top":
            return [(rmin, c) for c in range(cmin, cmax + 1)]
        if side == "bottom":
            return [(rmax, c) for c in range(cmin, cmax + 1)]
        if side == "left":
            return [(r, cmin) for r in range(rmin, rmax + 1)]
        if side == "right":
            return [(r, cmax) for r in range(rmin, rmax + 1)]
        return []

    def is_aligned(sr, sc, frame, side):
        """
        Check whether a seed lies directly outside the given frame side
        with a clear background gap.
        """
        rmin, cmin, rmax, cmax = frame

        if side in ("left", "right"):
            if not (rmin <= sr <= rmax):
                return False

            if side == "left":
                if not (sc < cmin):
                    return False
                for cc in range(sc + 1, cmin):
                    if grid[sr, cc] != bg:
                        return False
            else:
                if not (sc > cmax):
                    return False
                for cc in range(cmax + 1, sc):
                    if grid[sr, cc] != bg:
                        return False

            return True

        # top or bottom
        if not (cmin <= sc <= cmax):
            return False

        if side == "top":
            if not (sr < rmin):
                return False
            for rr in range(sr + 1, rmin):
                if grid[rr, sc] != bg:
                    return False
        else:
            if not (sr > rmax):
                return False
            for rr in range(rmax + 1, sr):
                if grid[rr, sc] != bg:
                    return False

        return True

    # Determine the emitting side for each frame.
    active_side = {}

    for frame in frames:
        rmin, cmin, rmax, cmax = frame
        h = rmax - rmin + 1
        w = cmax - cmin + 1

        if h > w:
            candidate_sides = ["left", "right"]
        elif w > h:
            candidate_sides = ["top", "bottom"]
        else:
            candidate_sides = ["left", "right", "top", "bottom"]

        best_score = None
        tied_sides = []

        for side in candidate_sides:
            cells = side_cells(frame, side)
            colors = [grid[rr, cc] for rr, cc in cells]

            if not colors:
                score = (-1, 0, 0)
            else:
                side_vals, side_counts = np.unique(colors, return_counts=True)
                idx = int(np.argmax(side_counts))
                mode_color = int(side_vals[idx])
                mode_count = int(side_counts[idx])

                # The non-emitting side usually has more 4s.
                count4 = sum(1 for x in colors if x == 4)

                # Prefer fewer 4s, then stronger dominance, then lower mode color.
                score = (-count4, mode_count, -mode_color)

            if best_score is None or score > best_score:
                best_score = score
                tied_sides = [side]
            elif score == best_score:
                tied_sides.append(side)

        if len(tied_sides) == 1:
            active_side[frame] = tied_sides[0]
        else:
            # If still tied, choose the side with more aligned seeds.
            best_side = tied_sides[0]
            best_count = -1

            for side in tied_sides:
                aligned_count = sum(
                    1
                    for sr, sc, _ in seeds
                    if is_aligned(sr, sc, frame, side)
                )
                if aligned_count > best_count:
                    best_count = aligned_count
                    best_side = side

            active_side[frame] = best_side

    # Collect direct coordinates from seeds aligned with active sides.
    coord_h = {}
    coord_v = {}
    color_order = []

    def register_color(color):
        if color not in coord_h and color not in coord_v:
            color_order.append(color)

    for sr, sc, color in seeds:
        for frame in frames:
            side = active_side[frame]
            if not is_aligned(sr, sc, frame, side):
                continue

            rmin, cmin, rmax, cmax = frame

            if side in ("left", "right"):
                p = sr - rmin + 1
                if 1 <= p <= L:
                    register_color(color)
                    coord_v.setdefault(color, []).append(p)
            else:
                p = sc - cmin + 1
                if 1 <= p <= L:
                    register_color(color)
                    coord_h.setdefault(color, []).append(p)

    # Resolve one horizontal and one vertical coordinate per seed color.
    coords = {}

    for color in color_order:
        hs = coord_h.get(color, [])
        vs = coord_v.get(color, [])

        h = None
        v = None

        if hs:
            if vs:
                consistent = [p for p in hs if (L + 1 - p) in vs]
                h = consistent[0] if consistent else hs[0]
            else:
                h = hs[0]

        if vs:
            if hs:
                consistent = [p for p in vs if (L + 1 - p) in hs]
                v = consistent[0] if consistent else vs[0]
            else:
                v = vs[0]

        # Reflect missing coordinate.
        if h is None and v is not None:
            h = L + 1 - v
        if v is None and h is not None:
            v = L + 1 - h

        if h is not None or v is not None:
            coords[color] = (h, v)

    # Draw rays from each active side.
    for frame in frames:
        rmin, cmin, rmax, cmax = frame
        side = active_side[frame]
        h_len = rmax - rmin + 1
        w_len = cmax - cmin + 1

        for color in color_order:
            if color not in coords:
                continue

            h_coord, v_coord = coords[color]

            if side in ("left", "right"):
                p = v_coord
                if p is None or not (1 <= p <= h_len):
                    continue

                r = rmin + p - 1

                if side == "left":
                    c = cmin - 1
                    dc = -1
                else:
                    c = cmax + 1
                    dc = 1

                while 0 <= r < H and 0 <= c < W:
                    if out[r, c] != bg:
                        break
                    out[r, c] = color
                    c += dc

            else:
                p = h_coord
                if p is None or not (1 <= p <= w_len):
                    continue

                c = cmin + p - 1

                if side == "top":
                    r = rmin - 1
                    dr = -1
                else:
                    r = rmax + 1
                    dr = 1

                while 0 <= r < H and 0 <= c < W:
                    if out[r, c] != bg:
                        break
                    out[r, c] = color
                    r += dr

    return out