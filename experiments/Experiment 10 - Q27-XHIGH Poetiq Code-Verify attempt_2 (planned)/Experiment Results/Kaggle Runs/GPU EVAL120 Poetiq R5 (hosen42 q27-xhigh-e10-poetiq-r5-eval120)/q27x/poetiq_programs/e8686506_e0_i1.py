import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered colored pieces into the bounding box of the largest
    connected non-background component(s).

    The largest component(s) are treated as a fixed frame. All other connected
    components are translated into empty cells inside that frame, without
    rotation. Pieces are placed largest-first; for each piece, valid placements
    are ordered by translation distance, with a secondary preference for
    preserving the piece's side relative to the frame center. Remaining empty
    cells keep the input background color.
    """
    class Component:
        __slots__ = (
            "color", "cells", "top_r", "top_c", "bottom_r", "bottom_c",
            "height", "width", "area", "offs"
        )

        def __init__(self, color, cells):
            self.color = int(color)
            self.cells = cells
            rs = [r for r, c in cells]
            cs = [c for r, c in cells]

            self.top_r = min(rs)
            self.top_c = min(cs)
            self.bottom_r = max(rs)
            self.bottom_c = max(cs)

            self.height = self.bottom_r - self.top_r + 1
            self.width = self.bottom_c - self.top_c + 1
            self.area = len(cells)
            self.offs = [(r - self.top_r, c - self.top_c) for r, c in cells]

    if grid is None:
        return np.array([])

    grid = np.asarray(grid)
    if grid.ndim != 2 or grid.size == 0:
        return grid.copy()

    H, W = grid.shape

    # Background is the most frequent color.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    nonbg = grid != bg
    if not nonbg.any():
        return grid.copy()

    # Extract 8-connected components of each non-background color.
    labels = np.zeros((H, W), dtype=int)
    comps = []
    comp_id = 0

    for r in range(H):
        for c in range(W):
            if nonbg[r, c] and labels[r, c] == 0:
                color = int(grid[r, c])
                comp_id += 1
                labels[r, c] = comp_id
                stack = [(r, c)]
                cells = []

                while stack:
                    cr, cc = stack.pop()
                    cells.append((cr, cc))

                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if (
                                0 <= nr < H and 0 <= nc < W
                                and nonbg[nr, nc]
                                and grid[nr, nc] == color
                                and labels[nr, nc] == 0
                            ):
                                labels[nr, nc] = comp_id
                                stack.append((nr, nc))

                comps.append(Component(color, cells))

    if not comps:
        return grid.copy()

    def side_cost(in_d, out_d):
        """Small penalty when the output side relative to center differs."""
        if in_d == 0 or out_d == 0:
            return 0 if in_d == 0 or out_d == 0 else 1
        return 0 if (in_d > 0) == (out_d > 0) else 2

    def assemble(frame):
        """Try to build the output using the given components as fixed frame."""
        frame_ids = {id(c) for c in frame}

        r0 = min(c.top_r for c in frame)
        c0 = min(c.top_c for c in frame)
        r1 = max(c.bottom_r for c in frame)
        c1 = max(c.bottom_c for c in frame)

        ch = r1 - r0 + 1
        cw = c1 - c0 + 1

        out = np.full((ch, cw), bg, dtype=grid.dtype)

        # Place fixed frame components.
        for comp in frame:
            for r, c in comp.cells:
                out[r - r0, c - c0] = comp.color

        pieces = [c for c in comps if id(c) not in frame_ids]
        pieces.sort(key=lambda x: (-x.area, x.top_r, x.top_c, x.color))

        def valid_placements(comp):
            res = []

            # Doubled differences of centers from canvas center.
            in_dy2 = 2 * comp.top_r + comp.height - 1 - (2 * r0 + ch - 1)
            in_dx2 = 2 * comp.top_c + comp.width - 1 - (2 * c0 + cw - 1)

            for R in range(ch - comp.height + 1):
                for C in range(cw - comp.width + 1):
                    ok = True
                    for dr, dc in comp.offs:
                        if out[R + dr, C + dc] != bg:
                            ok = False
                            break

                    if not ok:
                        continue

                    OR = R + r0
                    OC = C + c0

                    dist = (OR - comp.top_r) ** 2 + (OC - comp.top_c) ** 2

                    out_dy2 = 2 * R + comp.height - ch
                    out_dx2 = 2 * C + comp.width - cw

                    sc = side_cost(in_dy2, out_dy2) + side_cost(in_dx2, out_dx2)

                    res.append((dist, sc, R, C))

            return res

        def solve(idx):
            if idx == len(pieces):
                return True

            comp = pieces[idx]
            placements = valid_placements(comp)
            placements.sort()

            for _, _, R, C in placements:
                for dr, dc in comp.offs:
                    out[R + dr, C + dc] = comp.color

                if solve(idx + 1):
                    return True

                for dr, dc in comp.offs:
                    out[R + dr, C + dc] = bg

            return False

        if solve(0):
            return out
        return None

    # Primary frame: all largest-area components.
    max_area = max(c.area for c in comps)
    frame1 = [c for c in comps if c.area == max_area]
    out = assemble(frame1)
    if out is not None:
        return out

    # Fallback 1: largest bounding-box components.
    max_bbox = max(c.height * c.width for c in comps)
    frame2 = [c for c in comps if c.height * c.width == max_bbox]
    if any(id(c) not in {id(f) for f in frame1} for c in frame2):
        out = assemble(frame2)
        if out is not None:
            return out

    # Fallback 2: single largest bounding-box component.
    frame3 = [max(comps, key=lambda c: (c.height * c.width, c.area))]
    if any(id(c) not in {id(f) for f in frame1} for c in frame3):
        out = assemble(frame3)
        if out is not None:
            return out

    # Final graceful fallback: show the primary frame only.
    r0 = min(c.top_r for c in frame1)
    c0 = min(c.top_c for c in frame1)
    r1 = max(c.bottom_r for c in frame1)
    c1 = max(c.bottom_c for c in frame1)
    ch = r1 - r0 + 1
    cw = c1 - c0 + 1
    out = np.full((ch, cw), bg, dtype=grid.dtype)

    for comp in frame1:
        for r, c in comp.cells:
            out[r - r0, c - c0] = comp.color

    return out