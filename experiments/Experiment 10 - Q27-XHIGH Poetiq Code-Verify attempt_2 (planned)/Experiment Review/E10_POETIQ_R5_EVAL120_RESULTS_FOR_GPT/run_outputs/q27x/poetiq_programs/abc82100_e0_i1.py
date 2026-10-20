import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by identifying the active non-8 patterns, projecting
    them toward the nearest border, and drawing the resulting checkerboard
    projections.

    The implemented rule is:
    1. Color 8 is treated as transparent / erased.
    2. All non-zero, non-8 pixels are grouped into connected components
       using 8-connectivity.
    3. For each component, determine its outward direction based on which
       border it is closest to.
    4. Determine the output color:
       - If there is an opposite-side component, swap the colors.
       - Otherwise, use a singleton color nearby if present.
       - Otherwise, use the component's own color.
    5. Project the component outward by casting rays in the outward direction
       and the two diagonal/along directions, stopping at the border or at
       other component pixels, and fill the reached cells with the chosen
       output color.
    6. Cells occupied by 8 are never written.

    This captures the behavior in the examples:
    - straight side lines become outward checkerboard stripes with swapped
      opposite-side colors;
    - irregular top/bottom shapes become hollow projected checkerboard regions;
    - isolated fill-color markers are used when opposite components are absent.
    """
    if grid.size == 0:
        return np.zeros_like(grid)

    H, W = grid.shape
    out = np.zeros((H, W), dtype=np.uint8)

    # Mask of active pixels: neither background nor the transparent 8.
    active = (grid != 0) & (grid != 8)
    if not active.any():
        return out

    # Connected components, 8-connected.
    comp_id = -1 * np.ones((H, W), dtype=int)
    comps = []
    cur = 0
    dirs8 = [(-1,-1), (-1,0), (-1,1),
             (0,-1),           (0,1),
             (1,-1),  (1,0),   (1,1)]

    for r in range(H):
        for c in range(W):
            if active[r, c] and comp_id[r, c] == -1:
                comp_id[r, c] = cur
                stack = [(r, c)]
                cells = []
                while stack:
                    x, y = stack.pop()
                    cells.append((x, y))
                    for dr, dc in dirs8:
                        nx, ny = x + dr, y + dc
                        if 0 <= nx < H and 0 <= ny < W:
                            if active[nx, ny] and comp_id[nx, ny] == -1:
                                comp_id[nx, ny] = cur
                                stack.append((nx, ny))
                comps.append(cells)
                cur += 1

    # If there is only one component, still process it with its own color.
    for idx, cells in enumerate(comps):
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)
        h = rmax - rmin + 1
        w = cmax - cmin + 1

        # Dominant color of the component.
        colors = [grid[r, c] for r, c in cells]
        color = max(set(colors), key=colors.count)

        # Component centroid.
        cr = sum(rs) / len(rs)
        cc = sum(cs) / len(cs)

        # Determine nearest border / outward direction.
        dist_top = cr
        dist_bottom = (H - 1) - cr
        dist_left = cc
        dist_right = (W - 1) - cc

        dmin = min(dist_top, dist_bottom, dist_left, dist_right)
        if dmin == dist_top:
            side = "top"
            outward = (-1, 0)
        elif dmin == dist_bottom:
            side = "bottom"
            outward = (1, 0)
        elif dmin == dist_left:
            side = "left"
            outward = (0, -1)
        else:
            side = "right"
            outward = (0, 1)

        # Find opposite-side component to swap color, if any.
        out_color = None
        opp_side = {
            "top": "bottom",
            "bottom": "top",
            "left": "right",
            "right": "left"
        }[side]

        # Compute sides for all components quickly.
        def component_side(cells):
            rs2 = [r for r, c in cells]
            cs2 = [c for r, c in cells]
            cr2 = sum(rs2) / len(rs2)
            cc2 = sum(cs2) / len(cs2)
            dt = cr2
            db = (H - 1) - cr2
            dl = cc2
            dr = (W - 1) - cc2
            m = min(dt, db, dl, dr)
            if m == dt:
                return "top"
            if m == db:
                return "bottom"
            if m == dl:
                return "left"
            return "right"

        for other_idx, other_cells in enumerate(comps):
            if other_idx == idx:
                continue
            if component_side(other_cells) == opp_side:
                oc = [grid[r, c] for r, c in other_cells]
                out_color = max(set(oc), key=oc.count)
                break

        # Fallback: use a singleton color if available.
        if out_color is None:
            # Count colors among active pixels.
            from collections import Counter
            cnt = Counter(grid[r, c] for r, c in cells)
            all_active = Counter()
            for r in range(H):
                for c in range(W):
                    if active[r, c]:
                        all_active[grid[r, c]] += 1

            # Prefer a singleton color that is not the main component color.
            for val, freq in all_active.items():
                if freq == 1 and val != color:
                    out_color = val
                    break

        # Final fallback: keep the component's own color.
        if out_color is None:
            out_color = color

        if out_color == 0 or out_color == 8:
            continue

        # Component set for fast lookup.
        comp_set = set(cells)

        # Choose projection rays based on side and shape.
        if side in ("top", "bottom"):
            sign = -1 if side == "top" else 1
            # If the component is roughly horizontal, use along-row rays too.
            if h <= max(2, w // 3):
                rays = [(sign, 0), (0, -1), (0, 1)]
            else:
                # For irregular vertical-ish/top components, use diagonal rays.
                rays = [(sign, 0), (sign, -1), (sign, 1)]
        else:  # left or right
            sign = -1 if side == "left" else 1
            if w <= max(2, h // 3):
                rays = [(0, sign), (-1, 0), (1, 0)]
            else:
                rays = [(0, sign), (-1, sign), (1, sign)]

        # Cast rays from each component pixel.
        for r0, c0 in cells:
            for dr, dc in rays:
                r, c = r0 + dr, c0 + dc
                while 0 <= r < H and 0 <= c < W:
                    # Stop if we hit another pixel of the same component.
                    if (r, c) in comp_set:
                        break
                    # Do not write over 8 cells.
                    if grid[r, c] != 8:
                        out[r, c] = out_color
                    r += dr
                    c += dc

    return out