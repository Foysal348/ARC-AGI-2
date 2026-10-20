import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule
    --------------
    The examples contain:
      * a background color,
      * a dominant contour / wall color,
      * one small "marker" color that supplies the fill color,
      * optional other non-background colors that remain fixed and
        cast a straight shadow away from the filled region.

    The algorithm:
      1. Identify the background color from the grid border.
      2. Identify the main contour color as the most frequent non-background color.
      3. Identify the fill/marker color as the non-contour color whose pixels are
         inside the contour bounding box, or closest to the contour center.
      4. Treat the contour and all other non-background colors as walls.
      5. Close open contour ends by extending their tangent directions to the
         nearest grid border. This makes the intended enclosed basin finite.
      6. Flood-fill from the marker color through background cells, stopping at
         walls.
      7. Other fixed non-background objects cast a straight shadow in the
         dominant direction away from the filled region; shadowed cells are
         left background.
      8. Write the marker color into the filled background cells.
    """
    if grid is None or grid.ndim != 2:
        return grid.copy() if grid is not None else np.array([], dtype=int)

    h, w = grid.shape
    if h == 0 or w == 0:
        return grid.copy()

    # 1) Background: most common border color.
    border = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
    bvals, bcounts = np.unique(border, return_counts=True)
    bg = int(bvals[np.argmax(bcounts)])

    # 2) Main contour: most frequent non-background color.
    vals, counts = np.unique(grid, return_counts=True)
    non_bg = [(int(v), int(c)) for v, c in zip(vals, counts) if v != bg]
    if not non_bg:
        return grid.copy()

    contour = int(max(non_bg, key=lambda x: x[1])[0])
    candidates = [(int(v), int(c)) for v, c in non_bg if v != contour]
    if not candidates:
        return grid.copy()

    # 3) Choose the fill marker color.
    contour_coords = np.argwhere(grid == contour)
    r0, c0 = int(contour_coords[:, 0].min()), int(contour_coords[:, 1].min())
    r1, c1 = int(contour_coords[:, 0].max()), int(contour_coords[:, 1].max())
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0

    def fill_score(color: int):
        coords = np.argwhere(grid == color)
        inside = (
            (coords[:, 0] >= r0) & (coords[:, 0] <= r1) &
            (coords[:, 1] >= c0) & (coords[:, 1] <= c1)
        )
        if np.any(inside):
            dist = float(np.mean(np.hypot(coords[inside, 0] - center_r,
                                          coords[inside, 1] - center_c)))
            return (0, dist, -int(np.count_nonzero(inside)))
        dist = float(np.min(np.hypot(coords[:, 0] - center_r,
                                     coords[:, 1] - center_c)))
        return (1, dist, -int(len(coords)))

    fill = int(min(candidates, key=lambda x: fill_score(x[0]))[0])
    obstacle_colors = [int(v) for v, _ in candidates if v != fill]

    # 4) Walls: contour + opaque obstacles.
    base_walls = (grid == contour)
    for oc in obstacle_colors:
        base_walls |= (grid == oc)

    off8 = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    off4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def flood_fill(walls: np.ndarray) -> np.ndarray:
        mask = np.zeros((h, w), dtype=bool)
        q = deque()
        for r, c in np.argwhere(grid == fill):
            mask[int(r), int(c)] = True
            q.append((int(r), int(c)))
        while q:
            r, c = q.popleft()
            for dr, dc in off4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not walls[nr, nc]:
                    mask[nr, nc] = True
                    q.append((nr, nc))
        return mask

    # Find connected components of the contour (8-connected).
    contour_mask = (grid == contour)
    visited = np.zeros((h, w), dtype=bool)
    components = []

    for start in np.argwhere(contour_mask & ~visited):
        sr, sc = int(start[0]), int(start[1])
        q = deque([(sr, sc)])
        visited[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in off8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and contour_mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    q.append((nr, nc))
        components.append(comp)

    # Pre-fill to see which contour components actually bound the marker region.
    pre_fill = flood_fill(base_walls)

    selected_components = []
    for comp in components:
        touches = False
        for r, c in comp:
            for dr, dc in off8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and pre_fill[nr, nc]:
                    touches = True
                    break
            if touches:
                break
        if touches:
            selected_components.append(comp)

    # Fallback: choose the component closest to the marker if none touched.
    if not selected_components:
        fill_coords = np.argwhere(grid == fill)
        def comp_dist(comp):
            return min(
                float(np.hypot(r - fr, c - fc))
                for (r, c) in comp
                for (fr, fc) in fill_coords
            )
        selected_components = [min(components, key=comp_dist)]

    # 5) Close open ends of selected contour components.
    walls = base_walls.copy()

    def add_wall_line(r: int, c: int, dr: int, dc: int):
        while True:
            r += dr
            c += dc
            if not (0 <= r < h and 0 <= c < w):
                break
            # Do not erase the fill marker itself.
            if grid[r, c] != fill:
                walls[r, c] = True

    def closure_direction(r: int, c: int, vr: int, vc: int):
        """
        Given the direction from the contour neighbor to the endpoint,
        choose a direction to extend toward a grid border.
        Diagonal tangents are kept when they naturally hit a nearby border.
        """
        if vr == 0 and vc == 0:
            dists = {
                "T": r,
                "B": h - 1 - r,
                "L": c,
                "R": w - 1 - c
            }
            k = min(dists, key=dists.get)
            return {
                "T": (-1, 0),
                "B": (1, 0),
                "L": (0, -1),
                "R": (0, 1)
            }[k]

        # Preserve diagonal endpoint tangents when they hit a nearest border.
        if vr != 0 and vc != 0:
            nearest_border = min(r, h - 1 - r, c, w - 1 - c)
            hit = []
            if vr < 0:
                hit.append(r)
            if vr > 0:
                hit.append(h - 1 - r)
            if vc < 0:
                hit.append(c)
            if vc > 0:
                hit.append(w - 1 - c)
            if hit and min(hit) <= nearest_border + 1:
                return (1 if vr > 0 else -1, 1 if vc > 0 else -1)

        # Otherwise extend toward the nearest border.
        options = [
            ("T", r,       -1, 0),
            ("B", h - 1 - r, 1, 0),
            ("L", c,        0, -1),
            ("R", w - 1 - c, 0, 1),
        ]
        _, _, dr, dc = min(
            options,
            key=lambda t: (t[1], -(vr * t[2] + vc * t[3]))
        )
        return dr, dc

    for comp in selected_components:
        comp_set = set(comp)

        endpoints = []
        for r, c in comp:
            degree = 0
            for dr, dc in off8:
                nr, nc = r + dr, c + dc
                if (nr, nc) in comp_set:
                    degree += 1
            if degree <= 1:
                endpoints.append((r, c))

        for r, c in endpoints:
            neighbors = []
            for dr, dc in off8:
                nr, nc = r + dr, c + dc
                if (nr, nc) in comp_set:
                    neighbors.append((nr, nc))

            if neighbors:
                nr, nc = neighbors[0]
                vr, vc = r - nr, c - nc
            else:
                vr, vc = 0, 0

            dr, dc = closure_direction(r, c, vr, vc)
            if dr != 0 or dc != 0:
                add_wall_line(r, c, dr, dc)

    # 6) Final flood fill in the closed basin.
    filled = flood_fill(walls)

    # 7) Straight shadows from opaque obstacle colors.
    if filled.any():
        filled_coords = np.argwhere(filled)
        cy = float(filled_coords[:, 0].mean())
        cx = float(filled_coords[:, 1].mean())
        min_r, max_r = int(filled_coords[:, 0].min()), int(filled_coords[:, 0].max())
        min_c, max_c = int(filled_coords[:, 1].min()), int(filled_coords[:, 1].max())

        for oc in obstacle_colors:
            for r, c in np.argwhere(grid == oc):
                r, c = int(r), int(c)
                if not (min_r <= r <= max_r and min_c <= c <= max_c):
                    continue

                dr = float(r - cy)
                dc = float(c - cx)

                if abs(dr) >= abs(dc) and dr != 0:
                    sr, sc = (1 if dr > 0 else -1), 0
                elif dc != 0:
                    sr, sc = 0, (1 if dc > 0 else -1)
                else:
                    continue

                nr, nc = r + sr, c + sc
                while 0 <= nr < h and 0 <= nc < w and not walls[nr, nc]:
                    if filled[nr, nc]:
                        filled[nr, nc] = False
                    nr += sr
                    nc += sc

    # 8) Apply the fill color.
    out = grid.copy()
    out[filled] = fill
    return out