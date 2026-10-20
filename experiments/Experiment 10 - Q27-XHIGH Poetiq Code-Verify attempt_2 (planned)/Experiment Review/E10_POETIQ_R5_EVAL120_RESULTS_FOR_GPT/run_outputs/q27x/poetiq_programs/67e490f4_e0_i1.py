import numpy as np
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid into the compact pattern defined by the large
    rectangular mask in the input.

    Rule implemented
    ----------------
    1. The input contains one large rectangular "mask" region made of a solid
       mask color with holes of the background color.
    2. The output is exactly that mask rectangle.
    3. Each connected hole region inside the mask is filled with a color
       chosen from the non-background/non-mask objects outside the mask.
       For each hole component:
         - First, try to match it with outside components of the same exact
           shape, allowing rotations/reflections.  If there is a match, use the
           closest matching component's color.
         - If there is no exact shape match, use the closest outside component
           of any shape.
    """
    grid = np.asarray(grid)
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape

    # Background is the border color.
    bg = grid[0, 0]

    # Count non-background colors.
    non_bg = np.unique(grid[grid != bg])
    if len(non_bg) == 0:
        return np.zeros_like(grid)

    # Find connected components of non-background colors.
    visited = np.zeros((h, w), dtype=bool)

    def flood(mask_color, start_r, start_c):
        stack = [(start_r, start_c)]
        comp = []
        visited[start_r, start_c] = True
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == mask_color:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        return comp

    comps = []
    for color in non_bg:
        mask_color = color
        sub = (grid == mask_color) & (~visited)
        rs, cs = np.where(sub)
        for r0, c0 in zip(rs, cs):
            if not visited[r0, c0]:
                comp = flood(mask_color, r0, c0)
                if comp:
                    comps.append((mask_color, comp))

    if not comps:
        return np.zeros_like(grid)

    # The mask component is the largest non-background component.
    mask_color, mask_comp = max(comps, key=lambda x: len(x[1]))
    rs = [r for r, _ in mask_comp]
    cs = [c for _, c in mask_comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)

    # Extract mask rectangle.
    mask_rect = grid[r0:r1+1, c0:c1+1]

    # Outside components: non-background, not mask color.
    outside_comps = []
    for color, comp in comps:
        if color == mask_color:
            continue
        # Exclude components that are completely inside the mask rectangle.
        if all(r0 <= r <= r1 and c0 <= c <= c1 for r, c in comp):
            continue
        outside_comps.append((color, comp))

    # Normalize a component to a canonical tuple under rotations/reflections.
    def canonical_shapes(comp):
        pts = list(comp)
        min_r = min(r for r, _ in pts)
        min_c = min(c for _, c in pts)
        norm = {(r - min_r, c - min_c) for r, c in pts}

        shapes = set()
        cur = list(norm)
        for _ in range(4):
            cur = [(c, r) for r, c in cur]  # rotate 90
            min_r = min(r for r, _ in cur)
            min_c = min(c for _, c in cur)
            norm = frozenset((r - min_r, c - min_c) for r, c in cur)
            shapes.add(norm)

            # reflect
            cur = [(-r, c) for r, c in cur]
            min_r = min(r for r, _ in cur)
            min_c = min(c for _, c in cur)
            norm = frozenset((r - min_r, c - min_c) for r, c in cur)
            shapes.add(norm)
        return shapes

    def centroid(comp):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        return (sum(rs) / len(rs), sum(cs) / len(cs))

    def dist(c1, c2):
        return (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2

    # Precompute outside component shapes and centroids.
    outside_info = []
    for color, comp in outside_comps:
        outside_info.append({
            "color": color,
            "comp": comp,
            "centroid": centroid(comp),
            "shapes": canonical_shapes(comp),
        })

    # Find hole components inside the mask rectangle.
    holes = (mask_rect == bg)
    if not np.any(holes):
        return mask_rect.copy()

    visited_h = np.zeros(holes.shape, dtype=bool)
    hole_comps = []
    H, W = holes.shape

    for i in range(H):
        for j in range(W):
            if holes[i, j] and not visited_h[i, j]:
                stack = [(i, j)]
                comp = []
                visited_h[i, j] = True
                while stack:
                    r, c = stack.pop()
                    comp.append((r, c))
                    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < H and 0 <= nc < W and holes[nr, nc] and not visited_h[nr, nc]:
                            visited_h[nr, nc] = True
                            stack.append((nr, nc))
                hole_comps.append(comp)

    # Fill output.
    out = mask_rect.copy()

    for comp in hole_comps:
        cent = centroid(comp)
        shapes = canonical_shapes(comp)

        best_color = None
        best_d = None

        # 1) exact shape match, choose closest matching outside component
        for info in outside_info:
            if shapes & info["shapes"]:
                d = dist(cent, info["centroid"])
                if best_d is None or d < best_d:
                    best_d = d
                    best_color = info["color"]

        # 2) if no exact match, choose closest outside component of any shape
        if best_color is None:
            for info in outside_info:
                d = dist(cent, info["centroid"])
                if best_d is None or d < best_d:
                    best_d = d
                    best_color = info["color"]

        if best_color is None:
            best_color = bg

        for r, c in comp:
            out[r, c] = best_color

    return out