import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    ARC-style wall removal / foreground completion.

    The input contains:
      - one "wall" color that forms lines and a solid 3x3 junction,
      - one background color,
      - one or more foreground colors forming colored regions.

    The output is produced by:
      1. Detecting the wall color and the central solid 3x3 wall block.
      2. Keeping only the outer 8 cells of that 3x3 block as wall.
      3. Replacing all other wall cells by the nearest foreground region color.
         Background is ignored as a fill source.
      4. Filling the center of the kept 3x3 ring with the dominant foreground
         color (by kept source area, then total frequency).

    Foreground components that are too small are ignored, except that components
    touching the wall network are allowed to be a bit smaller.
    """
    g = np.array(grid, dtype=int)
    if g.ndim != 2:
        return g.copy()

    H, W = g.shape
    if H < 3 or W < 3:
        return g.copy()

    colors = np.unique(g)
    counts = {int(c): int(np.count_nonzero(g == c)) for c in colors}
    most_frequent_color = max(counts, key=lambda c: (counts[c], c))

    # ---------------------------------------------------------------
    # 1. Label 4-connected components of every color.
    # ---------------------------------------------------------------
    label = -np.ones((H, W), dtype=int)
    comp_stats = []

    dirs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(H):
        for c in range(W):
            if label[r, c] != -1:
                continue

            cid = len(comp_stats)
            color = int(g[r, c])
            q = deque([(r, c)])
            label[r, c] = cid

            area = 0
            min_r = max_r = r
            min_c = max_c = c

            while q:
                cr, cc = q.popleft()
                area += 1
                min_r = min(min_r, cr)
                max_r = max(max_r, cr)
                min_c = min(min_c, cc)
                max_c = max(max_c, cc)

                for dr, dc in dirs4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if label[nr, nc] == -1 and g[nr, nc] == color:
                            label[nr, nc] = cid
                            q.append((nr, nc))

            bbox_area = (max_r - min_r + 1) * (max_c - min_c + 1)
            density = area / bbox_area if bbox_area else 1.0
            comp_stats.append({
                "color": color,
                "area": area,
                "min_r": min_r,
                "max_r": max_r,
                "min_c": min_c,
                "max_c": max_c,
                "bbox_area": bbox_area,
                "density": density,
            })

    # ---------------------------------------------------------------
    # 2. Find 3x3 monochrome blocks.  Prefer blocks that are not
    #    contained in a larger 4x4 monochrome block, because the central
    #    wall junction is a 3x3 solid block rather than a large filled area.
    # ---------------------------------------------------------------
    blocks = []
    for r in range(H - 2):
        for c in range(W - 2):
            v = g[r, c]
            if not np.all(g[r:r + 3, c:c + 3] == v):
                continue

            in_4x4 = False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    r2 = r + dr
                    c2 = c + dc
                    if 0 <= r2 and 0 <= c2 and r2 + 4 <= H and c2 + 4 <= W:
                        if np.all(g[r2:r2 + 4, c2:c2 + 4] == v):
                            in_4x4 = True
                            break
                if in_4x4:
                    break

            blocks.append((r, c, int(v), not in_4x4))

    if not blocks:
        # Fallback: no usable 3x3 junction.  Do nothing.
        return g.copy()

    # Score candidate junctions:
    #   - prefer maximal 3x3 blocks
    #   - prefer larger connected-component bounding box
    #   - prefer lower density (thin wall network over filled region)
    #   - prefer closer to grid center
    #   - prefer non-most-frequent color only as a final tie-breaker
    best_key = None
    best_block = None
    center_r = (H - 1) / 2.0
    center_c = (W - 1) / 2.0

    for r, c, v, is_max in blocks:
        cid = label[r, c]
        st = comp_stats[cid]
        dist_to_center = abs((r + 1) - center_r) + abs((c + 1) - center_c)

        key = (
            0 if is_max else 1,
            -st["bbox_area"],
            st["density"],
            dist_to_center,
            0 if v != most_frequent_color else 1,
            -counts.get(v, 0),
        )

        if best_key is None or key < best_key:
            best_key = key
            best_block = (r, c, v, is_max)

    if best_block is None:
        return g.copy()

    junction_r, junction_c, wall_color, _ = best_block

    # ---------------------------------------------------------------
    # 3. Determine background color: most frequent color other than wall.
    # ---------------------------------------------------------------
    non_wall_counts = {
        c: n for c, n in counts.items() if c != wall_color
    }
    if not non_wall_counts:
        background_color = wall_color
    else:
        max_n = max(non_wall_counts.values())
        tied = [c for c, n in non_wall_counts.items() if n == max_n]
        # If tied, choose the color with the largest component bbox.
        def bg_score(c):
            max_bb = 0
            for st in comp_stats:
                if st["color"] == c:
                    max_bb = max(max_bb, st["bbox_area"])
            return (max_bb, counts[c])

        background_color = max(tied, key=bg_score)

    foreground_colors = [
        c for c in counts
        if c != wall_color and c != background_color
    ]

    # ---------------------------------------------------------------
    # 4. Choose valid foreground source components.
    #
    # Components of size >= 3 are kept.  Components of size 2 are kept
    # only if they touch the wall network.  This ignores tiny noise dots
    # while still allowing small legitimate wall-adjacent regions.
    # ---------------------------------------------------------------
    source_mask = np.zeros((H, W), dtype=bool)
    source_area = {c: 0 for c in foreground_colors}

    for cid, st in enumerate(comp_stats):
        if st["color"] not in source_area:
            continue

        color = st["color"]
        area = st["area"]

        keep = False
        if area >= 3:
            keep = True
        elif area == 2:
            # Check adjacency to wall color.
            cells = np.argwhere(label == cid)
            for r, c in cells:
                for dr, dc in dirs4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and g[nr, nc] == wall_color:
                        keep = True
                        break
                if keep:
                    break

        if keep:
            source_mask |= (label == cid)
            source_area[color] += area

    # If there are no valid foreground sources, use all foreground cells.
    if not source_mask.any() and foreground_colors:
        for c in foreground_colors:
            source_mask |= (g == c)
            source_area[c] = counts[c]

    # ---------------------------------------------------------------
    # 5. Multi-source distance maps for each foreground color.
    # ---------------------------------------------------------------
    dist_maps = {}
    for color in foreground_colors:
        src = source_mask & (g == color)
        if not src.any():
            continue

        dist = np.full((H, W), 10**9, dtype=int)
        q = deque()

        for r, c in np.argwhere(src):
            dist[r, c] = 0
            q.append((int(r), int(c)))

        while q:
            r, c = q.pop()
            nd = dist[r, c] + 1
            for dr, dc in dirs4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and dist[nr, nc] > nd:
                    dist[nr, nc] = nd
                    q.append((nr, nc))

        dist_maps[color] = dist

    # Tie-break score for colors: kept source area first, then total count.
    def color_score(c):
        return (source_area.get(c, 0), counts.get(c, 0))

    # ---------------------------------------------------------------
    # 6. Build output.
    # ---------------------------------------------------------------
    out = g.copy()

    junction_r0, junction_c0 = junction_r, junction_c
    central = np.zeros((H, W), dtype=bool)
    central[junction_r0:junction_r0 + 3, junction_c0:junction_c0 + 3] = True

    # Replace all non-central wall cells by nearest foreground color.
    for r in range(H):
        for c in range(W):
            if g[r, c] != wall_color or central[r, c]:
                continue

            best_color = None
            best_dist = 10**9
            best_score = (-1, -1)

            for color, dist in dist_maps.items():
                d = dist[r, c]
                sc = color_score(color)
                if d < best_dist or (d == best_dist and sc > best_score):
                    best_dist = d
                    best_color = color
                    best_score = sc

            if best_color is not None:
                out[r, c] = best_color
            else:
                out[r, c] = background_color

    # Keep the central 3x3 as a ring.
    out[junction_r0:junction_r0 + 3, junction_c0:junction_c0 + 3] = wall_color

    # Fill the center of the ring with the dominant foreground color.
    center_r = junction_r0 + 1
    center_c = junction_c0 + 1

    if foreground_colors:
        center_color = max(foreground_colors, key=color_score)
    else:
        center_color = background_color

    out[center_r, center_c] = center_color

    return out