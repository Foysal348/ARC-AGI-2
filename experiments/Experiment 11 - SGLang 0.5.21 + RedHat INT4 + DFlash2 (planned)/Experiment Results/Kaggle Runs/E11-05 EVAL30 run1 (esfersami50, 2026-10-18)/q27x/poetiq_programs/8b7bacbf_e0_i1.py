import numpy as np
from collections import deque, Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill the interiors of closed rooms that lie in the same bounded region as the
    unique key cell.  The key cell is the unique non-background cell with the
    largest color value.  Background is the most common color.

    The implementation is intentionally conservative:
      - detect all finite background components;
      - detect "rooms" = small enclosed background components;
      - estimate the outer boundary of the key region from the source wall color
        adjacent to the key;
      - fill rooms whose centers are inside that estimated region, or as a fallback
        the majority side of the source wall diagonal.
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape
    orig = grid.copy()

    # ---- 1. background color ---------------------------------------------
    colors = Counter(int(c) for row in grid for c in row)
    bg = colors.most_common(1)[0][0]

    # ---- 2. key color -----------------------------------------------------
    # Prefer colors that appear exactly once; if several, choose largest value.
    singleton = [c for c, n in colors.items() if n == 1 and c != bg]
    if singleton:
        key_color = max(singleton)
    else:
        # fallback: least frequent non-background color, then largest value
        nonbg = [(c, n) for c, n in colors.items() if c != bg]
        if not nonbg:
            return orig.copy()
        min_count = min(n for _, n in nonbg)
        key_color = max(c for c, n in nonbg if n == min_count)

    key_cells = [(r, c) for r in range(H) for c in range(W) if grid[r, c] == key_color]
    if not key_cells:
        return orig.copy()

    # If there are multiple key cells, use the one with highest coordinate sum
    # (consistent with examples where key is often at a corner/end).
    key_r, key_c = max(key_cells, key=lambda p: (p[0], p[1], p[0] + p[1]))

    # ---- 3. source wall color adjacent to key ----------------------------
    neighbors = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        r, c = key_r + dr, key_c + dc
        if 0 <= r < H and 0 <= c < W:
            col = grid[r, c]
            if col != bg and col != key_color:
                neighbors.append((col, abs(dr) + abs(dc), -r, -c))
    if neighbors:
        # choose most frequent adjacent non-bg color; tie by larger value
        adj_counts = Counter(col for col, *_ in neighbors)
        source_color = max(adj_counts, key=lambda c: (adj_counts[c], c))
    else:
        # fallback: most frequent non-bg/non-key color in grid
        other = [c for c in colors if c != bg and c != key_color]
        if not other:
            return orig.copy()
        source_color = max(other, key=lambda c: (colors[c], c))

    # ---- 4. connected components of background ---------------------------
    parent = [[i, j] for i in range(H) for j in range(W)]
    def find(x):
        while parent[x[0]][x[1]] != x:
            parent[x[0]][x[1]] = parent[parent[x[0]][x[1]][0]][parent[x[0]][x[1]][1]]
            x = parent[x[0]][x[1]]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra[0] < rb[0] or (ra[0] == rb[0] and ra[1] < rb[1]):
                parent[rb[0]][rb[1]] = ra
            else:
                parent[ra[0]][ra[1]] = rb

    for r in range(H):
        for c in range(W):
            if grid[r, c] != bg:
                continue
            if r + 1 < H and grid[r + 1, c] == bg:
                union((r, c), (r + 1, c))
            if c + 1 < W and grid[r, c + 1] == bg:
                union((r, c), (r, c + 1))

    comp_id = {}
    comp_cells = {}
    idx = 0
    for r in range(H):
        for c in range(W):
            if grid[r, c] == bg:
                root = find((r, c))
                if root not in comp_id:
                    comp_id[root] = idx
                    comp_cells[idx] = []
                    idx += 1
                comp_cells[comp_id[root]].append((r, c))

    comp_of = np.full((H, W), -1, dtype=int)
    for root, cid in comp_id.items():
        for r, c in comp_cells[cid]:
            comp_of[r, c] = cid

    # boundary-touching components are "outside"
    boundary_comps = set()
    for r, c in comp_cells.keys():
        pass
    for cid, cells in comp_cells.items():
        for r, c in cells:
            if r == 0 or c == 0 or r == H - 1 or c == W - 1:
                boundary_comps.add(cid)
                break

    # ---- 5. estimate source region ---------------------------------------
    # Source wall pixels: color == source_color, plus key cell if useful.
    src_pixels = [(r, c) for r in range(H) for c in range(W) if grid[r, c] == source_color]
    if key_color != source_color:
        src_pixels.append((key_r, key_c))

    if not src_pixels:
        # fallback: fill all enclosed small background components
        fill = np.zeros_like(grid)
        for cid, cells in comp_cells.items():
            if cid in boundary_comps:
                continue
            if len(cells) <= max(4, int(0.05 * H * W)):
                for r, c in cells:
                    fill[r, c] = key_color
        return (orig + fill).astype(orig.dtype, copy=False)

    # Bounding box of source wall + key
    rs = [p[0] for p in src_pixels]
    cs = [p[1] for p in src_pixels]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)

    # We use a generous bounding box plus half the diagonal length, because the
    # source wall often only traces a diagonal boundary of a larger region.
    diag = max(abs(rmax - rmin), abs(cmax - cmin))
    pad = max(2, diag // 2)
    rmin = max(0, rmin - pad)
    rmax = min(H - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(W - 1, cmax + pad)

    def inside_box_center(r, c):
        return rmin <= r <= rmax and cmin <= c <= cmax

    # Optional side test if source wall forms a diagonal-ish barrier.
    # Use the two extreme source pixels: one near key, one farthest from key.
    far_src = max(src_pixels, key=lambda p: abs(p[0] - key_r) + abs(p[1] - key_c))
    near_src = min(src_pixels, key=lambda p: abs(p[0] - key_r) + abs(p[1] - key_c))

    # direction vector from far to near (towards key)
    dr = near_src[0] - far_src[0]
    dc = near_src[1] - far_src[1]

    def side_sign(r, c):
        # cross product of (far->near) with (far->point)
        return dr * (c - far_src[1]) - dc * (r - far_src[0])

    # Determine majority side among source wall? Use source pixels offset slightly
    side_signs = []
    for r, c in src_pixels:
        s = side_sign(r, c)
        if s != 0:
            side_signs.append(1 if s > 0 else -1)
    if side_signs:
        majority_side = 1 if side_signs.count(1) > side_signs.count(-1) else -1
    else:
        majority_side = 0

    def passes_side(r, c):
        if majority_side == 0:
            return True
        s = side_sign(r, c)
        if s == 0:
            return True
        return (1 if s > 0 else -1) == majority_side

    # ---- 6. select enclosed components to fill ---------------------------
    out = orig.copy()
    filled_any = False

    max_room_size = max(16, int(0.15 * H * W))
    for cid, cells in comp_cells.items():
        if cid in boundary_comps:
            continue
        if len(cells) > max_room_size:
            continue
        # center of component
        mr = sum(r for r, _ in cells) / len(cells)
        mc = sum(c for _, c in cells) / len(cells)

        # A room is acceptable if its center is inside the source bbox and
        # on the chosen side (if any).
        if inside_box_center(int(round(mr)), int(round(mc))) and passes_side(mr, mc):
            for r, c in cells:
                out[r, c] = key_color
            filled_any = True

    # Fallback: if nothing filled, fill all small enclosed components that are
    # inside the source bounding box (without side test).
    if not filled_any:
        for cid, cells in comp_cells.items():
            if cid in boundary_comps:
                continue
            if len(cells) > max_room_size:
                continue
            mr = sum(r for r, _ in cells) / len(cells)
            mc = sum(c for _, c in cells) / len(cells)
            if inside_box_center(int(round(mr)), int(round(mc))):
                for r, c in cells:
                    out[r, c] = key_color
                filled_any = True

    # Final fallback: fill all small enclosed components.
    if not filled_any:
        for cid, cells in comp_cells.items():
            if cid in boundary_comps:
                continue
            if len(cells) <= max(4, int(0.08 * H * W)):
                for r, c in cells:
                    out[r, c] = key_color

    return out