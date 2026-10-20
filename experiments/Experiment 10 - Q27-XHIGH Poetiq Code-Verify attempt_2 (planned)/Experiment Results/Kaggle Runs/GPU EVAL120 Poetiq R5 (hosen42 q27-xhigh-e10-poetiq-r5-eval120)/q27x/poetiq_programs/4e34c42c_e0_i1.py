import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Extract the unique vertically symmetric glyphs from the input scene,
    merge fragments that share the same core signature, remove duplicates,
    and return them as a single horizontally concatenated palette.

    The rule used by the examples:
      1. The most frequent color is background.
      2. A core is a non-background pixel whose 3x3 neighborhood is
         vertically symmetric and has a dominant non-background ring color.
      3. Cores with the same (center_color, ring_color) signature belong to
         the same glyph.  They are aligned at their core pixels and merged by
         taking the union of their local 5x5 neighborhoods.
      4. Each merged glyph is cropped to its non-background bounding box.
      5. Duplicate glyphs are removed.
      6. Glyphs are vertically centered on a 5-row canvas and concatenated
         horizontally.
    """
    grid = np.asarray(grid, dtype=np.int64)
    if grid.size == 0:
        return grid

    h, w = grid.shape

    # Background is the most frequent color.
    vals, counts = np.unique(grid, return_counts=True)
    background = vals[int(np.argmax(counts))]

    # Gather all candidate cores.
    cores = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center = grid[r, c]
            if center == background:
                continue

            top = grid[r - 1, c - 1:c + 2]
            bot = grid[r + 1, c - 1:c + 2]
            if not np.array_equal(top, bot):
                continue

            ring = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    v = grid[r + dr, c + dc]
                    if v != background:
                        ring.append(v)

            if not ring:
                continue

            # Dominant non-background color in the 8-neighborhood.
            ring_vals, ring_counts = np.unique(np.array(ring), return_counts=True)
            best_count = int(ring_counts.max())
            best_vals = ring_vals[ring_counts == best_count]
            ring_color = int(best_vals[0])

            # Local 5x5 neighborhood colors define the palette of this glyph.
            r0, r1 = max(0, r - 2), min(h, r + 3)
            c0, c1 = max(0, c - 2), min(w, c + 3)
            local = grid[r0:r1, c0:c1]
            palette = [int(x) for x in np.unique(local) if int(x) != background]
            if not palette:
                continue

            cores.append((center, ring_color, r, c, palette))

    # Merge all cores having the same signature.
    groups = {}
    for center, ring_color, r, c, palette in cores:
        key = (int(center), int(ring_color))
        groups.setdefault(key, []).append((r, c, palette))

    glyphs = []
    seen = set()

    for key, members in groups.items():
        union = None
        palette = set()

        for r, c, pal in members:
            r0, r1 = max(0, r - 2), min(h, r + 3)
            c0, c1 = max(0, c - 2), min(w, c + 3)
            patch = np.full((5, 5), background, dtype=np.int64)
            patch[2 - (r - r0):2 + (r + 3 - r0), 2 - (c - c0):2 + (c + 3 - c0)] = grid[r0:r1, c0:c1]
            palette.update(pal)

            if union is None:
                union = patch.copy()
            else:
                # Union non-background cells.  If two different colors occupy
                # the same relative position, keep the one from the later patch;
                # the examples only require the stable visible shape.
                nonbg_patch = patch != background
                union = np.where(nonbg_patch, patch, union)

        # Crop to non-background bounding box.
        mask = union != background
        if not mask.any():
            continue
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        glyph = union[rmin:rmax + 1, cmin:cmax + 1]

        # Canonical representation for duplicate removal.
        canon = tuple(map(tuple, glyph.tolist()))
        if canon in seen:
            continue
        seen.add(canon)
        glyphs.append(glyph)

    if not glyphs:
        return np.zeros_like(grid, dtype=np.int64)

    # Sort for a deterministic output order:
    #   smaller glyphs first, then by top-left-most non-background color.
    def sort_key(g):
        nonbg = g[g != background]
        first_color = int(nonbg.flat[0]) if nonbg.size else 0
        return (g.shape[0] * g.shape[1], g.shape[0], g.shape[1], first_color)

    glyphs.sort(key=sort_key)

    # Place each glyph vertically centered on a 5-row canvas.
    out_h = 5
    out_w = sum(g.shape[1] for g in glyphs)
    out = np.full((out_h, out_w), background, dtype=np.int64)

    col = 0
    for g in glyphs:
        gh, gw = g.shape
        top = (out_h - gh) // 2
        out[top:top + gh, col:col + gw] = g
        col += gw

    return out