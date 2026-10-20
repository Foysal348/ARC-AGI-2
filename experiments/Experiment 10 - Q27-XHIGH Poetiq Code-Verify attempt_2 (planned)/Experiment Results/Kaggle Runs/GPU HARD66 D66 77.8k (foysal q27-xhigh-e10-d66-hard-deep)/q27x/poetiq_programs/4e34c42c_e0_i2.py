import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Extract a fixed set of symmetric object templates from the input, remove
    duplicates, and arrange the remaining objects in one horizontal strip.

    Rule implemented:
      1. The background is the most frequent color.
      2. Non-background pixels are split into 4-connected components.
      3. Inside each component, detect these templates:
           - 1xN horizontal line, N >= 3
           - 3x3 ring:
                 A A A
                 A B A
                 A A A
           - 3x3 checker:
                 A B A
                 B A B
                 A B A
           - 3x3 H-shape:
                 A . A
                 A A A
                 A . A
           - 5x3 block with right vertical stripe:
                 A A A
                 A A B
                 A A B
                 A A B
                 A A A
           - 5x3 framed left-bar object:
                 A . C
                 A D D
                 A D E
                 A D D
                 A . C
           - 5x6 notched frame:
                 . . A A A A
                 A A A B B A
                 . . A . . A
                 A A A B B A
                 . . A A A A
         where '.' is the background color and letters are object colors.
      4. Larger detected templates are marked first; horizontal lines are only
         detected in unoccupied pixels. This prevents lines that are merely
         rows of a larger object from being extracted separately.
      5. All extracted objects are sorted by:
           - increasing distance of their vertical center from the grid center
           - for ties, lower objects first
           - then left-to-right
         Duplicate exact objects are kept only once.
      6. Output is a single row of objects, centered vertically, with the
         original background color filling the remaining cells.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2 or grid.size == 0:
        return grid.copy()

    H, W = grid.shape

    # Background is the most common color.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    nonbg = grid != bg
    if not np.any(nonbg):
        return np.full((1, 1), bg, dtype=grid.dtype)

    # ------------------------------------------------------------------
    # 4-connected components of non-background pixels.
    # ------------------------------------------------------------------
    labels = -np.ones((H, W), dtype=int)
    comps = []

    for i in range(H):
        for j in range(W):
            if nonbg[i, j] and labels[i, j] == -1:
                cid = len(comps)
                labels[i, j] = cid
                stack = [(i, j)]

                minr = maxr = i
                minc = maxc = j

                while stack:
                    r, c = stack.pop()

                    minr = min(minr, r)
                    maxr = max(maxr, r)
                    minc = min(minc, c)
                    maxc = max(maxc, c)

                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if nonbg[nr, nc] and labels[nr, nc] == -1:
                                labels[nr, nc] = cid
                                stack.append((nr, nc))

                comps.append((cid, minr, maxr, minc, maxc))

    # Deterministic component processing order.
    target_center2 = H - 1  # twice the center row coordinate
    comps.sort(key=lambda comp: (
        abs(comp[1] + comp[2] - target_center2),
        -(comp[1] + comp[2]),
        comp[3]
    ))

    # ------------------------------------------------------------------
    # Template definitions.
    # '0' means background; letters are non-background placeholders.
    # Different letters are required to have different colors.
    # ------------------------------------------------------------------
    raw_templates = [
        # 3x3 ring
        (
            "AAA",
            "ABA",
            "AAA",
        ),
        # 3x3 checker
        (
            "ABA",
            "BAB",
            "ABA",
        ),
        # 3x3 H-shape
        (
            "A0A",
            "AAA",
            "A0A",
        ),
        # 5x3 block with right vertical stripe
        (
            "AAA",
            "AAB",
            "AAB",
            "AAB",
            "AAA",
        ),
        # 5x3 left-bar framed object
        (
            "A0C",
            "ADA",
            "ADE",
            "ADA",
            "A0C",
        ),
        # 5x6 notched frame
        (
            "00AAAA",
            "AAABBA",
            "00A00A",
            "AAABBA",
            "00AAAA",
        ),
    ]

    templates = []
    for pat in raw_templates:
        th, tw = len(pat), len(pat[0])
        cells = [(i, j, ch) for i, row in enumerate(pat) for j, ch in enumerate(row)]
        templates.append((th, tw, cells))

    # ------------------------------------------------------------------
    # Detect objects.
    # ------------------------------------------------------------------
    all_objects = []

    for cid, minr, maxr, minc, maxc in comps:
        candidates = []

        # Detect 2D templates first.
        for th, tw, cells in templates:
            if th > maxr - minr + 1 or tw > maxc - minc + 1:
                continue

            for r in range(minr, maxr - th + 2):
                for c in range(minc, maxc - tw + 2):
                    assigned = {}
                    used_values = set()
                    ok = True

                    for i, j, ch in cells:
                        rr, cc = r + i, c + j
                        val = int(grid[rr, cc])

                        if ch == '0':
                            if val != bg:
                                ok = False
                                break
                        else:
                            if val == bg or labels[rr, cc] != cid:
                                ok = False
                                break

                            if ch in assigned:
                                if assigned[ch] != val:
                                    ok = False
                                    break
                            else:
                                # Different template slots must use different colors.
                                if val in used_values:
                                    ok = False
                                    break
                                assigned[ch] = val
                                used_values.add(val)

                    if ok:
                        tile = grid[r:r + th, c:c + tw].copy()
                        area = int(np.count_nonzero(tile != bg))
                        candidates.append((area, r, c, tile))

        # Prefer larger objects when they overlap.
        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

        occupied = np.zeros((H, W), dtype=bool)
        selected = []

        for _, r, c, tile in candidates:
            th, tw = tile.shape
            if np.any(occupied[r:r + th, c:c + tw] & (tile != bg)):
                continue

            selected.append((c, r, th, tw, tile))
            occupied[r:r + th, c:c + tw] |= (tile != bg)

        # Detect horizontal lines in still-unoccupied non-background pixels.
        for r in range(minr, maxr + 1):
            c = minc
            while c <= maxc:
                if nonbg[r, c] and labels[r, c] == cid and not occupied[r, c]:
                    color = int(grid[r, c])
                    c2 = c

                    while (
                        c2 + 1 <= maxc
                        and nonbg[r, c2 + 1]
                        and labels[r, c2 + 1] == cid
                        and not occupied[r, c2 + 1]
                        and int(grid[r, c2 + 1]) == color
                    ):
                        c2 += 1

                    if c2 - c + 1 >= 3:
                        tile = grid[r, c:c2 + 1].copy()
                        selected.append((c, r, 1, tile.shape[1], tile))

                    c = c2 + 1
                else:
                    c += 1

        # Store with metadata for global ordering.
        for c, r, th, tw, tile in selected:
            center2 = 2 * r + th - 1
            dist = abs(center2 - target_center2)
            all_objects.append((dist, center2, c, r, th, tw, tile))

    if not all_objects:
        return np.full((1, 1), bg, dtype=grid.dtype)

    # ------------------------------------------------------------------
    # Global order: closest to the horizontal center first,
    # lower before higher on ties, then left-to-right.
    # ------------------------------------------------------------------
    all_objects.sort(key=lambda x: (x[0], -x[1], x[2], x[3], x[4], x[5]))

    seen = set()
    result = []

    for _, _, _, _, th, tw, tile in all_objects:
        key = (th, tw, tuple(tile.ravel().tolist()))
        if key in seen:
            continue
        seen.add(key)
        result.append(tile)

    # ------------------------------------------------------------------
    # Build output strip.
    # ------------------------------------------------------------------
    max_h = max(t.shape[0] for t in result)
    total_w = sum(t.shape[1] for t in result)

    out = np.full((max_h, total_w), bg, dtype=grid.dtype)

    x = 0
    for t in result:
        y = (max_h - t.shape[0]) // 2
        out[y:y + t.shape[0], x:x + t.shape[1]] = t
        x += t.shape[1]

    return out