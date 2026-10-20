import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Extract the distinct symmetric object tiles from the input and lay them
    out in a single horizontal strip.

    The rule implemented here is:
      1. The background is the most frequent color.
      2. Candidate object tiles are extracted from 3x3 and 5-row windows.
         - A 5-row window is kept if it is vertically symmetric.
         - A 3x3 window is kept if it is vertically symmetric and is not
           already contained inside a larger 5-row tile.
      3. A 3x3 tile that has an isolated off-center color is normalized to a
         solid block of its dominant non-background color.
      4. Duplicate tiles are removed.
      5. The remaining tiles are sorted deterministically and centered
         vertically on a common 5-row high canvas.
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape

    # Background color is the most frequent color.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    def is_bg_color(x):
        return int(x) == bg

    # ------------------------------------------------------------------
    # 5-row tile extraction
    # ------------------------------------------------------------------
    five_tiles = []

    if H >= 5 and W >= 3:
        # Sliding 5-row windows.
        for r0 in range(H - 4):
            # Try every possible width.
            for c0 in range(W - 2):
                for width in range(3, W - c0 + 1):
                    tile = grid[r0:r0 + 5, c0:c0 + width]

                    # Vertically symmetric:
                    # row 0 == row 4, row 1 == row 3.
                    if not np.array_equal(tile[0], tile[4]):
                        continue
                    if not np.array_equal(tile[1], tile[3]):
                        continue

                    # Must contain at least one non-background pixel.
                    if np.all(tile == bg):
                        continue

                    five_tiles.append(tile.copy())

    # Remove duplicate 5-row tiles and obvious contained duplicates.
    unique_five = []
    for t in five_tiles:
        if all(not np.array_equal(t, u) for u in unique_five):
            unique_five.append(t)

    # Prefer larger tiles when one tile is a horizontal subtile of another
    # at the same vertical alignment.
    def is_subtile(sub, full):
        if sub.shape[0] != full.shape[0]:
            return False
        if sub.shape[1] > full.shape[1]:
            return False
        sw = sub.shape[1]
        fw = full.shape[1]
        for off in range(0, fw - sw + 1):
            if np.array_equal(sub, full[:, off:off + sw]):
                return True
        return False

    filtered_five = []
    for t in unique_five:
        contained = False
        for u in unique_five:
            if np.array_equal(t, u):
                continue
            if t.shape != u.shape and is_subtile(t, u):
                contained = True
                break
        if not contained:
            filtered_five.append(t)

    # ------------------------------------------------------------------
    # 3x3 tile extraction
    # ------------------------------------------------------------------

    def normalize_3x3(tile: np.ndarray) -> np.ndarray:
        """
        Normalize a 3x3 tile:
        - If the center is the only pixel of its color, keep the tile.
        - If there is a unique non-background color off-center, fill the
          whole 3x3 with the most frequent non-background color.
        """
        flat = tile.reshape(-1)
        nonbg_mask = flat != bg
        if not np.any(nonbg_mask):
            return tile

        nonbg_vals = flat[nonbg_mask]
        vals3, counts3 = np.unique(nonbg_vals, return_counts=True)

        center_val = int(tile[1, 1])

        # If the center is unique, this is a normal framed object.
        if center_val != bg and counts3[np.where(vals3 == center_val)[0][0]] == 1:
            return tile

        # Find an off-center unique color.
        off_center_unique = False
        for i in range(9):
            if i == 4:
                continue
            v = int(tile[i // 3, i % 3])
            if v == bg:
                continue
            cnt = int(np.count_nonzero(tile == v))
            if cnt == 1:
                off_center_unique = True
                break

        if off_center_unique:
            # Use the most frequent non-background color to fill the core.
            dom = int(vals3[np.argmax(counts3)])
            out = np.full((3, 3), dom, dtype=grid.dtype)
            # Keep background cells as background.
            out[tile == bg] = bg
            return out

        return tile

    three_tiles = []
    if H >= 3 and W >= 3:
        seen_positions = set()
        for r0 in range(H - 2):
            for c0 in range(W - 2):
                tile = grid[r0:r0 + 3, c0:c0 + 3]

                # Vertically symmetric 3x3.
                if not np.array_equal(tile[0], tile[2]):
                    continue
                if np.all(tile == bg):
                    continue

                # Ignore tiles that are only background/line fragments.
                # They must have at least 2 distinct non-background colors
                # or at least 5 non-background pixels.
                nb = tile[tile != bg]
                if len(np.unique(nb)) < 2 and len(nb) < 5:
                    continue

                key = (r0, c0)
                if key in seen_positions:
                    continue
                seen_positions.add(key)

                three_tiles.append(tile.copy())

    # Remove duplicate 3x3 tiles.
    unique_three = []
    for t in three_tiles:
        if all(not np.array_equal(t, u) for u in unique_three):
            unique_three.append(t)

    # Normalize 3x3 tiles.
    normalized_three = [normalize_3x3(t) for t in unique_three]

    # Remove 3x3 tiles that are already contained in a selected 5-row tile.
    final_three = []
    for t in normalized_three:
        contained = False
        for f in filtered_five:
            # Check whether this 3x3 tile appears in the middle three rows
            # of any 5-row tile.
            mid = f[1:4, :]
            if t.shape[1] <= mid.shape[1]:
                for off in range(0, mid.shape[1] - t.shape[1] + 1):
                    if np.array_equal(t, mid[:, off:off + t.shape[1]]):
                        contained = True
                        break
            if contained:
                break
        if not contained:
            final_three.append(t)

    # Deduplicate again after normalization.
    final_three_dedup = []
    for t in final_three:
        if all(not np.array_equal(t, u) for u in final_three_dedup):
            final_three_dedup.append(t)

    # ------------------------------------------------------------------
    # Sort tiles deterministically.
    # The sort key approximates the visual order seen in the examples:
    #   - by top-left non-background color,
    #   - then by width,
    #   - then by flattened array.
    # ------------------------------------------------------------------

    def sort_key(tile: np.ndarray):
        # First non-background value in row-major order.
        flat = tile.reshape(-1)
        nb_idx = np.flatnonzero(flat != bg)
        if len(nb_idx) == 0:
            first_nb = 999
        else:
            first_nb = int(flat[nb_idx[0]])

        width = tile.shape[1]
        return (first_nb, width, tuple(flat.tolist()))

    tiles = filtered_five + final_three_dedup
    tiles.sort(key=sort_key)

    if not tiles:
        # Fallback: return a minimal background grid.
        return np.zeros_like(grid)

    # ------------------------------------------------------------------
    # Build output: center each tile vertically on a 5-row high canvas.
    # ------------------------------------------------------------------
    max_h = 5
    total_w = sum(t.shape[1] for t in tiles)

    out = np.full((max_h, total_w), bg, dtype=grid.dtype)
    x = 0
    for t in tiles:
        th, tw = t.shape
        y = (max_h - th) // 2
        out[y:y + th, x:x + tw] = t
        x += tw

    return out