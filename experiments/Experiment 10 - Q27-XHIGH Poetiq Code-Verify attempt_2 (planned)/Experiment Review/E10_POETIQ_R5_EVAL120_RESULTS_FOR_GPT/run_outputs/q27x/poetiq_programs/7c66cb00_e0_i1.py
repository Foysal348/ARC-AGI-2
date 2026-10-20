import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by:
      1. Removing all small source shapes.
      2. Keeping the full-width horizontal bands, rebuilt with their
         interior fill color and side border color.
      3. For each source shape and each band whose fill color appears in
         the shape, stamp the subpattern made of pixels of that fill color
         into the band at the same horizontal position, bottom-aligned,
         recolored to the band's border color.
    """
    grid = np.asarray(grid, dtype=int)

    if grid.ndim != 2 or grid.size == 0:
        return grid

    H, W = grid.shape

    # Infer the background color. In these puzzles it is the color at the
    # corners; fall back to the most frequent color if corners differ.
    corners = [grid[0, 0], grid[0, W - 1], grid[H - 1, 0], grid[H - 1, W - 1]]
    if len(set(int(c) for c in corners)) == 1:
        bg = int(corners[0])
    else:
        vals, counts = np.unique(grid, return_counts=True)
        bg = int(vals[np.argmax(counts)])

    def band_info(row):
        """Return (border_color, fill_color) if row is a full-width band row."""
        if W < 3:
            return None

        border = int(row[0])
        if int(row[-1]) != border:
            return None

        interior = row[1:-1]
        fill = int(interior[0])

        if not np.all(interior == fill):
            return None

        # A band should use two colors distinct from the background.
        if border == fill or border == bg or fill == bg:
            return None

        return border, fill

    # Detect all full-width bands.
    bands = []
    r = 0
    while r < H:
        info = band_info(grid[r])
        if info is None:
            r += 1
            continue

        border, fill = info
        bottom = r
        while bottom + 1 < H:
            next_info = band_info(grid[bottom + 1])
            if next_info is None or next_info[0] != border or next_info[1] != fill:
                break
            bottom += 1

        bands.append((r, bottom, border, fill))
        r = bottom + 1

    # Rebuild output: background everywhere, then draw the empty bands.
    out = np.full((H, W), bg, dtype=int)
    in_band = np.zeros((H, W), dtype=bool)

    for top, bottom, border, fill in bands:
        out[top:bottom + 1, :] = fill
        out[top:bottom + 1, 0] = border
        out[top:bottom + 1, -1] = border
        in_band[top:bottom + 1, :] = True

    # Find source shapes as connected components of non-background cells,
    # excluding the band areas.
    mask = (grid != bg) & (~in_band)

    if np.any(mask):
        visited = np.zeros((H, W), dtype=bool)
        components = []

        for y in range(H):
            for x in range(W):
                if not mask[y, x] or visited[y, x]:
                    continue

                stack = [(y, x)]
                visited[y, x] = True
                pixels = []
                colors = set()

                while stack:
                    cy, cx = stack.pop()
                    pixels.append((cy, cx))
                    colors.add(int(grid[cy, cx]))

                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if (
                            0 <= ny < H
                            and 0 <= nx < W
                            and mask[ny, nx]
                            and not visited[ny, nx]
                        ):
                            visited[ny, nx] = True
                            stack.append((ny, nx))

                components.append((pixels, colors))

        # Stamp relevant subpatterns into every matching band.
        for pixels, colors in components:
            for top, bottom, border, fill in bands:
                if fill not in colors:
                    continue

                sub_pixels = [(y, x) for y, x in pixels if grid[y, x] == fill]
                if not sub_pixels:
                    continue

                sub_top = min(y for y, _ in sub_pixels)
                sub_bottom = max(y for y, _ in sub_pixels)
                height = sub_bottom - sub_top + 1

                # Bottom-align this subpattern's bounding box inside the band.
                target_top = bottom - height + 1

                for y, x in sub_pixels:
                    ty = y - sub_top + target_top
                    if top <= ty <= bottom and 0 <= x < W:
                        out[ty, x] = border

    # Reassert band side borders in case a stamp touched them.
    for top, bottom, border, fill in bands:
        out[top:bottom + 1, 0] = border
        out[top:bottom + 1, -1] = border

    return out