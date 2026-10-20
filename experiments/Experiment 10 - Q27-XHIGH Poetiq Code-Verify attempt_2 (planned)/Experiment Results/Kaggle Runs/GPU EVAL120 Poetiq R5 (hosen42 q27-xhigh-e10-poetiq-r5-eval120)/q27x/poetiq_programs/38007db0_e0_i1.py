import numpy as np
from collections import Counter


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a grid of framed tiles into a single column of tiles.

    The input is interpreted as a regular grid of tiles separated by solid
    lines of a single separator color. For each row of tiles, this function
    selects the most anomalous tile:
      1. If exactly one tile has a pattern that appears only once in that row,
         that tile is selected.
      2. Otherwise, the tile with the largest total Hamming distance from the
         other tiles in the row is selected.

    The selected tiles are returned in row order as one tile-wide column,
    with separator rows and separator columns preserved.
    """
    if grid is None:
        return np.zeros((0, 0), dtype=int)

    grid = np.asarray(grid)

    if grid.size == 0:
        return grid.copy()

    if grid.ndim != 2:
        return grid.copy()

    H, W = grid.shape

    # ------------------------------------------------------------------
    # 1. Identify the separator color.
    #    It should be the color that forms many full rows and full columns.
    # ------------------------------------------------------------------
    colors = np.unique(grid)
    sep_color = grid[0, 0]
    best_score = -1

    for c in colors:
        full_rows = int(np.all(grid == c, axis=1).sum())
        full_cols = int(np.all(grid == c, axis=0).sum())

        if full_rows >= 2 and full_cols >= 2:
            score = full_rows + full_cols
            if score > best_score:
                best_score = score
                sep_color = c

    # Fallback: use the most common border color.
    if best_score <= 0:
        border = np.concatenate(
            (grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1])
        )
        vals, cnts = np.unique(border, return_counts=True)
        sep_color = vals[np.argmax(cnts)]

    # ------------------------------------------------------------------
    # 2. Find the non-separator bands.
    #    Each contiguous block of non-separator rows/columns is a tile band.
    # ------------------------------------------------------------------
    sep_row_mask = np.all(grid == sep_color, axis=1)
    sep_col_mask = np.all(grid == sep_color, axis=0)

    def contiguous_bands(mask):
        bands = []
        start = None

        for i, v in enumerate(mask):
            if v:
                if start is None:
                    start = i
            else:
                if start is not None:
                    bands.append((start, i))
                    start = None

        if start is not None:
            bands.append((start, len(mask)))

        return bands

    tile_row_bands = contiguous_bands(~sep_row_mask)
    tile_col_bands = contiguous_bands(~sep_col_mask)

    if not tile_row_bands or not tile_col_bands:
        return grid.copy()

    # ------------------------------------------------------------------
    # 3. Helper functions for comparing tiles.
    # ------------------------------------------------------------------
    def hamming_distance(a, b):
        """Number of differing pixels between two tile arrays."""
        if a.shape == b.shape:
            return int(np.count_nonzero(a != b))

        h = max(a.shape[0], b.shape[0])
        w = max(a.shape[1], b.shape[1])

        ap = np.zeros((h, w), dtype=int)
        bp = np.zeros((h, w), dtype=int)

        ap[:a.shape[0], :a.shape[1]] = a
        bp[:b.shape[0], :b.shape[1]] = b

        return int(np.count_nonzero(ap != bp))

    def choose_outlier_tile(tiles):
        """
        Choose the odd tile from a row of tiles.

        Priority:
          - unique exact pattern, if exactly one exists
          - largest total Hamming distance to all other tiles
          - tie-break by distance to the most frequent pattern
          - tie-break by leftmost position
        """
        n = len(tiles)
        if n <= 1:
            return 0

        # Exact pattern identity for uniqueness checks.
        keys = [tile.tobytes() for tile in tiles]
        counts = Counter(keys)

        unique_indices = [i for i, k in enumerate(keys) if counts[k] == 1]
        if len(unique_indices) == 1:
            return unique_indices[0]

        # Total difference from every other tile.
        sums = np.zeros(n, dtype=int)

        for i in range(n):
            for j in range(i + 1, n):
                d = hamming_distance(tiles[i], tiles[j])
                sums[i] += d
                sums[j] += d

        max_sum = int(sums.max())
        if max_sum == 0:
            return 0

        candidates = np.where(sums == max_sum)[0]

        if len(candidates) == 1:
            return int(candidates[0])

        # Tie-break: prefer the tile farthest from the most frequent pattern(s).
        modal_count = max(counts.values())
        modal_keys = {k for k, cnt in counts.items() if cnt == modal_count}
        modal_indices = [i for i, k in enumerate(keys) if k in modal_keys]

        def modal_distance(idx):
            if not modal_indices:
                return 0
            return max(hamming_distance(tiles[idx], tiles[m]) for m in modal_indices)

        return int(min(candidates, key=lambda i: (-modal_distance(i), i)))

    # ------------------------------------------------------------------
    # 4. Select one tile from each tile row.
    # ------------------------------------------------------------------
    selected_indices = []

    for row_start, row_end in tile_row_bands:
        tiles = []
        for col_start, col_end in tile_col_bands:
            tiles.append(grid[row_start:row_end, col_start:col_end])

        selected_indices.append(choose_outlier_tile(tiles))

    # ------------------------------------------------------------------
    # 5. Build the output: one tile column, with separator borders.
    # ------------------------------------------------------------------
    tile_widths = [end - start for start, end in tile_col_bands]
    out_width = max(tile_widths) + 2

    out = np.full((H, out_width), sep_color, dtype=grid.dtype)

    row_band_index = np.full(H, -1, dtype=int)
    for idx, (start, end) in enumerate(tile_row_bands):
        row_band_index[start:end] = idx

    for r in range(H):
        if sep_row_mask[r]:
            continue

        bi = row_band_index[r]
        if bi < 0:
            continue

        col_start, col_end = tile_col_bands[selected_indices[bi]]
        tile_row = grid[r, col_start:col_end]
        w = tile_row.shape[1]

        out[r, 1:1 + w] = tile_row

    return out