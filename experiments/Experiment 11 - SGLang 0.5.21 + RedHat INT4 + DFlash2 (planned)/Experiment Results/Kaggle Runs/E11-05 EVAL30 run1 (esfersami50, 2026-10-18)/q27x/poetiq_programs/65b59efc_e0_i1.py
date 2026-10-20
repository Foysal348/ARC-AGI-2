import numpy as np
from collections import Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform an ARC grid made of three logical bands:
      - top band: S x S pattern tiles, each with a distinguishing color
      - middle band: S x S binary mask tiles, each with a color that refers to a top tile
      - bottom band: a palette row containing one new color per top tile

    For each top tile:
      1. find the middle mask tile whose color matches the top tile's color
      2. recolor the top tile with the palette color belonging to that top tile's column
      3. stamp the recolored S x S tile into every cell of the middle mask
    The final output is an (S*S) x (S*S) grid.
    """

    if grid.size == 0:
        return grid

    H, W = grid.shape

    def _is_fence_row(r: int) -> bool:
        return np.count_nonzero(grid[r] == 5) > 0

    def _is_fence_col(c: int) -> bool:
        return np.count_nonzero(grid[:, c] == 5) > 0

    # ---------------------------------------------------------------------
    # Determine the size S of each square cell.
    # In all observed grids, top row has S cells in a row, separated by 1-wide
    # color-5 fences. We infer S from the first horizontal fence row, which
    # contains a repeating S-pattern: e.g. 5 0 5 0 5 for S=5.
    # ---------------------------------------------------------------------
    sep_rows = [r for r in range(H) if _is_fence_row(r)]
    if not sep_rows:
        return np.zeros_like(grid)

    # Use the first full separator row to infer S.
    first_sep = sep_rows[0]
    first_sep_arr = (grid[first_sep] == 5)

    # Find runs of non-separator columns after the first separator run.
    # For an S-wide cell, the separator pattern inside a cell usually has
    # S pixels with a repeating 5/0 pattern; the simplest robust method is to
    # count how many 5s are in the first separator row and derive S from known
    # structure. But to remain general, infer S from top-band tile width.
    #
    # We scan the top band (above the first separator row) for contiguous
    # non-zero regions; each such region is one tile.
    top_end = first_sep
    top_band = grid[:top_end]

    # Find columns that are fences in the top band.
    # A fence column may not be fully 5 in the top band, but the pattern of
    # 5s in the separator row tells us the cell width more reliably.
    #
    # Try common S values from 1..max(10) and pick the one that best explains
    # the separator row pattern and top band width.
    best_S = 1
    best_score = -1

    for S in range(1, 11):
        # Candidate column starts: 0, S+1, 2S+2, ...
        starts = list(range(0, W, S + 1))
        # Adjust if the grid is cropped/padded on the right.
        # A good S should make most top-band columns align with starts or start+S.
        score = 0
        # Use the separator row: it should have 5s within the first starts region
        # in a repeating pattern.
        for c in range(min(W, (len(starts) * (S + 1)))):
            in_cell = any(0 <= c - st < S for st in starts)
            has5 = (grid[first_sep, c] == 5)
            if has5 and in_cell:
                score += 1
        # Also prefer S that makes W close to n*(S+1)-1 for some n.
        if W > 0:
            for n in range(1, W):
                if abs(n * (S + 1) - 1 - W) <= 2:
                    score += 3
        if score > best_score:
            best_score = score
            best_S = S

    S = best_S

    # ---------------------------------------------------------------------
    # Determine the number of logical columns C.
    # We use the top band and separator row.
    # ---------------------------------------------------------------------
    C = max(1, (W + S) // (S + 1))
    # If the grid has extra zero padding on the right, trim C to actual tiles
    # that contain non-zero pixels in the top band.
    while C > 1:
        start = (C - 1) * (S + 1)
        end = min(W, start + S)
        if not np.any(top_band[:, :end] != 0):
            C -= 1
        else:
            break

    # ---------------------------------------------------------------------
    # Locate the second separator row (between middle and bottom bands).
    # ---------------------------------------------------------------------
    sep_rows_after_first = [r for r in sep_rows if r > first_sep]
    second_sep = sep_rows_after_first[0] if sep_rows_after_first else H

    # ---------------------------------------------------------------------
    # Extract helper functions
    # ---------------------------------------------------------------------
    def _cell(row_start, row_end, col_start, col_end):
        return grid[row_start:row_end, col_start:col_end]

    def _tile_color(tile: np.ndarray) -> int:
        # Ignore background 0 and fence 5; take most common remaining color.
        mask = (tile != 0) & (tile != 5)
        if not np.any(mask):
            return 0
        vals = tile[mask]
        cnt = Counter(vals.tolist())
        return cnt.most_common(1)[0][0]

    def _tile_mask(tile: np.ndarray) -> np.ndarray:
        # Binary mask of non-background, non-fence pixels.
        return ((tile != 0) & (tile != 5)).astype(int)

    def _palette_color_for_column(col_idx: int) -> int:
        # In the bottom band, each logical column should contain a palette color.
        # Find the most common non-zero, non-5 color in that column region.
        col_start = col_idx * (S + 1)
        col_end = min(W, col_start + S)
        if second_sep >= H:
            return 0
        band = grid[second_sep + 1:, col_start:col_end]
        mask = (band != 0) & (band != 5)
        if not np.any(mask):
            return 0
        vals = band[mask]
        cnt = Counter(vals.tolist())
        return cnt.most_common(1)[0][0]

    # ---------------------------------------------------------------------
    # Extract top tiles and middle masks.
    # ---------------------------------------------------------------------
    top_tiles = []
    top_colors = []
    for c in range(C):
        c0 = c * (S + 1)
        c1 = min(W, c0 + S)
        tile = _cell(0, top_end, c0, c1)
        # If the extracted tile is narrower due to padding, pad to SxS.
        if tile.shape[0] < S:
            tile = np.vstack([tile, np.zeros((S - tile.shape[0], tile.shape[1]))])
        if tile.shape[1] < S:
            tile = np.hstack([tile, np.zeros((tile.shape[0], S - tile.shape[1]))])
        tile = tile[:S, :S]
        top_tiles.append(tile)
        top_colors.append(_tile_color(tile))

    middle_tiles = []
    middle_masks = []
    middle_colors = []
    for c in range(C):
        c0 = c * (S + 1)
        c1 = min(W, c0 + S)
        tile = _cell(top_end + 1, second_sep, c0, c1)
        if tile.shape[0] < S:
            tile = np.vstack([tile, np.zeros((S - tile.shape[0], tile.shape[1]))])
        if tile.shape[1] < S:
            tile = np.hstack([tile, np.zeros((tile.shape[0], S - tile.shape[1]))])
        tile = tile[:S, :S]
        middle_tiles.append(tile)
        middle_masks.append(_tile_mask(tile))
        middle_colors.append(_tile_color(tile))

    # ---------------------------------------------------------------------
    # Build output: (S*S) x (S*S).
    # ---------------------------------------------------------------------
    out = np.zeros((S * S, S * S), dtype=grid.dtype)

    for j in range(C):
        top_tile = top_tiles[j]
        top_color = top_colors[j]

        # Find the middle mask whose color matches this top tile's color.
        match_idx = None
        for k in range(C):
            if middle_colors[k] == top_color:
                match_idx = k
                break

        if match_idx is None:
            continue

        mask = middle_masks[match_idx]
        new_color = _palette_color_for_column(j)
        if new_color == 0:
            continue

        # Build recolored tile: all non-background pixels become new_color.
        recolored = np.where(_tile_mask(top_tile) == 1, new_color, 0)

        # Stamp the SxS tile at every SxS block where mask is 1.
        for i in range(S):
            for k in range(S):
                if mask[i, k]:
                    r0 = i * S
                    c0 = k * S
                    out[r0:r0 + S, c0:c0 + S] = recolored

    return out