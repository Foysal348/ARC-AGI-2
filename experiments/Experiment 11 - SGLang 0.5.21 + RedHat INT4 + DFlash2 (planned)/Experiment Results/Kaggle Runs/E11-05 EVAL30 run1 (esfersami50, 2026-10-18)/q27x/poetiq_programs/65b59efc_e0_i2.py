import numpy as np
from collections import Counter


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform an ARC grid of the form:

      - top band: S x S pattern tiles, each having one pattern color
      - middle band: S x S mask tiles, each having one color referring to a top tile
      - bottom band: palette colors, one for each top tile column

    The output is an S*S by S*S grid made of S x S blocks.

    For every middle mask tile:
      1. Its color identifies a top pattern tile.
      2. For every nonzero cell in the mask, place a copy of that top pattern
         at the corresponding block position.
      3. Recolor the pattern using the palette color associated with that
         top tile column.

    The function detects S, tile starts, top/middle masks, and palette colors
    from the input layout.
    """

    if grid is None or grid.size == 0:
        return grid

    H, W = grid.shape

    def row_is_fence_like(row: np.ndarray) -> bool:
        """A separator/fence row contains only 0 and 5, and at least one 5."""
        if row.size == 0:
            return False
        if not np.any(row == 5):
            return False
        return bool(np.all((row == 0) | (row == 5)))

    def count5(row: np.ndarray) -> int:
        return int(np.sum(row == 5))

    # ------------------------------------------------------------------
    # 1. Find the horizontal separator rows.
    #    We prefer rows with a sufficiently dense 5-pattern.
    # ------------------------------------------------------------------
    fence_rows = [r for r in range(H) if row_is_fence_like(grid[r])]

    if not fence_rows:
        return np.zeros_like(grid)

    # Heuristic density threshold.  Small grids still get threshold 2.
    threshold = max(2, W // 6 + 2)

    strong_rows = [
        r for r in fence_rows
        if count5(grid[r]) >= threshold
    ]

    first_sep = None
    second_sep = None

    # Prefer a strong row r such that row r+1+r is also a fence row.
    # This uses the invariant:
    #   top height = S = first_sep
    #   middle height = S
    #   second separator = first_sep + 1 + S
    for r in strong_rows:
        if r == 0:
            continue
        candidate_second = r + 1 + r
        if candidate_second < H and row_is_fence_like(grid[candidate_second]):
            first_sep = r
            second_sep = candidate_second
            break

    # Fallback: any strong row followed by any later fence row.
    if first_sep is None:
        for r in strong_rows:
            if r == 0:
                continue
            later = [q for q in fence_rows if q > r]
            if later:
                first_sep = r
                second_sep = later[0]
                break

    # Fallback: weakest sane choice.
    if first_sep is None:
        for r in fence_rows:
            if r == 0:
                continue
            candidate_second = r + 1 + r
            if candidate_second < H and row_is_fence_like(grid[candidate_second]):
                first_sep = r
                second_sep = candidate_second
                break

    if first_sep is None:
        return np.zeros_like(grid)

    S = first_sep
    if S <= 0:
        return np.zeros_like(grid)

    middle_start = first_sep + 1
    if second_sep is None or second_sep >= H:
        second_sep = H
    middle_end = second_sep
    bottom_start = second_sep + 1

    # ------------------------------------------------------------------
    # 2. Find horizontal tile starts.
    #    Tiles are width S, separated by 1-column fences.
    #    We infer a common phase modulo S+1 from vertical fence columns
    #    in the top band.
    # ------------------------------------------------------------------
    starts = [0]

    # A vertical fence column in the top band has only 0/5 and at least one 5.
    vertical_fences = []
    for c in range(W):
        col = grid[:first_sep, c]
        if col.size == 0:
            continue
        if np.any(col == 5) and np.all((col == 0) | (col == 5)):
            vertical_fences.append(c)

    if vertical_fences:
        step = S + 1
        residues = Counter()
        for f in vertical_fences:
            residues[(f - S) % step] += 1
        start0 = residues.most_common(1)[0][0]

        # Safety: if there is pattern color before this phase, reset to 0.
        if start0 > 0:
            pre = grid[:first_sep, :start0]
            if pre.size and np.any((pre != 0) & (pre != 5)):
                start0 = 0
    else:
        # Fallback: use the first 5 in the separator row as left phase.
        sep_row = grid[first_sep]
        idx5 = np.where(sep_row == 5)[0]
        start0 = int(idx5[0]) if idx5.size else 0

    if start0 < 0 or start0 + S > W:
        start0 = 0

    step = S + 1
    starts = []
    c = start0
    while c + S <= W:
        starts.append(c)
        c += step

    if not starts:
        starts = [0]

    # ------------------------------------------------------------------
    # 3. Extract top pattern tiles.
    # ------------------------------------------------------------------
    def tile_mask(tile: np.ndarray) -> np.ndarray:
        return (tile != 0) & (tile != 5)

    def dominant_color(tile: np.ndarray) -> int:
        mask = tile_mask(tile)
        if not np.any(mask):
            return 0
        vals = tile[mask]
        if vals.size == 0:
            return 0
        return Counter(vals.tolist()).most_common(1)[0][0]

    top_info = []  # (color, mask)
    for s in starts:
        tile = grid[:first_sep, s:s + S]

        # In valid inputs this is already S x S, but guard against edge crops.
        if tile.shape[0] < S or tile.shape[1] < S:
            fixed = np.zeros((S, S), dtype=tile.dtype)
            h = min(S, tile.shape[0])
            w = min(S, tile.shape[1])
            fixed[:h, :w] = tile[:h, :w]
            tile = fixed
        else:
            tile = tile[:S, :S]

        top_info.append((dominant_color(tile), tile_mask(tile)))

    # ------------------------------------------------------------------
    # 4. Extract middle mask tiles.
    # ------------------------------------------------------------------
    mid_info = []  # (color, mask)
    for s in starts:
        tile = grid[middle_start:middle_end, s:s + S]

        if tile.shape[0] < S or tile.shape[1] < S:
            fixed = np.zeros((S, S), dtype=tile.dtype)
            h = min(S, tile.shape[0])
            w = min(S, tile.shape[1])
            fixed[:h, :w] = tile[:h, :w]
            tile = fixed
        else:
            tile = tile[:S, :S]

        mid_info.append((dominant_color(tile), tile_mask(tile)))

    # ------------------------------------------------------------------
    # 5. Extract palette colors from the bottom band.
    #    The palette color for top tile i is the dominant non-0/non-5 color
    #    in the bottom band above the same tile column.
    # ------------------------------------------------------------------
    palette = []
    for s in starts:
        bottom = grid[bottom_start:, s:s + S]
        if bottom.size == 0:
            palette.append(0)
            continue
        palette.append(dominant_color(bottom))

    # ------------------------------------------------------------------
    # 6. Build output.
    # ------------------------------------------------------------------
    out = np.zeros((S * S, S * S), dtype=grid.dtype)

    # Map pattern color -> top tile index.
    # In valid examples colors are unique; if not, first occurrence wins.
    color_to_top = {}
    for i, (color, mask) in enumerate(top_info):
        if color != 0 and np.any(mask) and color not in color_to_top:
            color_to_top[color] = i

    # Stamp by middle mask order.
    for i, (mid_color, mid_mask) in enumerate(mid_info):
        if mid_color == 0:
            continue
        if not np.any(mid_mask):
            continue

        top_idx = color_to_top.get(mid_color)
        if top_idx is None:
            continue

        new_color = palette[top_idx]
        if new_color == 0:
            continue

        top_mask = top_info[top_idx][1]
        if not np.any(top_mask):
            continue

        # For every active block, paint only the top pattern pixels.
        for r, c in np.argwhere(mid_mask):
            if r < 0 or c < 0 or r >= S or c >= S:
                continue

            block = out[r * S:(r + 1) * S, c * S:(c + 1) * S]
            block[top_mask] = new_color

    return out