import numpy as np
from collections import Counter, defaultdict


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Split the input into framed tiles separated by full rows/columns of a
    separator color. For each tile row, choose the odd tile:
      - if exactly one tile shape occurs once in that row, use it;
      - otherwise use the shape least repeated globally (fewest tile rows,
        then fewest total tiles), with ties broken by more non-background
        pixels and then leftmost.
    Return one column of the selected tiles, preserving separator rows.
    """
    try:
        arr = np.asarray(grid, dtype=int)
    except Exception:
        arr = np.asarray(grid)
        try:
            arr = arr.astype(int, copy=False)
        except Exception:
            pass

    if arr.ndim != 2 or arr.size == 0:
        return arr.copy()

    H, W = arr.shape

    def contiguous_bands(mask):
        bands = []
        start = None
        for i, val in enumerate(mask):
            if val:
                if start is None:
                    start = i
            else:
                if start is not None:
                    bands.append((start, i))
                    start = None
        if start is not None:
            bands.append((start, len(mask)))
        return bands

    # Identify separator color: color forming full rows and full columns.
    colors = np.unique(arr)
    sep_color = arr[0, 0]
    best_score = -1

    for c in colors:
        full_rows = int(np.count_nonzero(np.all(arr == c, axis=1)))
        full_cols = int(np.count_nonzero(np.all(arr == c, axis=0)))
        if full_rows > 0 and full_cols > 0:
            score = full_rows + full_cols
            if score > best_score or (score == best_score and c == arr[0, 0]):
                best_score = score
                sep_color = c

    if best_score <= 0:
        border = np.concatenate((arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]))
        vals, cnts = np.unique(border, return_counts=True)
        sep_color = vals[np.argmax(cnts)]

    row_is_sep = np.all(arr == sep_color, axis=1)
    col_is_sep = np.all(arr == sep_color, axis=0)

    row_bands = contiguous_bands(~row_is_sep)
    col_bands = contiguous_bands(~col_is_sep)

    if not row_bands or not col_bands:
        return arr.copy()

    def pattern_distance(a, b):
        if a.shape == b.shape:
            return int(np.count_nonzero(a != b))
        h = max(a.shape[0], b.shape[0])
        w = max(a.shape[1], b.shape[1])
        pa = np.zeros((h, w), dtype=np.uint8)
        pb = np.zeros((h, w), dtype=np.uint8)
        pa[:a.shape[0], :a.shape[1]] = a
        pb[:b.shape[0], :b.shape[1]] = b
        return int(np.count_nonzero(pa != pb))

    # Extract tiles and normalized shape keys.
    tiles = []
    patterns = []
    keys = []
    fgs = []

    for rs, re in row_bands:
        row_tiles = []
        row_pats = []
        row_keys = []
        row_fgs = []

        for cs, ce in col_bands:
            t = arr[rs:re, cs:ce]

            # Background is assumed to be the dominant color on the tile border.
            boundary = np.concatenate((t[0, :], t[-1, :], t[:, 0], t[:, -1]))
            bvals, bcounts = np.unique(boundary, return_counts=True)
            bg = bvals[int(np.argmax(bcounts))]

            pat = (t != bg).astype(np.uint8)
            all_vals = np.unique(t)
            non_bg = tuple(int(x) for x in all_vals if x != bg)

            key = (t.shape, np.ascontiguousarray(pat).tobytes(), non_bg)

            row_tiles.append(t)
            row_pats.append(pat)
            row_keys.append(key)
            row_fgs.append(int(pat.sum()))

        tiles.append(row_tiles)
        patterns.append(row_pats)
        keys.append(row_keys)
        fgs.append(row_fgs)

    # Global statistics for each tile shape.
    global_count = Counter()
    global_rows = defaultdict(set)

    for bi in range(len(row_bands)):
        for k in keys[bi]:
            global_count[k] += 1
            global_rows[k].add(bi)

    # Choose one tile per tile row.
    selected = []

    for bi in range(len(row_bands)):
        n = len(tiles[bi])
        if n == 0:
            selected.append(0)
            continue

        row_keys = keys[bi]
        row_pats = patterns[bi]
        row_fgs = fgs[bi]

        local_counts = Counter(row_keys)
        unique_indices = [i for i, k in enumerate(row_keys) if local_counts[k] == 1]

        # Clear local odd-one-out case.
        if len(unique_indices) == 1:
            selected.append(unique_indices[0])
            continue

        # No clear local unique tile: use global rarity and tie-breaks.
        cands = list(range(n))

        min_rows = min(len(global_rows[row_keys[i]]) for i in cands)
        cands = [i for i in cands if len(global_rows[row_keys[i]]) == min_rows]

        min_global = min(global_count[row_keys[i]] for i in cands)
        cands = [i for i in cands if global_count[row_keys[i]] == min_global]

        max_fg = max(row_fgs[i] for i in cands)
        cands = [i for i in cands if row_fgs[i] == max_fg]

        if len(set(row_keys[i] for i in cands)) > 1:
            max_local = max(local_counts.values())
            mode_keys = [k for k, c in local_counts.items() if c == max_local]
            ref_key = max(mode_keys, key=lambda k: global_count[k])
            ref_idx = next(i for i in range(n) if row_keys[i] == ref_key)
            ref_pat = row_pats[ref_idx]

            max_dist = max(pattern_distance(row_pats[i], ref_pat) for i in cands)
            cands = [i for i in cands if pattern_distance(row_pats[i], ref_pat) == max_dist]

        selected.append(min(cands))

    # Build output: one selected tile column.
    tile_widths = [ce - cs for cs, ce in col_bands]
    out_width = max(tile_widths) + 2
    out = np.full((H, out_width), sep_color, dtype=arr.dtype)

    row_to_band = np.full(H, -1, dtype=int)
    for bi, (rs, re) in enumerate(row_bands):
        row_to_band[rs:re] = bi

    for r in range(H):
        bi = row_to_band[r]
        if bi < 0:
            continue

        cs, ce = col_bands[selected[bi]]
        w = ce - cs
        out[r, 1:1 + w] = arr[r, cs:ce]

    return out