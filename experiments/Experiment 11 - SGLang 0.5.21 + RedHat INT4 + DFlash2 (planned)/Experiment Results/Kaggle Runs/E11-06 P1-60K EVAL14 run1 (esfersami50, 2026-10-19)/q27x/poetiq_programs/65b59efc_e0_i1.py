import numpy as np


def transform(grid: np.ndarray) -> np.ndarray:
    def dominant_color(arr):
        vals = arr[arr != 0]
        vals = vals[vals != 5]
        if vals.size == 0:
            return 0
        vals, counts = np.unique(vals, return_counts=True)
        return int(vals[np.argmax(counts)])

    H, W = grid.shape

    def first_separator_row():
        for r in range(H):
            if grid[r, 0] == 5:
                return r
        counts = (grid == 5).sum(axis=1)
        best = 0
        for r, c in enumerate(counts):
            if c > best + 2 and c > 2:
                return r
            best = max(best, c)
        return 0

    def second_separator_row(start):
        for r in range(start + 1, H):
            if grid[r, 0] == 5:
                return r
        counts = (grid == 5).sum(axis=1)
        best = 0
        for r in range(start + 1, H):
            c = counts[r]
            if c > best + 2 and c > 2:
                return r
            best = max(best, c)
        return min(start + 1, H - 1)

    sep1 = first_separator_row()
    if sep1 == 0:
        counts = (grid == 5).sum(axis=1)
        for r, c in enumerate(counts):
            if c > 2:
                sep1 = r
                break

    s = max(1, sep1)
    sep2 = second_separator_row(sep1)

    starts = []
    c = 0
    while c + s <= W:
        block = grid[:s, c:c + s]
        if np.any((block != 0) & (block != 5)):
            starts.append(c)
        c += s + 1

    if not starts:
        return np.zeros_like(grid)

    top_patterns = []
    top_colors = []
    for c in starts:
        block = grid[:s, c:c + s]
        col = dominant_color(block)
        if col == 0:
            mask = np.zeros((s, s), dtype=bool)
        else:
            mask = (block == col)
        top_patterns.append(mask)
        top_colors.append(col)

    color_to_top = {}
    for i, col in enumerate(top_colors):
        if col != 0 and col not in color_to_top:
            color_to_top[col] = i

    key_row = None
    for r in range(sep2 + 1, H):
        if np.any((grid[r] != 0) & (grid[r] != 5)):
            key_row = r
            break

    key_colors = []
    for c in starts:
        if key_row is None:
            key_colors.append(0)
            continue
        seg = grid[key_row, c:c + s]
        key_colors.append(dominant_color(seg))

    labels = np.full((s, s), -1, dtype=int)
    middle = grid[sep1 + 1:sep2, :]

    for c in starts:
        if middle.size == 0:
            continue
        block = middle[:s, c:c + s]
        col = dominant_color(block)
        if col == 0 or col not in color_to_top:
            continue
        label = color_to_top[col]
        mask = (block == col)
        rows, cols = np.argwhere(mask).T
        for r, cc in zip(rows, cols):
            if r < s and cc < s and labels[r, cc] == -1:
                labels[r, cc] = label

    out = np.zeros((s * s, s * s), dtype=grid.dtype)

    for r in range(s):
        for c in range(s):
            label = labels[r, c]
            if label < 0 or label >= len(top_patterns):
                continue
            color = key_colors[label]
            if color == 0:
                continue
            block = np.zeros((s, s), dtype=grid.dtype)
            block[top_patterns[label]] = color
            out[r * s:(r + 1) * s, c * s:(c + 1) * s] = block

    return out