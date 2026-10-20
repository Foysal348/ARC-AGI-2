import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    For each color that forms rectangular line frames, count the number of
    enclosed rectangular "rooms" created by that color's lines.

    The output has:
      - one row per such color,
      - columns equal to the maximum room count,
      - row i filled with its color for its count, then filled with the
        distractor/noise color.

    Rows are sorted by room count ascending. Ties are broken by reading
    order of the color's top-left pixel, then color value.
    """
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid, dtype=int)
    else:
        grid = grid.astype(int, copy=False)

    if grid.size == 0:
        return np.array([], dtype=int)

    H, W = grid.shape
    nonzero_colors = [int(c) for c in np.unique(grid) if c != 0]

    if not nonzero_colors:
        return np.zeros((1, 1), dtype=int)

    def max_run_in_1d(indices):
        """Longest consecutive run among sorted 1D indices."""
        if len(indices) == 0:
            return 0
        best = 1
        cur = 1
        for i in range(1, len(indices)):
            if indices[i] == indices[i - 1] + 1:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 1
        return best

    def add_segments(mask, wall):
        """
        Mark full horizontal/vertical line segments whose pixels are mostly
        present in mask. This recovers lines with small missing/noisy pixels.
        """
        min_span = 3
        min_count = 3
        min_density = 0.6
        min_run = 2
        max_gap = 5

        # Horizontal segments.
        for y in range(H):
            xs = np.where(mask[y])[0]
            if len(xs) < min_count:
                continue

            groups = []
            start = prev = int(xs[0])
            for x in xs[1:]:
                x = int(x)
                if x - prev <= max_gap:
                    prev = x
                else:
                    groups.append((start, prev))
                    start = prev = x
            groups.append((start, prev))

            for l, r in groups:
                span = r - l + 1
                if span < min_span:
                    continue

                seg = mask[y, l:r + 1]
                cnt = int(seg.sum())
                if cnt < min_count or cnt < min_density * span:
                    continue

                run = 0
                best = 0
                for v in seg:
                    if v:
                        run += 1
                        if run > best:
                            best = run
                    else:
                        run = 0

                if best >= min_run:
                    wall[y, l:r + 1] = True

        # Vertical segments.
        for x in range(W):
            ys = np.where(mask[:, x])[0]
            if len(ys) < min_count:
                continue

            groups = []
            start = prev = int(ys[0])
            for y in ys[1:]:
                y = int(y)
                if y - prev <= max_gap:
                    prev = y
                else:
                    groups.append((start, prev))
                    start = prev = y
            groups.append((start, prev))

            for l, r in groups:
                span = r - l + 1
                if span < min_span:
                    continue

                seg = mask[l:r + 1, x]
                cnt = int(seg.sum())
                if cnt < min_count or cnt < min_density * span:
                    continue

                run = 0
                best = 0
                for v in seg:
                    if v:
                        run += 1
                        if run > best:
                            best = run
                    else:
                        run = 0

                if best >= min_run:
                    wall[l:r + 1, x] = True

    def fill_hidden_gaps(mask, wall):
        """
        Fill gaps between two pixels of the same color when the gap is mostly
        occupied by a single non-zero color. This recovers walls that are
        mostly overwritten by another object color or by distractor pixels.
        """
        max_gap = 10

        # Horizontal gaps.
        for y in range(H):
            xs = np.where(mask[y])[0]
            for i in range(len(xs) - 1):
                x1 = int(xs[i])
                x2 = int(xs[i + 1])
                gap = x2 - x1 - 1
                if gap < 2 or gap > max_gap:
                    continue

                seg = grid[y, x1 + 1:x2]
                nz_idx = np.where(seg != 0)[0]
                nz_count = len(nz_idx)
                if nz_count < 2 or nz_count * 2 < gap:
                    continue

                if nz_count:
                    vals = seg[nz_idx]
                    vals, counts = np.unique(vals, return_counts=True)
                    if counts.max() < 0.8 * nz_count:
                        continue

                wall[y, x1 + 1:x2] = True

        # Vertical gaps.
        for x in range(W):
            ys = np.where(mask[:, x])[0]
            for i in range(len(ys) - 1):
                y1 = int(ys[i])
                y2 = int(ys[i + 1])
                gap = y2 - y1 - 1
                if gap < 2 or gap > max_gap:
                    continue

                seg = grid[y1 + 1:y2, x]
                nz_idx = np.where(seg != 0)[0]
                nz_count = len(nz_idx)
                if nz_count < 2 or nz_count * 2 < gap:
                    continue

                if nz_count:
                    vals = seg[nz_idx]
                    vals, counts = np.unique(vals, return_counts=True)
                    if counts.max() < 0.8 * nz_count:
                        continue

                wall[y1 + 1:y2, x] = True

    def count_rooms(mask):
        """Count finite empty components enclosed by this color's wall mask."""
        wall = mask.copy()
        fill_hidden_gaps(mask, wall)
        add_segments(mask, wall)

        empty = ~wall
        if not empty.any():
            return 0

        labels = np.zeros((H, W), dtype=int)
        next_label = 0

        def bfs(start_y, start_x):
            nonlocal next_label
            next_label += 1
            label = next_label
            q = deque([(start_y, start_x)])
            labels[start_y, start_x] = label

            while q:
                y, x = q.popleft()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        if empty[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = label
                            q.append((ny, nx))
            return label

        outside_labels = set()

        # Label all components touching the border as outside.
        for y in range(H):
            for x in (0, W - 1):
                if empty[y, x] and labels[y, x] == 0:
                    outside_labels.add(bfs(y, x))
        for x in range(W):
            for y in (0, H - 1):
                if empty[y, x] and labels[y, x] == 0:
                    outside_labels.add(bfs(y, x))

        if next_label == 0:
            return 0

        all_labels = set(int(v) for v in np.unique(labels))
        all_labels.discard(0)
        return len(all_labels - outside_labels)

    stats = {}
    for c in nonzero_colors:
        mask = (grid == c)
        ys, xs = np.where(mask)

        max_h = 0
        for y in range(H):
            max_h = max(max_h, max_run_in_1d(np.where(mask[y])[0]))

        max_v = 0
        for x in range(W):
            max_v = max(max_v, max_run_in_1d(np.where(mask[:, x])[0]))

        stats[c] = {
            "count": int(mask.sum()),
            "max_run": max(max_h, max_v),
            "top": int(ys.min()),
            "left": int(xs.min()),
        }

    objects = []
    for c in nonzero_colors:
        # Real frame colors must contain at least a 3-pixel line and enough pixels.
        if stats[c]["max_run"] < 3 or stats[c]["count"] < 8:
            continue

        cnt = count_rooms((grid == c))
        if cnt > 0:
            objects.append((cnt, c, stats[c]["top"], stats[c]["left"]))

    if not objects:
        return np.zeros((1, 1), dtype=int)

    object_colors = {c for _, c, _, _ in objects}

    # The filler is the non-object color with the least line-like behavior.
    non_object = [c for c in nonzero_colors if c not in object_colors]
    if non_object:
        filler = min(non_object, key=lambda c: (stats[c]["max_run"], -stats[c]["count"]))
    else:
        filler = 0

    # Sort by count, then visual reading order, then color value.
    objects.sort(key=lambda t: (t[0], t[2], t[3], t[1]))

    max_cnt = max(t[0] for t in objects)
    out = np.full((len(objects), max_cnt), filler, dtype=int)

    for i, (cnt, c, _, _) in enumerate(objects):
        out[i, :cnt] = c

    return out