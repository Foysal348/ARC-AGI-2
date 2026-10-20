import numpy as np


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a scene of colored rectangular frames into a compact summary grid.

    Each non-filler color forms a rectangular frame. Internal line segments of
    the same frame color can divide the frame into multiple empty cells.
    Other colors are treated as noise/occlusion and do not act as walls for
    that frame.

    For every frame:
      1. Estimate its bounding box from its pixels.
      2. Reconstruct wall segments:
           - the outer border is always a wall,
           - straight frame-colored runs are walls,
           - nearly complete full rows/columns are filled in.
      3. Count connected empty components inside the frame, capped at 4.

    Output rows are sorted by:
      - number of cells (capped at 4),
      - then leftmost column of the frame,
      - then topmost row,
      - then color.

    Each output row is [frame_color ... filler_color] with 4 cells.
    """
    grid = np.asarray(grid)
    if grid.ndim != 2:
        return grid.copy()

    h, w = grid.shape
    if h == 0 or w == 0:
        return np.zeros_like(grid)

    MIN_RUN = 3
    MAX_GAP = 3
    DENSITY = 0.5
    FULL_RATIO = 0.6

    def max_run_bool(arr):
        """Maximum length of consecutive True values in a 1D boolean array."""
        best = 0
        cur = 0
        for val in arr:
            if val:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 0
        return best

    def max_horizontal_run(mask):
        best = 0
        for r in range(mask.shape[0]):
            best = max(best, max_run_bool(mask[r]))
        return best

    def max_vertical_run(mask):
        best = 0
        for c in range(mask.shape[1]):
            best = max(best, max_run_bool(mask[:, c]))
        return best

    def long_run_counts(mask):
        row_long = 0
        col_long = 0
        for r in range(mask.shape[0]):
            if max_run_bool(mask[r]) >= MIN_RUN:
                row_long += 1
        for c in range(mask.shape[1]):
            if max_run_bool(mask[:, c]) >= MIN_RUN:
                col_long += 1
        return row_long, col_long

    # ------------------------------------------------------------------
    # 1. Identify frame colors.
    # ------------------------------------------------------------------
    nonzero_colors = [int(c) for c in np.unique(grid) if c != 0]
    frames = []

    for color in nonzero_colors:
        mask = grid == color
        if not mask.any():
            continue

        mh = max_horizontal_run(mask)
        mv = max_vertical_run(mask)
        if mh < MIN_RUN or mv < MIN_RUN:
            continue

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            continue

        r0, r1 = int(rows[0]), int(rows[-1])
        c0, c1 = int(cols[0]), int(cols[-1])

        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue

        row_long, col_long = long_run_counts(mask)
        if row_long + col_long < 3:
            continue

        if int(mask.sum()) < 6:
            continue

        frames.append({
            "color": int(color),
            "r0": r0,
            "r1": r1,
            "c0": c0,
            "c1": c1,
        })

    if not frames:
        return np.zeros_like(grid)

    # ------------------------------------------------------------------
    # 2. Choose filler/noise color: most common non-frame non-zero color.
    # ------------------------------------------------------------------
    frame_set = {f["color"] for f in frames}
    non_frame_colors = [c for c in nonzero_colors if c not in frame_set]

    if non_frame_colors:
        filler = max(non_frame_colors, key=lambda c: int(np.count_nonzero(grid == c)))
    else:
        filler = 0

    # ------------------------------------------------------------------
    # Helpers for one frame.
    # ------------------------------------------------------------------
    def mark_segment_lines(mask, walls):
        """
        Mark straight wall segments made of the frame color.

        Runs separated by small gaps are merged. A merged cluster is accepted
        only if it contains at least one reasonably long run, which avoids
        treating two isolated vertical-line pixels as a fake horizontal line.
        """
        H, W = mask.shape

        # Horizontal segments.
        for r in range(H):
            arr = mask[r]
            if not arr.any():
                continue

            runs = []
            j = 0
            while j < W:
                if arr[j]:
                    s = j
                    while j < W and arr[j]:
                        j += 1
                    runs.append((s, j - 1))
                else:
                    j += 1

            if not runs:
                continue

            clusters = [[runs[0][0], runs[0][1], runs[0][1] - runs[0][0] + 1]]
            for s, e in runs[1:]:
                gap = s - clusters[-1][1] - 1
                if gap <= MAX_GAP:
                    clusters[-1][1] = e
                    clusters[-1][2] = max(clusters[-1][2], e - s + 1)
                else:
                    clusters.append([s, e, e - s + 1])

            for s, e, max_seg_len in clusters:
                length = e - s + 1
                if length >= MIN_RUN and max_seg_len >= 2:
                    cnt = int(arr[s:e + 1].sum())
                    if cnt >= max(2, int(np.ceil(DENSITY * length))):
                        walls[r, s:e + 1] = True

        # Vertical segments.
        for c in range(W):
            arr = mask[:, c]
            if not arr.any():
                continue

            runs = []
            i = 0
            while i < H:
                if arr[i]:
                    s = i
                    while i < H and arr[i]:
                        i += 1
                    runs.append((s, i - 1))
                else:
                    i += 1

            if not runs:
                continue

            clusters = [[runs[0][0], runs[0][1], runs[0][1] - runs[0][0] + 1]]
            for s, e in runs[1:]:
                gap = s - clusters[-1][1] - 1
                if gap <= MAX_GAP:
                    clusters[-1][1] = e
                    clusters[-1][2] = max(clusters[-1][2], e - s + 1)
                else:
                    clusters.append([s, e, e - s + 1])

            for s, e, max_seg_len in clusters:
                length = e - s + 1
                if length >= MIN_RUN and max_seg_len >= 2:
                    cnt = int(arr[s:e + 1].sum())
                    if cnt >= max(2, int(np.ceil(DENSITY * length))):
                        walls[s:e + 1, c] = True

    def full_line_threshold(n):
        if n <= 1:
            return 1
        return max(2, int(np.ceil(FULL_RATIO * n)))

    def fill_full_lines(mask, walls):
        """
        Fill in nearly complete internal horizontal/vertical lines.

        This handles lines that are mostly present but broken by occlusion.
        The threshold is deliberately conservative so that a partial divider
        spanning only half the frame is not turned into a full divider.
        """
        H, W = mask.shape
        inter_h = H - 2
        inter_w = W - 2

        for c in range(1, W - 1):
            if inter_h <= 0:
                continue
            col_mask = mask[1:-1, c]
            col_wall = walls[1:-1, c]

            ev = int(max(col_mask.sum(), col_wall.sum()))
            mr = max(max_run_bool(col_mask), max_run_bool(col_wall))

            if ev >= full_line_threshold(inter_h):
                min_mr = 1 if inter_h == 1 else 2
                if mr >= min_mr:
                    walls[1:-1, c] = True

        for r in range(1, H - 1):
            if inter_w <= 0:
                continue
            row_mask = mask[r, 1:-1]
            row_wall = walls[r, 1:-1]

            ev = int(max(row_mask.sum(), row_wall.sum()))
            mr = max(max_run_bool(row_mask), max_run_bool(row_wall))

            if ev >= full_line_threshold(inter_w):
                min_mr = 1 if inter_w == 1 else 2
                if mr >= min_mr:
                    walls[r, 1:-1] = True

    def count_rooms(mask, H, W):
        """
        Count connected empty cells inside the frame, capped at 4.
        """
        walls = np.zeros((H, W), dtype=bool)

        # The outer border is always a wall, even if some of it is occluded.
        walls[0, :] = True
        walls[-1, :] = True
        walls[:, 0] = True
        walls[:, -1] = True

        if H <= 2 or W <= 2:
            return 1

        mark_segment_lines(mask, walls)
        fill_full_lines(mask, walls)

        interior = ~walls[1:-1, 1:-1]
        ih, iw = interior.shape

        if ih <= 0 or iw <= 0:
            return 1

        visited = np.zeros_like(interior, dtype=bool)
        count = 0

        for i in range(ih):
            for j in range(iw):
                if not interior[i, j] or visited[i, j]:
                    continue

                count += 1
                if count >= 4:
                    return 4

                stack = [(i, j)]
                visited[i, j] = True

                while stack:
                    r, c = stack.pop()

                    if r > 0 and interior[r - 1, c] and not visited[r - 1, c]:
                        visited[r - 1, c] = True
                        stack.append((r - 1, c))
                    if r + 1 < ih and interior[r + 1, c] and not visited[r + 1, c]:
                        visited[r + 1, c] = True
                        stack.append((r + 1, c))
                    if c > 0 and interior[r, c - 1] and not visited[r, c - 1]:
                        visited[r, c - 1] = True
                        stack.append((r, c - 1))
                    if c + 1 < iw and interior[r, c + 1] and not visited[r, c + 1]:
                        visited[r, c + 1] = True
                        stack.append((r, c + 1))

        return max(1, min(4, count))

    # ------------------------------------------------------------------
    # 3. Count rooms for each frame.
    # ------------------------------------------------------------------
    for f in frames:
        r0, r1 = f["r0"], f["r1"]
        c0, c1 = f["c0"], f["c1"]
        H = r1 - r0 + 1
        W = c1 - c0 + 1

        sub = grid[r0:r1 + 1, c0:c1 + 1]
        mask = (sub == f["color"])
        f["k"] = count_rooms(mask, H, W)

    # ------------------------------------------------------------------
    # 4. Sort and build output.
    # ------------------------------------------------------------------
    frames.sort(key=lambda f: (f["k"], f["c0"], f["r0"], f["color"]))

    out = np.full((len(frames), 4), int(filler), dtype=grid.dtype)

    for i, f in enumerate(frames):
        k = int(f["k"])
        k = max(0, min(4, k))
        if k > 0:
            out[i, :k] = f["color"]

    return out