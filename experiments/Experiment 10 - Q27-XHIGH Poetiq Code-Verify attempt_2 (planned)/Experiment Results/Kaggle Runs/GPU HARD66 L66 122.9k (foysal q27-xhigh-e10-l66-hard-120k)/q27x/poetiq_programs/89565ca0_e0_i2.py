import numpy as np


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a scene of colored rectangular frames into a small summary grid.

    For each frame color, estimate how many internal line segments it has.
    A frame with d detected internal divisions contributes d + 1 filled cells,
    capped at 4. Output rows are sorted by filled-cell count, then by position.
    The blank cells use the most common non-frame filler color.
    """
    grid = np.asarray(grid)
    if grid.size == 0 or grid.ndim != 2:
        return grid.copy()

    def max_run_bool(arr):
        """Maximum number of consecutive True values in a 1D boolean array."""
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

    def mark_long_runs(mask, min_len):
        """
        Return a boolean mask of pixels that lie in a horizontal or vertical
        run of True values of length at least min_len.
        """
        h, w = mask.shape
        support = np.zeros_like(mask, dtype=bool)

        # Horizontal runs.
        for i in range(h):
            run = 0
            for j in range(w + 1):
                if j < w and mask[i, j]:
                    run += 1
                else:
                    if run >= min_len:
                        support[i, j - run:j] = True
                    run = 0

        # Vertical runs.
        for j in range(w):
            run = 0
            for i in range(h + 1):
                if i < h and mask[i, j]:
                    run += 1
                else:
                    if run >= min_len:
                        support[i - run:i, j] = True
                    run = 0

        return support

    def max_horizontal_run_color(color):
        mask = (grid == color)
        best = 0
        for r in range(mask.shape[0]):
            best = max(best, max_run_bool(mask[r, :]))
        return best

    def max_vertical_run_color(color):
        mask = (grid == color)
        best = 0
        for c in range(mask.shape[1]):
            best = max(best, max_run_bool(mask[:, c]))
        return best

    def supported_bbox(color, min_run=3):
        """
        Bounding box based on pixels belonging to long runs of this color.
        This makes the bbox less sensitive to isolated stray pixels.
        """
        mask = (grid == color)
        support = mark_long_runs(mask, min_run)

        # Fallback if support is too small.
        if support.sum() < 4:
            support = mask

        if not support.any():
            return None

        rows = np.where(support.any(axis=1))[0]
        cols = np.where(support.any(axis=0))[0]
        return (int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1]))

    # ------------------------------------------------------------------
    # 1. Identify frame colors.
    # ------------------------------------------------------------------
    nonzero_colors = [int(c) for c in np.unique(grid) if c != 0]
    frame_colors = []

    for c in nonzero_colors:
        mh = max_horizontal_run_color(c)
        mv = max_vertical_run_color(c)

        # A frame should have at least a short horizontal and vertical outline.
        if mh >= 3 and mv >= 3:
            bbox = supported_bbox(c, min_run=3)
            if bbox is None:
                continue
            r0, r1, c0, c1 = bbox
            if (r1 - r0 + 1) >= 3 and (c1 - c0 + 1) >= 3:
                frame_colors.append(c)

    if not frame_colors:
        return np.zeros_like(grid)

    frame_set = set(frame_colors)

    # ------------------------------------------------------------------
    # 2. Identify filler/noise color: most common non-frame non-zero color.
    # ------------------------------------------------------------------
    non_frame_colors = [c for c in nonzero_colors if c not in frame_set]
    if non_frame_colors:
        filler = max(non_frame_colors, key=lambda x: int(np.count_nonzero(grid == x)))
    else:
        filler = 0

    # ------------------------------------------------------------------
    # 3. Count internal divisions for each frame.
    # ------------------------------------------------------------------
    def count_frame_lines(color, bbox):
        r0, r1, c0, c1 = bbox
        h = r1 - r0 + 1
        w = c1 - c0 + 1

        if h < 3 or w < 3:
            return 1

        vertical = 0
        horizontal = 0

        # ---------------- Vertical lines ----------------
        for c in range(c0 + 1, c1):
            col_f = (grid[r0:r1 + 1, c] == color)
            max_f = max_run_bool(col_f)
            f_count = int(col_f.sum())

            # Strong evidence from the frame's own color.
            if (
                max_f >= max(3, h // 2)
                or (
                    max_f >= max(2, h // 4)
                    and f_count >= max(3, (h + 1) // 2)
                )
            ):
                vertical += 1
                continue

            # Evidence from a non-zero wall segment, possibly hidden by
            # another frame/noise, but still aligned with this frame.
            col_wall = (grid[r0:r1 + 1, c] != 0)
            max_wall = max_run_bool(col_wall)

            if max_wall >= max(3, h // 2):
                outside = (
                    (r0 > 0 and grid[r0 - 1, c] != 0)
                    or (r1 + 1 < grid.shape[0] and grid[r1 + 1, c] != 0)
                )
                if outside:
                    top_f = col_f[:min(2, h)].any()
                    bottom_f = col_f[-min(2, h):].any()
                    if top_f and bottom_f and f_count >= max(2, h // 4):
                        vertical += 1

        # ---------------- Horizontal lines ----------------
        for r in range(r0 + 1, r1):
            row_f = (grid[r, c0:c1 + 1] == color)
            max_f = max_run_bool(row_f)
            f_count = int(row_f.sum())

            # Strong evidence from the frame's own color.
            if (
                max_f >= max(3, w // 2)
                or (
                    max_f >= max(2, w // 4)
                    and f_count >= max(3, (w + 1) // 2)
                )
            ):
                horizontal += 1
                continue

            # Evidence from a non-zero wall segment crossing the frame.
            row_wall = (grid[r, c0:c1 + 1] != 0)
            max_wall = max_run_bool(row_wall)

            if max_wall >= max(3, w // 2):
                outside = (
                    (c0 > 0 and grid[r, c0 - 1] != 0)
                    or (c1 + 1 < grid.shape[1] and grid[r, c1 + 1] != 0)
                )
                if outside:
                    left_f = row_f[:min(2, w)].any()
                    right_f = row_f[-min(2, w):].any()
                    if left_f and right_f and f_count >= max(2, w // 4):
                        horizontal += 1

        k = 1 + vertical + horizontal
        return max(1, min(4, k))

    frames = []
    for c in frame_colors:
        bbox = supported_bbox(c, min_run=3)
        if bbox is None:
            mask = (grid == c)
            if not mask.any():
                continue
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            bbox = (int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1]))

        r0, r1, c0, c1 = bbox
        k = count_frame_lines(c, bbox)
        frames.append({
            "color": int(c),
            "k": int(k),
            "r0": int(r0),
            "c0": int(c0),
        })

    if not frames:
        return np.zeros_like(grid)

    # ------------------------------------------------------------------
    # 4. Sort and build output.
    # ------------------------------------------------------------------
    frames.sort(key=lambda f: (f["k"], f["r0"], f["c0"], f["color"]))

    out = np.full((len(frames), 4), filler, dtype=grid.dtype)
    for i, f in enumerate(frames):
        k = f["k"]
        if k > 0:
            out[i, :k] = f["color"]

    return out