import numpy as np
from collections import deque


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the small summary grid from a scene of colored frames.

    The input contains several rectangular frame colors and one scattered
    filler color. For each frame, we estimate how many internal divisions it
    has (partially overwritten rows/columns of its own color). A frame with
    d divisions contributes 1 + d filled cells, capped at 3. The bottommost
    frame is treated as the special 4-cell frame.

    Output rows are sorted by (filled_count, color). For a row with k filled
    cells, the first k cells are the frame color and the rest are the filler
    color.
    """
    grid = np.asarray(grid)
    if grid.size == 0:
        return grid.copy()

    def max_run_1d(arr):
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

    def max_horizontal_run(mask):
        best = 0
        for r in range(mask.shape[0]):
            run = max_run_1d(mask[r, :])
            if run > best:
                best = run
        return best

    def max_vertical_run(mask):
        best = 0
        for c in range(mask.shape[1]):
            run = max_run_1d(mask[:, c])
            if run > best:
                best = run
        return best

    # Identify frame colors: they have long horizontal and vertical runs.
    # The filler color usually appears only as scattered short marks.
    nonzero_colors = [int(c) for c in np.unique(grid) if c != 0]
    frame_colors = []
    for c in nonzero_colors:
        mask = (grid == c)
        if max_horizontal_run(mask) >= 4 and max_vertical_run(mask) >= 4:
            frame_colors.append(c)

    if not frame_colors:
        return np.zeros_like(grid)

    # Filler color is the most common non-frame non-zero color.
    non_frame_colors = [c for c in nonzero_colors if c not in frame_colors]
    if non_frame_colors:
        counts = {c: int(np.count_nonzero(grid == c)) for c in non_frame_colors}
        filler = max(counts, key=lambda x: counts[x])
    else:
        filler = 0

    def largest_component_mask(mask):
        """Return the 4-connected component of mask with the most pixels."""
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        best_pixels = []

        for i in range(h):
            for j in range(w):
                if not mask[i, j] or visited[i, j]:
                    continue

                q = deque([(i, j)])
                visited[i, j] = True
                pixels = []

                while q:
                    r, c = q.popleft()
                    pixels.append((r, c))

                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))

                if len(pixels) > len(best_pixels):
                    best_pixels = pixels

        best_mask = np.zeros_like(mask, dtype=bool)
        if best_pixels:
            rows, cols = zip(*best_pixels)
            best_mask[rows, cols] = True
        return best_mask

    def bbox_from_runs(mask):
        """
        Bounding box based on edge runs, not raw stray pixels.

        Top/bottom rows are rows with a horizontal run of at least 3.
        Left/right columns are columns with a vertical run of at least 3.
        This avoids tiny overlapping pixels extending the frame too far.
        """
        h, w = mask.shape

        row_candidates = []
        for r in range(h):
            if max_run_1d(mask[r, :]) >= 3:
                row_candidates.append(r)
        if not row_candidates:
            row_candidates = [r for r in range(h) if mask[r, :].any()]
        r0 = min(row_candidates)
        r1 = max(row_candidates)

        col_candidates = []
        for c in range(w):
            if max_run_1d(mask[:, c]) >= 3:
                col_candidates.append(c)
        if not col_candidates:
            col_candidates = [c for c in range(w) if mask[:, c].any()]
        c0 = min(col_candidates)
        c1 = max(col_candidates)

        return r0, r1, c0, c1

    frames = []
    for c in frame_colors:
        mask = (grid == c)
        comp = largest_component_mask(mask)
        if not comp.any():
            continue
        r0, r1, c0, c1 = bbox_from_runs(comp)
        if r0 > r1 or c0 > c1:
            continue
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        frames.append({
            "color": int(c),
            "r0": r0,
            "r1": r1,
            "c0": c0,
            "c1": c1,
            "area": area,
        })

    if not frames:
        return np.zeros_like(grid)

    # The bottommost frame is the special frame with 4 filled cells.
    bottom_frame = max(frames, key=lambda f: (f["r1"], f["area"]))

    for f in frames:
        if f is bottom_frame:
            f["k"] = 4
            continue

        color = f["color"]
        r0, r1, c0, c1 = f["r0"], f["r1"], f["c0"], f["c1"]
        H = r1 - r0 + 1
        W = c1 - c0 + 1

        vertical_dividers = 0
        for c in range(c0 + 1, c1):
            col = (grid[r0:r1 + 1, c] == color)
            frac = col.sum() / H
            if frac >= 0.75:
                min_run = max(3, H // 4)
                if max_run_1d(col) >= min_run:
                    vertical_dividers += 1

        horizontal_dividers = 0
        for r in range(r0 + 1, r1):
            row = (grid[r, c0:c1 + 1] == color)
            frac = row.sum() / W
            if frac >= 0.5:
                min_run = max(3, W // 4)
                if max_run_1d(row) >= min_run:
                    horizontal_dividers += 1

        k = min(3, 1 + vertical_dividers + horizontal_dividers)
        if k < 1:
            k = 1
        f["k"] = k

    # Sort rows by filled count, then by color.
    frames.sort(key=lambda f: (f["k"], f["color"]))

    out_w = max(4, max(f["k"] for f in frames))
    out = np.full((len(frames), out_w), filler, dtype=grid.dtype)

    for i, f in enumerate(frames):
        k = f["k"]
        out[i, :k] = f["color"]

    return out