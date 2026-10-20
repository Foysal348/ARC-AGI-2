import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Restore clean axis-aligned foreground lines from a noisy drawing.

    The non-background color is treated as foreground. The transform:
      - keeps long horizontal/vertical runs,
      - keeps short anchored runs,
      - completes small gaps in sufficiently long runs,
      - removes unsupported stray pixels.
    """
    arr = np.array(grid, dtype=grid.dtype)

    if arr.ndim != 2:
        return arr.copy()

    if arr.size == 0:
        return arr.copy()

    # Identify background and foreground.
    vals, counts = np.unique(arr, return_counts=True)
    background = int(vals[np.argmax(counts)])

    if counts.max() == arr.size:
        return arr.copy()

    fg_vals, fg_counts = np.unique(arr[arr != background], return_counts=True)
    foreground = int(fg_vals[np.argmax(fg_counts)])

    mask = (arr == foreground)
    H, W = arr.shape

    # A run is a contiguous block of foreground pixels in one row/column.
    def runs_1d(vec: np.ndarray):
        runs = []
        start = None
        for i, v in enumerate(vec):
            if v and start is None:
                start = i
            elif not v and start is not None:
                runs.append((start, i - 1, i - start))
                start = None
        if start is not None:
            runs.append((start, len(vec) - 1, len(vec) - start))
        return runs

    h_runs = []
    v_runs = []

    for r in range(H):
        h_runs.append(runs_1d(mask[r, :]))

    for c in range(W):
        v_runs.append(runs_1d(mask[:, c]))

    out = np.zeros((H, W), dtype=bool)

    # Thresholds.
    KEEP_RUN = 3       # short but stable runs are kept
    FILL_RUN = 5       # only long runs actively fill gaps
    MAX_GAP = 3        # maximum gap length that can be completed

    # Keep all reasonably long horizontal runs.
    for r, runs in h_runs:
        for s, e, length in runs:
            if length >= KEEP_RUN:
                out[r, s:e + 1] = True

    # Keep all reasonably long vertical runs.
    # Also keep length-2 vertical runs at the top/bottom border.
    for c, runs in v_runs:
        for s, e, length in runs:
            if length >= KEEP_RUN or (length == 2 and (s == 0 or e == H - 1)):
                out[s:e + 1, c] = True

    # Complete one small horizontal gap to the right from the longest run.
    for r, runs in h_runs:
        if not runs:
            continue

        longest = max(runs, key=lambda x: x[2])
        if longest[2] < FILL_RUN:
            continue

        for idx, (s, e, length) in enumerate(runs):
            if s == longest[0] and e == longest[1]:
                if idx + 1 < len(runs):
                    ns, ne, nl = runs[idx + 1]
                    gap = ns - e - 1
                    if 0 <= gap <= MAX_GAP:
                        out[r, e + 1:ns] = True
                        out[r, ns:ne + 1] = True
                break

    # Complete small vertical gaps downward from the longest run.
    for c, runs in v_runs:
        if not runs:
            continue

        longest = max(runs, key=lambda x: x[2])
        if longest[2] < FILL_RUN:
            continue

        for idx, (s, e, length) in enumerate(runs):
            if s == longest[0] and e == longest[1]:
                cur_e = e
                out[s:e + 1, c] = True

                j = idx + 1
                while j < len(runs):
                    ns, ne, nl = runs[j]
                    gap = ns - cur_e - 1

                    if gap < 0:
                        j += 1
                        continue

                    if gap > MAX_GAP:
                        break

                    # Stop before merging into another substantial independent run.
                    if nl >= 4:
                        break

                    out[cur_e + 1:ns, c] = True
                    out[ns:ne + 1, c] = True
                    cur_e = ne
                    j += 1

                break

    # Preserve length-2 runs that are attached to already-kept structure.
    for r, runs in h_runs:
        for s, e, length in runs:
            if length != 2:
                continue

            r1 = max(0, r - 1)
            r2 = min(H, r + 2)
            c1 = max(0, s - 1)
            c2 = min(W, e + 2)

            if out[r1:r2, c1:c2].any():
                out[r, s:e + 1] = True

    for c, runs in v_runs:
        for s, e, length in runs:
            if length != 2:
                continue

            r1 = max(0, s - 1)
            r2 = min(H, e + 2)
            c1 = max(0, c - 1)
            c2 = min(W, c + 2)

            if out[r1:r2, c1:c2].any():
                out[s:e + 1, c] = True

    # Preserve isolated pixels that are clearly aligned with an existing line.
    # This keeps useful single endpoints without retaining general noise.
    for r, c in zip(*np.where(mask & ~out)):
        # Only consider truly isolated pixels (no orthogonal foreground neighbor).
        if (c > 0 and mask[r, c - 1]) or \
           (c < W - 1 and mask[r, c + 1]) or \
           (r > 0 and mask[r - 1, c]) or \
           (r < H - 1 and mask[r + 1, c]):
            continue

        keep = False

        # Aligned vertically with an existing line.
        if out[:, c].any():
            col_rows = np.where(out[:, c])[0]
            dcol = int(np.min(np.abs(col_rows - r)))

            if dcol in (3, 4):
                perpendicular_close = False
                if out[r, :].any():
                    row_cols = np.where(out[r, :])[0]
                    drow = int(np.min(np.abs(row_cols - c)))
                    if drow <= 3:
                        perpendicular_close = True

                if dcol == 4 and not perpendicular_close:
                    keep = True
                elif r == 0 or r == H - 1:
                    keep = True

        # Aligned horizontally with an existing line.
        if not keep and out[r, :].any():
            row_cols = np.where(out[r, :])[0]
            drow = int(np.min(np.abs(row_cols - c)))

            if drow in (3, 4):
                perpendicular_close = False
                if out[:, c].any():
                    col_rows = np.where(out[:, c])[0]
                    dcol = int(np.min(np.abs(col_rows - r)))
                    if dcol <= 3:
                        perpendicular_close = True

                if drow == 4 and not perpendicular_close:
                    keep = True
                elif c == 0 or c == W - 1:
                    keep = True

        if keep:
            out[r, c] = True

    result = np.full(arr.shape, background, dtype=arr.dtype)
    result[out] = foreground
    return result