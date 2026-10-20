import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct clean axis-aligned line segments from a noisy single-color drawing.

    The transformation works as follows:
      1. Treat the most frequent color as background.
      2. Treat the most frequent non-background color as foreground.
      3. Keep strong horizontal/vertical runs.
      4. For each row, if the longest horizontal run has a short gap to the right
         followed by a short run, complete that gap.
      5. For each column, if the longest vertical run has a short gap downward
         followed by a short run, complete that gap.
      6. Keep short runs only when they are clearly connected to already-kept
         structure, which preserves small endpoints without keeping stray noise.
    """
    arr = np.asarray(grid)

    if arr.ndim != 2:
        return arr.copy()

    if arr.size == 0:
        return arr.copy()

    # Identify background and the dominant foreground color.
    vals, counts = np.unique(arr, return_counts=True)
    background = int(vals[np.argmax(counts)])

    nonbg = arr != background
    if not nonbg.any():
        return arr.copy()

    fg_vals, fg_counts = np.unique(arr[nonbg], return_counts=True)
    foreground = int(fg_vals[np.argmax(fg_counts)])

    mask = arr == foreground
    H, W = arr.shape

    def runs_1d(vec):
        """Return maximal contiguous True runs as (start, end, length)."""
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

    # Thresholds chosen from the examples:
    # - length 3 is already a real segment,
    # - gaps up to 3 can be missing pixels,
    # - only short terminal runs are absorbed during completion.
    KEEP_RUN = 3
    MAX_GAP = 3
    SHORT_TERMINAL = 3

    out = np.zeros((H, W), dtype=bool)

    h_runs = [runs_1d(mask[r]) for r in range(H)]
    v_runs = [runs_1d(mask[:, c]) for c in range(W)]

    # Keep all clearly real horizontal segments.
    for r, runs in enumerate(h_runs):
        for s, e, length in runs:
            if length >= KEEP_RUN:
                out[r, s:e + 1] = True

    # Keep all clearly real vertical segments.
    # Also keep border length-2 vertical segments, since they often appear
    # as short visible endpoints in these puzzles.
    for c, runs in enumerate(v_runs):
        for s, e, length in runs:
            if length >= KEEP_RUN or (length == 2 and (s == 0 or e == H - 1)):
                out[s:e + 1, c] = True

    # Complete one short horizontal gap to the right of the longest run in a row.
    for r, runs in enumerate(h_runs):
        if not runs:
            continue

        longest = max(runs, key=lambda x: x[2])
        if longest[2] < KEEP_RUN:
            continue

        for idx, (s, e, length) in enumerate(runs):
            if (s, e) != (longest[0], longest[1]):
                continue

            if idx + 1 < len(runs):
                ns, ne, nl = runs[idx + 1]
                gap = ns - e - 1
                if 0 <= gap <= MAX_GAP and nl <= SHORT_TERMINAL:
                    out[r, e + 1:ns] = True
                    out[r, ns:ne + 1] = True
            break

    # Complete short vertical gaps downward from the longest run in a column.
    for c, runs in enumerate(v_runs):
        if not runs:
            continue

        longest = max(runs, key=lambda x: x[2])
        if longest[2] < KEEP_RUN:
            continue

        for idx, (s, e, length) in enumerate(runs):
            if (s, e) != (longest[0], longest[1]):
                continue

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

                # Do not merge into another substantial independent run.
                if nl > SHORT_TERMINAL:
                    break

                out[cur_e + 1:ns, c] = True
                out[ns:ne + 1, c] = True
                cur_e = ne
                j += 1

            break

    # Keep short horizontal/vertical runs if they touch or are immediately
    # adjacent to already-kept structure. This preserves small endpoints
    # while still removing isolated noise.
    for r, runs in enumerate(h_runs):
        for s, e, length in runs:
            if length >= KEEP_RUN:
                continue

            r1 = max(0, r - 1)
            r2 = min(H, r + 2)
            c1 = max(0, s - 1)
            c2 = min(W, e + 2)

            if out[r1:r2, c1:c2].any():
                out[r, s:e + 1] = True

    for c, runs in enumerate(v_runs):
        for s, e, length in runs:
            if length >= KEEP_RUN:
                continue

            r1 = max(0, s - 1)
            r2 = min(H, e + 2)
            c1 = max(0, c - 1)
            c2 = min(W, c + 2)

            if out[r1:r2, c1:c2].any():
                out[s:e + 1, c] = True

    result = np.full(arr.shape, background, dtype=arr.dtype)
    result[out] = foreground
    return result