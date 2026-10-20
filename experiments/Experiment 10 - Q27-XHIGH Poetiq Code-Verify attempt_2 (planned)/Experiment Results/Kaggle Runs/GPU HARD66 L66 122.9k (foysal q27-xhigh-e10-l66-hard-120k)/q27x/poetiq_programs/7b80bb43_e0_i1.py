import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Recover clean horizontal/vertical line segments from a noisy foreground drawing.

    The non-background color is treated as foreground. The function:
      - keeps strong axis-aligned line segments,
      - fills small gaps along those segments,
      - removes stray/noise pixels that are not part of the recovered lines.

    This heuristic is designed for ARC-style line-completion/noise-removal grids.
    """
    grid = np.asarray(grid)

    if grid.ndim != 2:
        return grid.copy()

    if grid.size == 0:
        return grid.copy()

    # Background is the most frequent color.
    vals, counts = np.unique(grid, return_counts=True)
    background = int(vals[np.argmax(counts)])

    nonbg = grid != background
    if not nonbg.any():
        return grid.copy()

    # Foreground is the most frequent non-background color.
    fg_vals, fg_counts = np.unique(grid[nonbg], return_counts=True)
    foreground = int(fg_vals[np.argmax(fg_counts)])

    mask = grid == foreground
    H, W = grid.shape

    out = np.full(grid.shape, background, dtype=grid.dtype)

    TOL = 2  # tolerance used to find supporting pixels near a line

    def runs_1d(arr: np.ndarray):
        """Return contiguous True runs in a 1D boolean array as (start, end, length)."""
        idx = np.where(arr)[0]
        runs = []
        if len(idx) == 0:
            return runs

        start = prev = int(idx[0])
        for cur in idx[1:]:
            cur = int(cur)
            if cur == prev + 1:
                prev = cur
            else:
                runs.append((start, prev, prev - start + 1))
                start = prev = cur

        runs.append((start, prev, prev - start + 1))
        return runs

    def has_vertical_support(col: int, r1: int, r2: int) -> bool:
        """Whether any foreground pixel exists in the gap rows near this column."""
        if r1 > r2:
            return True
        c1 = max(0, col - TOL)
        c2 = min(W - 1, col + TOL)
        return bool(mask[r1:r2 + 1, c1:c2 + 1].any())

    vertical_segments = set()

    # ------------------------------------------------------------------
    # Vertical line recovery
    # ------------------------------------------------------------------
    for c in range(W):
        runs = runs_1d(mask[:, c])
        n = len(runs)
        if n == 0:
            continue

        for i, (s, e, length) in enumerate(runs):
            base = False

            # Strong enough to be a line on its own.
            if length >= 3:
                base = True

            # Short border segment.
            elif length >= 2 and (s == 0 or e == H - 1):
                base = True

            # Single border pixel is kept only if it is close to another run.
            elif length == 1 and (s == 0 or e == H - 1):
                for j, (s2, e2, _l2) in enumerate(runs):
                    if i == j:
                        continue
                    if s2 > e:
                        gap = s2 - e - 1
                    elif e2 < s:
                        gap = s - e2 - 1
                    else:
                        gap = 0
                    if gap <= 4:
                        base = True
                        break

            if not base:
                continue

            # Extend downward.
            cur_s, cur_e = s, e
            j = i + 1
            while j < n:
                ns, ne, nl = runs[j]
                gap = ns - cur_e - 1

                if gap > 3:
                    break
                if nl > 3:
                    break

                if gap <= 1:
                    bridge = True
                else:
                    bridge = has_vertical_support(c, cur_e + 1, ns - 1)

                if not bridge:
                    break

                cur_e = ne
                j += 1

            vertical_segments.add((c, cur_s, cur_e))

            # Extend upward as well.
            cur_s, cur_e = s, e
            j = i - 1
            while j >= 0:
                ns, ne, nl = runs[j]
                gap = cur_s - ne - 1

                if gap > 3:
                    break
                if nl > 3:
                    break

                if gap <= 1:
                    bridge = True
                else:
                    bridge = has_vertical_support(c, ne + 1, cur_s - 1)

                if not bridge:
                    break

                cur_s = ns
                j -= 1

            vertical_segments.add((c, cur_s, cur_e))

    # ------------------------------------------------------------------
    # Horizontal line recovery
    # ------------------------------------------------------------------
    horizontal_segments = set()

    for r in range(H):
        runs = runs_1d(mask[r, :])
        n = len(runs)
        if n == 0:
            continue

        for i, (s, e, length) in enumerate(runs):
            # Use only sufficiently strong horizontal runs as anchors.
            if length < 3:
                continue

            cur_s, cur_e = s, e

            # Fill at most one short gap to the right.
            # This completes short endpoint gaps while avoiding chaining
            # across multiple unrelated short runs.
            j = i + 1
            if j < n:
                ns, ne, nl = runs[j]
                gap = ns - cur_e - 1
                if gap <= 3 and nl <= 3:
                    cur_e = ne

            horizontal_segments.add((r, cur_s, cur_e))

    # ------------------------------------------------------------------
    # Draw recovered segments
    # ------------------------------------------------------------------
    for c, r1, r2 in vertical_segments:
        if r1 < 0:
            r1 = 0
        if r2 >= H:
            r2 = H - 1
        if 0 <= c < W and r1 <= r2:
            out[r1:r2 + 1, c] = foreground

    for r, c1, c2 in horizontal_segments:
        if c1 < 0:
            c1 = 0
        if c2 >= W:
            c2 = W - 1
        if 0 <= r < H and c1 <= c2:
            out[r, c1:c2 + 1] = foreground

    return out