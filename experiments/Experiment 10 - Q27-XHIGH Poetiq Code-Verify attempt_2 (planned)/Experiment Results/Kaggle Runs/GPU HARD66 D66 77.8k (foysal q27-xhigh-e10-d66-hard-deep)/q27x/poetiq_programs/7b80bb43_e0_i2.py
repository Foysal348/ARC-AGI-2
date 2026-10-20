import numpy as np


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Restore broken axis-aligned line structures.

    The examples contain one background color and one foreground color.  The
    intended picture consists mainly of straight horizontal/vertical line
    segments.  The input corrupts this picture by:
      * deleting a few cells, creating short gaps;
      * shifting some line cells horizontally by 1 or 2 cells;
      * adding small diagonal/stray foreground pixels.

    This function reconstructs the intended lines by:
      1. Choosing the primary vertical line: the column with the strongest
         evidence of horizontally displaced pixels whose target cell is empty.
      2. Choosing the primary horizontal line: the row containing that vertical
         line and having a short gap close to it.
      3. Moving displaced pixels back onto the primary vertical line.
      4. Filling 1-cell vertical gaps on the primary vertical line.
      5. Removing horizontally displaced stray pixels next to long vertical
         lines when the aligned vertical cell already exists.
      6. Filling short horizontal gaps in the primary row near the primary
         vertical line.
      7. Removing pure diagonal components (components with no orthogonal
         adjacency).
      8. Removing detached short segments on the primary vertical line far
         away from the main segment, plus isolated pixels directly attached to
         the removed segment.

    The rule is intentionally conservative: it only modifies structures that
    strongly match the observed repair pattern in the examples.
    """
    grid = np.asarray(grid)
    if grid.ndim != 2 or grid.size == 0:
        return grid.copy()

    H, W = grid.shape

    # Identify background and foreground color.
    vals, counts = np.unique(grid, return_counts=True)
    background = int(vals[np.argmax(counts)])

    nonbg = grid[grid != background]
    if nonbg.size == 0:
        return grid.copy()

    obj_vals, obj_counts = np.unique(nonbg, return_counts=True)
    fg_color = int(obj_vals[np.argmax(obj_counts)])

    out = grid.copy()

    # ---------------------------------------------------------------
    # Small helpers
    # ---------------------------------------------------------------
    def runs_1d(mask):
        """Return maximal True runs as (start, end, length)."""
        mask = np.asarray(mask, dtype=bool)
        runs = []
        n = len(mask)
        i = 0
        while i < n:
            if mask[i]:
                j = i
                while j + 1 < n and mask[j + 1]:
                    j += 1
                runs.append((i, j, j - i + 1))
                i = j + 1
            else:
                i += 1
        return runs

    def compute_hlen(arr):
        """Horizontal run length at each foreground cell."""
        res = np.zeros(arr.shape, dtype=int)
        for r in range(H):
            for a, b, l in runs_1d(arr[r]):
                res[r, a:b + 1] = l
        return res

    def compute_vlen(arr):
        """Vertical run length at each foreground cell."""
        res = np.zeros(arr.shape, dtype=int)
        for c in range(W):
            for a, b, l in runs_1d(arr[:, c]):
                res[a:b + 1, c] = l
        return res

    def hlen_at(arr, r, c):
        """Horizontal run length of arr[r, c], assuming it is foreground."""
        if not (0 <= r < H and 0 <= c < W) or arr[r, c] != fg_color:
            return 0
        l = c
        while l - 1 >= 0 and arr[r, l - 1] == fg_color:
            l -= 1
        rgt = c
        while rgt + 1 < W and arr[r, rgt + 1] == fg_color:
            rgt += 1
        return rgt - l + 1

    input_fg = (grid != background).astype(bool)
    in_hlen = compute_hlen(input_fg)

    # ---------------------------------------------------------------
    # Choose primary vertical line.
    #
    # A good primary column has empty cells with foreground pixels displaced
    # horizontally by 1 or 2, and those source pixels are not part of a
    # horizontal line.
    # ---------------------------------------------------------------
    vc = None
    vc_best = None

    for c in range(W):
        col_runs = runs_1d(input_fg[:, c])
        if not col_runs:
            continue
        max_col_run = max(l for _, _, l in col_runs)
        if max_col_run < 3:
            continue

        score = 0
        for r in range(H):
            if input_fg[r, c]:
                continue
            found = False
            for d in (1, 2):
                for cc in (c - d, c + d):
                    if 0 <= cc < W and input_fg[r, cc] and in_hlen[r, cc] <= 1:
                        found = True
                        break
                if found:
                    break
            if found:
                score += 1

        if vc_best is None or (score, max_col_run) > (vc_best[0], vc_best[1]):
            vc_best = (score, max_col_run, c)

    if vc_best is not None and vc_best[0] > 0:
        vc = vc_best[2]
    else:
        # Fallback: column with a 1-cell gap between two reasonably long runs.
        fallback_best = None
        for c in range(W):
            col_runs = runs_1d(input_fg[:, c])
            if len(col_runs) < 2:
                continue
            gap_score = 0
            max_run = max(l for _, _, l in col_runs)
            for i in range(len(col_runs) - 1):
                gap = col_runs[i + 1][0] - col_runs[i][1] - 1
                if gap == 1 and col_runs[i][2] >= 2 and col_runs[i + 1][2] >= 2:
                    gap_score += 1
            if gap_score > 0:
                key = (gap_score, max_run, c)
                if fallback_best is None or key > fallback_best[0]:
                    fallback_best = (key, c)
        if fallback_best is not None:
            vc = fallback_best[1]

    # ---------------------------------------------------------------
    # Choose primary horizontal row.
    #
    # It should contain the primary vertical column and have a short gap close
    # to that column.
    # ---------------------------------------------------------------
    hr = None
    hr_best = None

    if vc is not None:
        for r in range(H):
            if not input_fg[r, vc]:
                continue

            row_runs = runs_1d(input_fg[r])
            if not row_runs:
                continue
            max_row_run = max(l for _, _, l in row_runs)
            if max_row_run < 5:
                continue

            for i in range(len(row_runs) - 1):
                gs = row_runs[i][1] + 1
                ge = row_runs[i + 1][0] - 1
                glen = ge - gs + 1
                if 1 <= glen <= 3:
                    dist = min(abs(gs - vc), abs(ge - vc))
                    if dist <= 3:
                        score = max_row_run + glen + (10 - dist)
                        if hr_best is None or score > hr_best[0]:
                            hr_best = (score, r)

        if hr_best is not None:
            hr = hr_best[1]

    # ---------------------------------------------------------------
    # Move horizontally displaced pixels back onto the primary vertical line.
    # ---------------------------------------------------------------
    if vc is not None:
        for r in range(H):
            if out[r, vc] != background:
                continue

            # Prefer distance 1 over distance 2.
            for d in (1, 2):
                if out[r, vc] != background:
                    break

                for cc in (vc - d, vc + d):
                    if 0 <= cc < W and out[r, cc] == fg_color:
                        # Do not steal pixels that already form a horizontal run.
                        if hlen_at(out, r, cc) <= 1:
                            out[r, vc] = fg_color
                            out[r, cc] = background
                            break

    # ---------------------------------------------------------------
    # Fill 1-cell vertical gaps on the primary vertical line.
    # ---------------------------------------------------------------
    if vc is not None:
        for r in range(1, H - 1):
            if (
                out[r, vc] == background
                and out[r - 1, vc] == fg_color
                and out[r + 1, vc] == fg_color
            ):
                out[r, vc] = fg_color

    # ---------------------------------------------------------------
    # Remove horizontally displaced stray pixels next to long vertical lines.
    #
    # If a long vertical line already has a foreground pixel at (r, c), and a
    # nearby isolated pixel sits at (r, c +/- d), that nearby pixel is usually
    # noise rather than a missing continuation.
    # ---------------------------------------------------------------
    cur_fg = (out != background).astype(bool)
    cur_hlen = compute_hlen(cur_fg)
    cur_vlen = compute_vlen(cur_fg)

    stray_remove = set()

    for r in range(H):
        for c in range(W):
            if out[r, c] != fg_color or cur_vlen[r, c] < 5:
                continue

            for d in (1, 2, 3):
                for cc in (c - d, c + d):
                    if not (0 <= cc < W):
                        continue
                    if out[r, cc] != fg_color:
                        continue

                    # Skip pixels that already belong to a short vertical line
                    # or any horizontal line.
                    if cur_vlen[r, cc] >= 2:
                        continue
                    if cur_hlen[r, cc] >= 2:
                        continue

                    stray_remove.add((r, cc))

    for r, c in stray_remove:
        out[r, c] = background

    # ---------------------------------------------------------------
    # Fill short horizontal gaps in the primary row near the primary column.
    # ---------------------------------------------------------------
    if hr is not None and vc is not None:
        row_runs = runs_1d(out[hr])
        for i in range(len(row_runs) - 1):
            gs = row_runs[i][1] + 1
            ge = row_runs[i + 1][0] - 1
            glen = ge - gs + 1
            if 1 <= glen <= 3:
                dist = min(abs(gs - vc), abs(ge - vc))
                if dist <= 3:
                    out[hr, gs:ge + 1] = fg_color

    # ---------------------------------------------------------------
    # Remove pure diagonal components: 8-connected components of size >= 2
    # that contain no orthogonal adjacency.
    # ---------------------------------------------------------------
    cur_fg = (out != background).astype(bool)
    visited = np.zeros((H, W), dtype=bool)
    remove = set()

    for r in range(H):
        for c in range(W):
            if not cur_fg[r, c] or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            comp = []

            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))

                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if out[nr, nc] == fg_color and not visited[nr, nc]:
                                visited[nr, nc] = True
                                stack.append((nr, nc))

            has_orth = False
            for rr, cc in comp:
                if any(
                    0 <= rr + dr < H and 0 <= cc + dc < W and out[rr + dr, cc + dc] == fg_color
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                ):
                    has_orth = True
                    break

            if len(comp) >= 2 and not has_orth:
                remove.update(comp)

    for r, c in remove:
        out[r, c] = background

    # ---------------------------------------------------------------
    # Remove detached short segments on the primary vertical line.
    #
    # In Example 1, the primary vertical line has a small detached fragment
    # far below the main segment; it should disappear.  Also remove isolated
    # pixels directly attached to the removed fragment.
    # ---------------------------------------------------------------
    if vc is not None and hr is not None:
        col_runs = runs_1d(out[:, vc] == fg_color)

        if col_runs:
            primary = None
            for a, b, _ in col_runs:
                if a <= hr <= b:
                    primary = (a, b, b - a + 1)
                    break

            if primary is None:
                primary = max(col_runs, key=lambda x: x[2])

            removed = []
            for a, b, l in col_runs:
                if (a, b, l) == primary:
                    continue

                if b < primary[0]:
                    gap = primary[0] - b - 1
                elif a > primary[1]:
                    gap = a - primary[1] - 1
                else:
                    continue

                if gap > 3 and l < primary[2]:
                    for rr in range(a, b + 1):
                        removed.append((rr, vc))

            for rr, cc in removed:
                out[rr, cc] = background

            # Remove isolated pixels 8-adjacent to the removed fragment.
            cur_fg = (out != background).astype(bool)
            cur_hlen = compute_hlen(cur_fg)
            cur_vlen = compute_vlen(cur_fg)

            for rr, cc in removed:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = rr + dr, cc + dc
                        if not (0 <= nr < H and 0 <= nc < W):
                            continue
                        if out[nr, nc] != fg_color:
                            continue
                        if nr == hr or nc == vc:
                            continue
                        if cur_hlen[nr, nc] <= 1 and cur_vlen[nr, nc] <= 1:
                            out[nr, nc] = background

    return out