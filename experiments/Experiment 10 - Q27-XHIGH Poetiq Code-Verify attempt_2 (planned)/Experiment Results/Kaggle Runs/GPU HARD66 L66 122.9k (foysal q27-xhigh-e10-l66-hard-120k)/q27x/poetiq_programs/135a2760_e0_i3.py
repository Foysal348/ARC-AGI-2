import numpy as np


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Complete noisy periodic patterns inside framed rectangular panels.

    Each panel is a thin rectangular frame. Its interior is filled with a
    small repeated tile, with a few corrupted cells.  The function:
      1. Finds thin rectangular frames that enclose a region containing the
         global background and at least one non-background pattern color.
      2. For each panel interior, searches for a plausible 2D period.
      3. Reconstructs the tile by majority voting and repairs only the
         non-fixed cells of the panel.

    Candidate periods must repeat enough to be reliable, and a candidate is
    rejected if it would erase a sufficiently frequent observed color.
    """
    out = np.array(grid, dtype=int, copy=True)
    if out.ndim != 2 or out.size == 0:
        return out

    h, w = out.shape
    base = out.copy()

    # Global background: mode of the outer border.  If that fails to find any
    # panels, retry with the mode of the whole grid.
    border = np.concatenate((out[0, :], out[-1, :], out[:, 0], out[:, -1]))
    border_vals, border_counts = np.unique(border, return_counts=True)
    global_bg = int(border_vals[np.argmax(border_counts)])

    def _find_frames(bg: int):
        """
        Find thin rectangular frames.

        A frame is a 4-connected component of one non-background color whose
        component size equals the perimeter of its bounding box.  The interior
        must contain the global background and at least one pattern color.
        """
        frames = []
        seen = set()

        for raw_color in np.unique(base):
            color = int(raw_color)
            if color == bg:
                continue

            mask = (base == color)
            if mask.sum() < 8:
                continue

            visited = np.zeros((h, w), dtype=bool)

            for r0 in range(h):
                for c0 in range(w):
                    if not mask[r0, c0] or visited[r0, c0]:
                        continue

                    stack = [(r0, c0)]
                    visited[r0, c0] = True

                    min_r = max_r = r0
                    min_c = max_c = c0
                    size = 0

                    while stack:
                        r, c = stack.pop()
                        size += 1

                        if r < min_r:
                            min_r = r
                        if r > max_r:
                            max_r = r
                        if c < min_c:
                            min_c = c
                        if c > max_c:
                            max_c = c

                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = r + dr, c + dc
                            if (
                                0 <= nr < h
                                and 0 <= nc < w
                                and mask[nr, nc]
                                and not visited[nr, nc]
                            ):
                                visited[nr, nc] = True
                                stack.append((nr, nc))

                    ht = max_r - min_r + 1
                    wd = max_c - min_c + 1

                    if ht < 3 or wd < 3:
                        continue

                    perimeter = 2 * ht + 2 * wd - 4
                    if size != perimeter:
                        continue

                    # The bounding-box perimeter must be the frame color.
                    if not (
                        np.all(base[min_r, min_c:max_c + 1] == color)
                        and np.all(base[max_r, min_c:max_c + 1] == color)
                        and np.all(base[min_r:max_r + 1, min_c] == color)
                        and np.all(base[min_r:max_r + 1, max_c] == color)
                    ):
                        continue

                    # Trim full frame-color rows/columns from the boundary.
                    top = min_r
                    while top < max_r and np.all(base[top, min_c:max_c + 1] == color):
                        top += 1

                    bottom = max_r
                    while bottom > top and np.all(base[bottom, min_c:max_c + 1] == color):
                        bottom -= 1

                    left = min_c
                    while left < max_c and np.all(base[top:bottom + 1, left] == color):
                        left += 1

                    right = max_c
                    while right > left and np.all(base[top:bottom + 1, right] == color):
                        right -= 1

                    if top >= bottom or left >= right:
                        continue

                    interior = base[top:bottom + 1, left:right + 1]
                    if interior.size < 2:
                        continue

                    if np.all(interior == color):
                        continue

                    # Panel interiors in this task contain the global
                    # background plus at least one pattern color.
                    if not np.any(interior == bg):
                        continue
                    if not np.any((interior != color) & (interior != bg)):
                        continue

                    key = (top, bottom, left, right)
                    if key in seen:
                        continue
                    seen.add(key)

                    area = (bottom - top + 1) * (right - left + 1)
                    frames.append(
                        (top, bottom, left, right,
                         min_r, max_r, min_c, max_c,
                         color, area)
                    )

        return frames

    def _complete_pattern(arr: np.ndarray, fixed_mask: np.ndarray, bg_global: int) -> np.ndarray:
        """
        Complete the periodic pattern in one panel interior.

        fixed_mask cells are preserved unchanged and ignored when voting for
        the periodic tile.
        """
        H, W = arr.shape
        if H == 0 or W == 0:
            return arr

        if fixed_mask is None:
            fixed_mask = np.zeros((H, W), dtype=bool)
        else:
            fixed_mask = np.asarray(fixed_mask, dtype=bool)

        obs_mask = ~fixed_mask
        n_obs = int(np.sum(obs_mask))
        if n_obs < 2:
            return arr

        rows, cols = np.nonzero(obs_mask)
        vals = arr[rows, cols]
        colors = np.unique(vals)

        if colors.size <= 1:
            return arr

        # Choose the panel background: prefer the global background if present.
        if bg_global in colors:
            bg_val = int(bg_global)
        else:
            _, counts = np.unique(vals, return_counts=True)
            bg_val = int(colors[np.argmax(counts)])

        C = len(colors)
        color_to_idx = {int(c): i for i, c in enumerate(colors)}
        val_idx = np.array([color_to_idx[int(v)] for v in vals], dtype=np.int64)
        bg_idx = color_to_idx[int(bg_val)]

        # Colors that appear often enough should be preserved.
        obs_counts = np.bincount(val_idx, minlength=C)
        min_significant = max(2, int(0.03 * n_obs) + 1)
        significant = obs_counts >= min_significant

        def _choose_tile(counts: np.ndarray):
            """
            Choose a tile color for each tile position by majority vote.
            Ties prefer the background.  If a significant color would be
            missing but is tied for a tile cell, use that cell to preserve it
            without increasing the mismatch count.
            """
            T = counts.shape[0]
            maxc = counts.max(axis=1)
            tile = np.zeros(T, dtype=np.int64)

            for t in range(T):
                if maxc[t] == 0:
                    tile[t] = bg_idx
                else:
                    tied = np.flatnonzero(counts[t] == maxc[t])
                    if bg_idx in tied:
                        tile[t] = bg_idx
                    else:
                        tile[t] = int(tied[0])

            present = np.bincount(tile, minlength=C)

            # Preserve significant colors if possible without changing mismatches.
            for _ in range(C):
                missing = np.where(significant & (present == 0))[0]
                if missing.size == 0:
                    break

                c = int(missing[0])
                found = -1
                for t in range(T):
                    if maxc[t] > 0 and counts[t, c] == maxc[t] and tile[t] != c:
                        found = t
                        break

                if found == -1:
                    break

                tile[found] = c
                present = np.bincount(tile, minlength=C)

            present = np.bincount(tile, minlength=C)
            if not np.all(significant & (present > 0)):
                return None

            return tile

        # Try stricter repetition requirements first.
        valid_candidates = []
        for min_rep in (4, 3, 2, 1):
            periods = []

            for ph in range(1, H + 1):
                if ph < H and (H // ph) < 2:
                    continue
                rep_h = 1 if ph == H else H // ph

                for pw in range(1, W + 1):
                    if pw < W and (W // pw) < 2:
                        continue

                    # Full interior is not a useful periodic hypothesis.
                    if ph == H and pw == W:
                        continue

                    rep_w = 1 if pw == W else W // pw
                    if rep_h * rep_w < min_rep:
                        continue

                    periods.append((ph, pw))

            if not periods:
                continue

            candidates = []

            for ph, pw in periods:
                counts = np.zeros((ph * pw, C), dtype=np.int64)
                tile_idx = (rows % ph) * pw + (cols % pw)
                np.add.at(counts, (tile_idx, val_idx), 1)

                tile = _choose_tile(counts)
                if tile is None:
                    continue

                pred = tile[tile_idx]
                mismatch = int(np.sum(pred != val_idx))
                area = ph * pw

                # Slight penalty for larger tiles, but mismatches dominate.
                score = 20 * mismatch + area
                candidates.append(
                    (score, area, mismatch, ph, pw, tile)
                )

            if candidates:
                valid_candidates = candidates
                break

        if not valid_candidates:
            return arr

        # Best candidate: lowest score; tie-break by smaller tile, then period.
        valid_candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
        _, _, _, best_ph, best_pw, best_tile = valid_candidates[0]

        # Apply reconstruction only to non-fixed cells.
        tile_idx = (rows % best_ph) * best_pw + (cols % best_pw)
        pred = best_tile[tile_idx]
        arr[rows, cols] = colors[pred]
        return arr

    frames = _find_frames(global_bg)

    # Fallback: if border-mode background was wrong, try overall mode.
    if not frames:
        all_vals, all_counts = np.unique(base, return_counts=True)
        overall_bg = int(all_vals[np.argmax(all_counts)])
        if overall_bg != global_bg:
            frames2 = _find_frames(overall_bg)
            if frames2:
                frames = frames2
                global_bg = overall_bg

    if not frames:
        return out

    # Process smaller panels first; this also helps with nested frames.
    frames.sort(key=lambda f: (f[9], f[0], f[2]))

    locked = np.zeros((h, w), dtype=bool)

    for (top, bottom, left, right,
         min_r, max_r, min_c, max_c,
         frame_color, _area) in frames:

        interior_slice = (slice(top, bottom + 1), slice(left, right + 1))

        if locked[interior_slice].all():
            continue

        sub = out[interior_slice].copy()
        fixed = locked[interior_slice].copy()

        # Keep the frame color itself fixed if it leaks into the interior.
        fixed |= (sub == frame_color)

        out[interior_slice] = _complete_pattern(sub, fixed, global_bg)

        # Lock the whole frame bounding box so outer panels do not repaint it.
        locked[min_r:max_r + 1, min_c:max_c + 1] = True

    return out