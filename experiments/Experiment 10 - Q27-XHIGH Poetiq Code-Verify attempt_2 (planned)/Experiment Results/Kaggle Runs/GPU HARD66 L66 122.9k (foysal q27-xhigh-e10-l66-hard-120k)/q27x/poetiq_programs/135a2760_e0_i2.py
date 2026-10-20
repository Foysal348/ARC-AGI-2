import numpy as np


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Complete noisy periodic patterns inside framed rectangular panels.

    The input contains one or more panels framed by a non-background color.
    Inside each panel, the background color is normally the global background
    color, and one or more pattern colors form a repeated 2D tile with a few
    corrupted/noise cells.

    This function:
      1. Finds framed rectangular panels.
      2. Trims the frame border (including possible multi-cell thick borders).
      3. Reconstructs each panel interior by choosing a simple period whose
         repeated tile best matches the observed non-fixed cells.
      4. Leaves frames and already-processed nested panels unchanged.
    """
    out = np.array(grid, dtype=int, copy=True)
    if out.size == 0 or out.ndim != 2:
        return out

    h, w = out.shape

    # Global background is the most common color on the outer border.
    border = np.concatenate((out[0, :], out[-1, :], out[:, 0], out[:, -1]))
    border_vals, border_counts = np.unique(border, return_counts=True)
    global_bg = int(border_vals[np.argmax(border_counts)])

    def _find_frames(require_global: bool):
        """
        Find rectangular frame components.

        A frame is a connected component of a non-global-background color whose
        bounding-box perimeter is entirely that color. The interior is obtained
        by trimming all consecutive full frame-color rows/columns from the
        boundary.
        """
        frames = []
        seen_boxes = set()

        for raw_color in np.unique(out):
            color = int(raw_color)
            if color == global_bg:
                continue

            mask = (out == color)
            if mask.sum() < 8:
                continue

            visited = np.zeros((h, w), dtype=bool)

            for r0 in range(h):
                for c0 in range(w):
                    if not mask[r0, c0] or visited[r0, c0]:
                        continue

                    # 4-connected component.
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

                    if size < 8:
                        continue

                    ht = max_r - min_r + 1
                    wd = max_c - min_c + 1
                    if ht < 3 or wd < 3:
                        continue

                    # The bounding-box perimeter must be all frame color.
                    if not (
                        np.all(out[min_r, min_c:max_c + 1] == color)
                        and np.all(out[max_r, min_c:max_c + 1] == color)
                        and np.all(out[min_r:max_r + 1, min_c] == color)
                        and np.all(out[min_r:max_r + 1, max_c] == color)
                    ):
                        continue

                    # Trim full frame-color rows/columns from the outside.
                    top = min_r
                    while top < max_r and np.all(out[top, min_c:max_c + 1] == color):
                        top += 1

                    bottom = max_r
                    while bottom > top and np.all(out[bottom, min_c:max_c + 1] == color):
                        bottom -= 1

                    left = min_c
                    while left < max_c and np.all(out[top:bottom + 1, left] == color):
                        left += 1

                    right = max_c
                    while right > left and np.all(out[top:bottom + 1, right] == color):
                        right -= 1

                    if top >= bottom or left >= right:
                        continue

                    interior = out[top:bottom + 1, left:right + 1]
                    if interior.size < 2:
                        continue

                    if np.all(interior == color):
                        continue

                    # In the normal task, panel interiors contain the global
                    # background plus at least one pattern color.
                    if require_global:
                        if not np.any(interior == global_bg):
                            continue
                        if not np.any((interior != color) & (interior != global_bg)):
                            continue
                    else:
                        non_frame = interior[interior != color]
                        if np.unique(non_frame).size < 2:
                            continue

                    key = (min_r, max_r, min_c, max_c)
                    if key in seen_boxes:
                        continue
                    seen_boxes.add(key)

                    area = (bottom - top + 1) * (right - left + 1)
                    frames.append(
                        (top, bottom, left, right, min_r, max_r, min_c, max_c, color, area)
                    )

        return frames

    def _complete_region(arr: np.ndarray, fixed_mask: np.ndarray) -> np.ndarray:
        """
        Complete a noisy periodic pattern in one rectangular region.

        fixed_mask cells are not changed and are ignored when voting for the
        periodic tile.
        """
        arr = np.array(arr, dtype=int, copy=True)
        H, W = arr.shape

        if H == 0 or W == 0:
            return arr

        if fixed_mask is None or fixed_mask.shape != arr.shape:
            fixed_mask = np.zeros((H, W), dtype=bool)
        else:
            fixed_mask = np.asarray(fixed_mask, dtype=bool)

        observed = ~fixed_mask
        if not observed.any():
            return arr

        obs_vals = arr[observed]
        unique_colors, color_counts_arr = np.unique(obs_vals, return_counts=True)

        if len(unique_colors) <= 1:
            return arr

        # Choose the panel background. Prefer global background if present.
        if np.any(unique_colors == global_bg):
            bg_val = int(global_bg)
        else:
            bg_val = int(unique_colors[np.argmax(color_counts_arr)])

        # Order colors: background first, then other colors by frequency.
        freq = {int(v): int(c) for v, c in zip(unique_colors, color_counts_arr)}
        others = [int(v) for v in unique_colors if int(v) != bg_val]
        others.sort(key=lambda v: (-freq[v], v))
        colors = [bg_val] + others
        C = len(colors)
        color_to_idx = {c: i for i, c in enumerate(colors)}

        rows, cols = np.nonzero(observed)
        n = len(rows)

        val_idx = np.array([color_to_idx[int(v)] for v in obs_vals], dtype=np.int32)

        # Small regions: keep every observed color.
        # Larger regions: require reasonably frequent colors to be preserved.
        if n <= 10:
            min_significant = 1
        else:
            min_significant = max(3, int(0.05 * n))

        obs_color_counts = np.bincount(val_idx, minlength=C)
        significant = obs_color_counts >= min_significant

        candidates = []

        # Try all periods. A non-full period should repeat at least twice in
        # that dimension; a full dimension is allowed if the other dimension
        # repeats.
        for ph in range(1, H + 1):
            for pw in range(1, W + 1):
                if ph == H and pw == W:
                    continue

                ok_period = True
                if ph != H and (H // ph) < 2:
                    ok_period = False
                if pw != W and (W // pw) < 2:
                    ok_period = False

                counts = np.zeros((ph * pw, C), dtype=np.int32)
                idx = (rows % ph) * pw + (cols % pw)
                np.add.at(counts, (idx, val_idx), 1)

                best = np.argmax(counts, axis=1)
                total = counts.sum(axis=1)
                free = int(np.sum(total > 0))

                if free == 0:
                    continue

                pred = best[idx]
                mismatch = int(np.sum(pred != val_idx))

                # Reject tiles that erase a significant observed color.
                present = np.zeros(C, dtype=bool)
                present[best] = True
                if not np.all(present[significant]):
                    continue

                candidates.append((mismatch, free, ok_period, ph * pw, ph, pw))

        if not candidates:
            return arr

        def ok_cost(c):
            mismatch, free, ok, area, ph, pw = c
            return (mismatch + 0.5 * free, mismatch, free, area, ph, pw)

        def all_cost(c):
            mismatch, free, ok, area, ph, pw = c
            penalty = 0.0 if ok else 10.0 + 0.1 * free
            return (mismatch + 0.5 * free + penalty, mismatch, free, area, ph, pw)

        ok_candidates = [c for c in candidates if c[2]]
        chosen = None

        # Prefer simple periods that are low-noise.
        for budget in (max(4, int(0.1 * n) + 1), max(6, int(0.2 * n) + 2)):
            within = [c for c in ok_candidates if c[0] <= budget]
            if within:
                within.sort(key=lambda c: (c[1], c[0], c[3], c[4], c[5]))
                chosen = within[0]
                break

        if chosen is None and ok_candidates:
            chosen = min(ok_candidates, key=ok_cost)
        elif chosen is None:
            chosen = min(candidates, key=all_cost)

        # Reconstruct the chosen period and apply it to observed cells only.
        ph, pw = chosen[4], chosen[5]
        counts = np.zeros((ph * pw, C), dtype=np.int32)
        idx = (rows % ph) * pw + (cols % pw)
        np.add.at(counts, (idx, val_idx), 1)
        best = np.argmax(counts, axis=1)
        tile = best.reshape(ph, pw)

        pred_tile = tile[rows % ph, cols % pw]
        arr[rows, cols] = np.array(colors, dtype=arr.dtype)[pred_tile]

        return arr

    # First try the normal case: interiors contain global background.
    frames = _find_frames(True)
    # If nothing is found, relax the global-background requirement.
    if not frames:
        frames = _find_frames(False)

    if not frames:
        return out

    # Process smaller panels first; lock processed panels so outer panels do
    # not repaint nested panels.
    frames.sort(key=lambda x: x[9])

    locked = np.zeros((h, w), dtype=bool)

    for top, bottom, left, right, r0, r1, c0, c1, color, area in frames:
        if top > bottom or left > right:
            continue

        interior_slice = (slice(top, bottom + 1), slice(left, right + 1))
        if locked[interior_slice].all():
            continue

        sub = out[interior_slice]
        fixed_mask = locked[interior_slice].astype(bool, copy=True)

        # Keep the current frame color fixed if it leaks into the interior.
        fixed_mask |= (sub == color)

        completed = _complete_region(sub, fixed_mask)
        out[interior_slice] = completed

        # Lock the whole panel bounding box.
        locked[r0:r1 + 1, c0:c1 + 1] = True

    return out