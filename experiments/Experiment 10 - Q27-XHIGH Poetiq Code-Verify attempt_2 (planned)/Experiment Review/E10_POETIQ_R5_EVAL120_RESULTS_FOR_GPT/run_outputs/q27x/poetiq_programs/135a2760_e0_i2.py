import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Restore corrupted pixels inside framed rectangular cells.

    Each cell is a rectangle bounded by a non-frame color. Its interior is a
    two-color periodic pattern using the global frame color as background and
    one local foreground color. Corrupted pixels are repaired by inferring the
    simplest consistent repeating tile for the interior.
    """
    if grid is None:
        return grid

    out = np.array(grid, dtype=int, copy=True)
    if out.ndim != 2 or out.size == 0:
        return out

    h, w = out.shape
    if h < 3 or w < 3:
        return out

    def _components(mask):
        """Return list of 4-connected components as lists of (r, c)."""
        ch, cw = mask.shape
        visited = np.zeros((ch, cw), dtype=bool)
        comps = []
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

        for r in range(ch):
            for c in range(cw):
                if mask[r, c] and not visited[r, c]:
                    stack = [(r, c)]
                    visited[r, c] = True
                    comp = []

                    while stack:
                        x, y = stack.pop()
                        comp.append((x, y))

                        for dr, dc in dirs:
                            nx, ny = x + dr, y + dc
                            if 0 <= nx < ch and 0 <= ny < cw:
                                if mask[nx, ny] and not visited[nx, ny]:
                                    visited[nx, ny] = True
                                    stack.append((nx, ny))

                    comps.append(comp)

        return comps

    def _frame_color():
        """Dominant color on the outer boundary."""
        border = np.concatenate((out[0, :], out[-1, :], out[:, 0], out[:, -1]))
        vals, counts = np.unique(border, return_counts=True)
        return int(vals[np.argmax(counts)])

    frame_color = _frame_color()

    def _overlap(a, b):
        # a, b: (r_start, r_end, c_start, c_end, color)
        return not (a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2])

    def _interiors_from_components(comps, color, min_perim_ratio):
        """Find rectangular ring components and return their interiors."""
        res = []

        for comp in comps:
            if not comp:
                continue

            r0 = min(p[0] for p in comp)
            r1 = max(p[0] for p in comp)
            c0 = min(p[1] for p in comp)
            c1 = max(p[1] for p in comp)

            bh = r1 - r0 + 1
            bw = c1 - c0 + 1

            if bh < 3 or bw < 3:
                continue

            perim_size = 2 * bh + 2 * bw - 4
            if perim_size <= 0:
                continue

            perim_count = sum(
                1 for r, c in comp
                if r == r0 or r == r1 or c == c0 or c == c1
            )

            if perim_count < min_perim_ratio * perim_size:
                continue

            # A thin rectangular frame should not contain many extra pixels.
            interior_area = (bh - 2) * (bw - 2)
            if len(comp) > perim_size + 0.2 * interior_area:
                continue

            r_s, r_e = r0 + 1, r1
            c_s, c_e = c0 + 1, c1

            if r_s < r_e and c_s < c_e:
                interior = out[r_s:r_e, c_s:c_e]
                if interior.size == 0:
                    continue

                # Skip solid blocks where the interior is entirely the frame color.
                if not np.any(interior != color):
                    continue

                res.append((r_s, r_e, c_s, c_e, color))

        return res

    # Detect framed cell interiors.
    nonframe_colors = [int(v) for v in np.unique(out) if int(v) != frame_color]
    interiors = []

    # First pass: strict rectangular ring detection for every non-frame color.
    for color in nonframe_colors:
        mask = out == color
        if np.any(mask):
            comps = _components(mask)
            interiors.extend(_interiors_from_components(comps, color, 0.85))

    # Second pass: for the most frequent non-frame color, allow slightly broken frames.
    if nonframe_colors:
        vals, counts = np.unique(out[out != frame_color], return_counts=True)
        if len(vals):
            fallback_color = int(vals[np.argmax(counts)])
            mask = out == fallback_color
            if np.any(mask):
                comps = _components(mask)
                low = _interiors_from_components(comps, fallback_color, 0.55)
                for rect in low:
                    if not any(_overlap(rect, existing) for existing in interiors):
                        interiors.append(rect)

    if not interiors:
        return out

    # Process smaller interiors first.
    interiors.sort(key=lambda x: (x[1] - x[0]) * (x[3] - x[2]))

    def _repair(sub, frame_color, border_color):
        """Repair one cell interior using periodic tiling."""
        sh, sw = sub.shape
        if sh == 0 or sw == 0:
            return sub.copy()

        vals, counts = np.unique(sub, return_counts=True)
        count_by_val = {int(v): int(c) for v, c in zip(vals, counts)}

        # Choose the local object color. Prefer colors that are neither the
        # global frame nor the cell border.
        fg_candidates = []
        for v in vals:
            iv = int(v)
            if iv == frame_color:
                continue
            if iv == border_color:
                continue
            fg_candidates.append((iv, count_by_val[iv]))

        if not fg_candidates and border_color is not None:
            if border_color in count_by_val:
                fg_candidates.append((border_color, count_by_val[border_color]))

        if not fg_candidates:
            return sub.copy()

        foreground = max(fg_candidates, key=lambda x: x[1])[0]

        if sh == 1 and sw == 1:
            return sub.copy()

        # Binary representation: foreground vs background.
        B = (sub == foreground).astype(np.int8)
        total = sh * sw
        ones_total = int(B.sum())
        uniform_score = max(ones_total, total - ones_total)

        # Score all non-full periods.
        candidates = []
        for rh in range(1, sh + 1):
            for rw in range(1, sw + 1):
                if rh == sh and rw == sw:
                    continue

                score = 0
                for i in range(rh):
                    rows = B[i::rh]
                    if rows.size == 0:
                        continue
                    for j in range(rw):
                        arr = rows[:, j::rw]
                        n = arr.size
                        if n == 0:
                            continue
                        ones_c = int(arr.sum())
                        score += max(ones_c, n - ones_c)

                candidates.append((score, rh * rw, rh, rw))

        if not candidates:
            return sub.copy()

        max_score = max(c[0] for c in candidates)

        # Allow some corruption, but prefer simple periods.
        tolerance = max(2, int(total * 0.10))
        valid = [c for c in candidates if c[0] >= max_score - tolerance]

        # Require the periodic model to improve over a uniform fill.
        valid = [c for c in valid if c[0] > uniform_score]

        if not valid:
            if max_score > uniform_score:
                valid = [c for c in candidates if c[0] == max_score]
            else:
                return sub.copy()

        # Choose the simplest sufficient period:
        # smallest tile area, then best score.
        chosen = min(valid, key=lambda x: (x[1], -x[0], x[2], x[3]))
        score, area, rh, rw = chosen

        if score <= uniform_score:
            return sub.copy()

        # Build the inferred tile by majority vote in each residue class.
        tile = np.zeros((rh, rw), dtype=int)
        for i in range(rh):
            for j in range(rw):
                arr = B[i::rh, j::rw]
                n = arr.size
                ones_c = int(arr.sum())
                tile[i, j] = 1 if 2 * ones_c > n else 0

        row_mod = np.tile(np.arange(rh), (sh + rh - 1) // rh)[:sh]
        col_mod = np.tile(np.arange(rw), (sw + rw - 1) // rw)[:sw]
        filled = tile[row_mod[:, None], col_mod[None, :]]

        return np.where(filled == 1, foreground, frame_color)

    processed = np.zeros((h, w), dtype=bool)

    for r_s, r_e, c_s, c_e, color in interiors:
        if processed[r_s:r_e, c_s:c_e].any():
            continue

        sub = out[r_s:r_e, c_s:c_e].copy()
        fixed = _repair(sub, frame_color, color)

        if fixed is not None and fixed.shape == sub.shape:
            out[r_s:r_e, c_s:c_e] = fixed
            processed[r_s:r_e, c_s:c_e] = True

    return out