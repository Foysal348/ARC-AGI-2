import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Complete periodic patterns inside framed rectangular panels.

    Each panel is a thin rectangular border surrounding an interior region.
    The interior is assumed to be a noisy 2D periodic pattern using mainly
    two colors: a background/separator color and one or more pattern colors.
    For each panel, this function tries small periods, reconstructs the
    implied tile by majority vote, and chooses the simplest period that is
    consistent with the observed interior.
    """
    out = np.array(grid, copy=True)
    if out.size == 0:
        return out

    h, w = out.shape
    global_bg = out[0, 0]

    def _find_frames():
        """
        Find all thin rectangular frames.

        A frame is a 4-connected component of some non-background color whose
        bounding-box boundary is fully that color and whose component size
        equals the rectangle perimeter.
        """
        frames = []

        for color in np.unique(out):
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

                    if not (
                        np.all(out[min_r, min_c:max_c + 1] == color)
                        and np.all(out[max_r, min_c:max_c + 1] == color)
                        and np.all(out[min_r:max_r + 1, min_c] == color)
                        and np.all(out[min_r:max_r + 1, max_c] == color)
                    ):
                        continue

                    interior = out[min_r + 1:max_r, min_c + 1:max_c]
                    if interior.size > 0 and np.all(interior == color):
                        continue

                    frames.append((min_r, max_r, min_c, max_c))

        # Remove any duplicate bounding boxes.
        return list(dict.fromkeys(frames))

    def _reconstruct(arr, ph, pw, bg):
        """
        Reconstruct arr assuming it is periodic with period (ph, pw).

        For each tile cell, choose the majority color among all positions
        congruent to that tile cell modulo (ph, pw). Ties prefer background.
        """
        recon = np.empty(arr.shape, dtype=arr.dtype)

        for i in range(ph):
            for j in range(pw):
                vals = arr[i::ph, j::pw]
                vals_unique, counts = np.unique(vals, return_counts=True)
                best_count = counts.max()
                tied = vals_unique[counts == best_count]

                if np.any(tied == bg):
                    value = bg
                else:
                    value = int(tied[0])

                recon[i::ph, j::pw] = value

        return recon

    def _complete_pattern(arr):
        """Complete the periodic pattern inside one panel interior."""
        hgt, wid = arr.shape
        if hgt == 0 or wid == 0:
            return arr

        unique = np.unique(arr)
        if len(unique) <= 1:
            return arr

        # Use the global background if present; otherwise use the most common color.
        if global_bg in unique:
            bg = int(global_bg)
        else:
            vals, counts = np.unique(arr, return_counts=True)
            bg = int(vals[np.argmax(counts)])

        obj_colors = set(unique.tolist()) - {bg}
        if not obj_colors:
            return arr

        obj_arr = np.array(list(obj_colors), dtype=arr.dtype)
        area = hgt * wid

        # Allow reasonably small tiles. For small panels this allows all
        # non-full periods.
        max_tile_area = max(100, area // 2)
        if max_tile_area > area:
            max_tile_area = area

        candidates = []

        for ph in range(1, hgt + 1):
            if ph > max_tile_area:
                break

            for pw in range(1, wid + 1):
                tile_area = ph * pw

                if tile_area > max_tile_area:
                    continue

                # The full interior is always a perfect fit but is not a
                # useful "periodic pattern" here.
                if ph == hgt and pw == wid:
                    continue

                recon = _reconstruct(arr, ph, pw, bg)

                # Reject degenerate reconstructions that lose the background
                # or all non-background pattern colors.
                if not np.any(recon == bg):
                    continue
                if not np.any(np.isin(recon, obj_arr)):
                    continue

                mismatch = int(np.sum(recon != arr))
                candidates.append((mismatch, tile_area, ph, pw))

        if not candidates:
            return arr

        min_mismatch = min(cand[0] for cand in candidates)

        # A small adaptive noise tolerance: the best candidate defines the
        # noise level, and we accept candidates slightly above it, bounded
        # by an absolute fraction of the panel size.
        absolute_threshold = max(5, int(0.15 * area) + 1)
        margin = max(3, int(0.05 * area))
        accept_threshold = min(absolute_threshold, min_mismatch + margin)

        accepted = [cand for cand in candidates if cand[0] <= accept_threshold]

        if accepted:
            # Prefer the simplest period: smallest tile area first.
            accepted.sort(key=lambda x: (x[1], x[0], x[2], x[3]))
            best = accepted[0]
        else:
            # Fallback: best-fitting non-full period.
            candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
            best = candidates[0]

        return _reconstruct(arr, best[2], best[3], bg)

    frames = _find_frames()

    if not frames:
        return out

    # Process smaller frames first in case of nested/unusual layouts.
    frames.sort(key=lambda box: (box[1] - box[0]) * (box[3] - box[2]))

    for min_r, max_r, min_c, max_c in frames:
        interior = out[min_r + 1:max_r, min_c + 1:max_c]
        if interior.size == 0:
            continue

        out[min_r + 1:max_r, min_c + 1:max_c] = _complete_pattern(interior)

    return out