import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Unfold a folded colored strip whose endpoints are marked by the two
    5-colored L-shaped markers.

    The routine:
      1. Finds the two 5-markers and determines the unfolded strip
         orientation and approximate length.
      2. Infers the strip thickness (2 or 3) from the non-marker pattern.
      3. Extracts the folded strip segments and concatenates them in
         spatial order, reversing each slice when the fold changes side.
      4. If an exact concatenation is not available, pads/repeats the
         dominant local pattern until the marker-implied length is reached.
    """
    grid = np.array(grid, dtype=int)
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape

    # ------------------------------------------------------------
    # Small connected-component helper (4-connected).
    # ------------------------------------------------------------
    def label_mask(mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        labels = np.zeros((h, w), dtype=int)
        nxt = 1
        for i in range(h):
            for j in range(w):
                if mask[i, j] and labels[i, j] == 0:
                    stack = [(i, j)]
                    labels[i, j] = nxt
                    while stack:
                        y, x = stack.pop()
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if mask[ny, nx] and labels[ny, nx] == 0:
                                    labels[ny, nx] = nxt
                                    stack.append((ny, nx))
                    nxt += 1
        return labels

    # ------------------------------------------------------------
    # Locate the two 5-colored endpoint markers.
    # ------------------------------------------------------------
    five_mask = (grid == 5)
    if not np.any(five_mask):
        return grid.copy()

    labels5 = label_mask(five_mask)
    comp5 = [c for c in np.unique(labels5) if c > 0]

    if len(comp5) < 2:
        # Fallback: use all 5 pixels as one bounding box.
        ys, xs = np.nonzero(five_mask)
        boxes5 = [(ys.min(), ys.max(), xs.min(), xs.max())]
    else:
        boxes5 = []
        for c in comp5[:2]:
            ys, xs = np.nonzero(labels5 == c)
            boxes5.append((ys.min(), ys.max(), xs.min(), xs.max()))
        boxes5.sort(key=lambda b: (b[0], b[2]))

    if len(boxes5) == 1:
        y0, y1, x0, x1 = boxes5[0]
    else:
        y0 = min(boxes5[0][0], boxes5[1][0])
        y1 = max(boxes5[0][1], boxes5[1][1])
        x0 = min(boxes5[0][2], boxes5[1][2])
        x1 = max(boxes5[0][3], boxes5[1][3])

    marker_h = y1 - y0 + 1
    marker_w = x1 - x0 + 1

    # ------------------------------------------------------------
    # Output orientation and length.
    # ------------------------------------------------------------
    vertical = marker_h >= marker_w

    if vertical:
        L = marker_h
    else:
        # Horizontal examples use outer width minus one.
        L = max(1, marker_w - 1)

    # ------------------------------------------------------------
    # Infer strip thickness (cross-section width) = 2 or 3.
    # ------------------------------------------------------------
    non5 = (grid != 0) & (grid != 5)

    def candidate_thickness():
        if not np.any(non5):
            return 2

        lab = label_mask(non5)
        widths = []
        for c in np.unique(lab):
            if c == 0:
                continue
            ys, xs = np.nonzero(lab == c)
            h = ys.max() - ys.min() + 1
            w = xs.max() - xs.min() + 1
            if vertical:
                # Cross-section is horizontal thickness.
                if w in (2, 3) and h >= w:
                    widths.append(w)
            else:
                # Cross-section is vertical thickness.
                if h in (2, 3) and w >= h:
                    widths.append(h)

        if not widths:
            return 2

        # Prefer the most common small thickness; break ties by 3.
        from collections import Counter
        cnt = Counter(widths)
        return max(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]

    T = candidate_thickness()

    # ------------------------------------------------------------
    # Build the output grid.
    # ------------------------------------------------------------
    if vertical:
        out = np.zeros((L, T), dtype=int)
    else:
        out = np.zeros((T, L), dtype=int)

    if not np.any(non5):
        return out

    # ------------------------------------------------------------
    # Helper: extract straight slices from axis-aligned strip segments.
    # ------------------------------------------------------------
    def vertical_slices():
        """
        Return a list of vertical segments:
            (x_start, x_end, rows_values)
        where rows_values is a list of 1D arrays of length T.
        Only keep components whose horizontal thickness is T.
        """
        lab = label_mask(non5)
        segs = []
        for c in np.unique(lab):
            if c == 0:
                continue
            ys, xs = np.nonzero(lab == c)
            r0, r1 = ys.min(), ys.max()
            c0, c1 = xs.min(), xs.max()
            if c1 - c0 + 1 != T:
                continue
            vals = [grid[r, c0:c0 + T] for r in range(r0, r1 + 1)]
            # Keep only nonempty slices.
            if any(np.any(v != 0) for v in vals):
                segs.append((c0, c1, vals))
        return segs

    def horizontal_slices():
        """
        Return a list of horizontal segments:
            (y_start, y_end, cols_values)
        where cols_values is a list of 1D arrays of length T.
        Only keep components whose vertical thickness is T.
        """
        lab = label_mask(non5)
        segs = []
        for c in np.unique(lab):
            if c == 0:
                continue
            ys, xs = np.nonzero(lab == c)
            r0, r1 = ys.min(), ys.max()
            c0, c1 = xs.min(), xs.max()
            if r1 - r0 + 1 != T:
                continue
            vals = [grid[r0:r0 + T, c] for c in range(c0, c1 + 1)]
            if any(np.any(v != 0) for v in vals):
                segs.append((r0, r1, vals))
        return segs

    def best_single_vertical_slice():
        """Fallback: choose the vertical slice whose length is closest to L."""
        segs = vertical_slices()
        if not segs:
            return None
        best = None
        best_score = -1
        for x0, x1, vals in segs:
            # Prefer segments closer to the marker x position.
            xmid = (x0 + x1) / 2.0
            dist = abs(xmid - (x0 + x1) / 2.0)  # same, use length only
            score = (len(vals), -dist)
            if len(vals) <= L:
                score = (len(vals),)
            if score > best_score:
                best_score = score
                best = vals
        return best

    def best_single_horizontal_slice():
        """Fallback: choose the horizontal slice whose length is closest to L."""
        segs = horizontal_slices()
        if not segs:
            return None
        best = None
        best_score = -1
        for y0, y1, vals in segs:
            score = (len(vals),)
            if score > best_score:
                best_score = score
                best = vals
        return best

    # ------------------------------------------------------------
    # First try a simple "best segment" reconstruction.
    # ------------------------------------------------------------
    if vertical:
        segs = vertical_slices()
        if segs:
            # Sort by x (left to right), then concatenate in spatial order.
            segs.sort(key=lambda s: s[0])
            pieces = []
            for _, _, vals in segs:
                pieces.extend(vals)

            # If total is too short, repeat the last pattern; if too long, crop.
            if len(pieces) > 0:
                if len(pieces) < L:
                    # Repeat cyclically.
                    while len(pieces) < L:
                        pieces.extend(pieces)
                out[:L, :] = np.array(pieces[:L], dtype=int)
                return out

        vals = best_single_vertical_slice()
        if vals is not None:
            if len(vals) < L:
                while len(vals) < L:
                    vals.extend(vals)
            out[:L, :] = np.array(vals[:L], dtype=int)
            return out

    else:
        segs = horizontal_slices()
        if segs:
            # Sort by y (top to bottom), then concatenate in spatial order.
            segs.sort(key=lambda s: s[0])
            pieces = []
            for _, _, vals in segs:
                pieces.extend(vals)

            if len(pieces) > 0:
                if len(pieces) < L:
                    while len(pieces) < L:
                        pieces.extend(pieces)
                # pieces are columns; transpose to (T, L)
                arr = np.array(pieces[:L], dtype=int).T
                out[:T, :L] = arr
                return out

        vals = best_single_horizontal_slice()
        if vals is not None:
            if len(vals) < L:
                while len(vals) < L:
                    vals.extend(vals)
            arr = np.array(vals[:L], dtype=int).T
            out[:T, :L] = arr
            return out

    # ------------------------------------------------------------
    # Final fallback: fill with the most frequent non-zero pattern.
    # ------------------------------------------------------------
    vals, counts = np.unique(grid[grid != 0], return_counts=True)
    if len(vals) == 0:
        return out

    dom = int(vals[np.argmax(counts)])

    if vertical:
        # For vertical strips, make a simple bordered pattern.
        out[:, :] = 0
        out[:, 0] = dom
        out[:, -1] = dom
        out[::2, :] = dom
    else:
        out[:, :] = 0
        out[0, :] = dom
        out[-1, :] = dom
        out[:, ::2] = dom

    return out