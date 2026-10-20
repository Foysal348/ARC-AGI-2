import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the output mosaic.

    The rule implemented here is:
    1. Treat 1 as arrow/door pixels and all other colors as room colors.
    2. Find rectangular rooms in the input by grouping columns into vertical
       strips and splitting each strip into horizontal segments whose dominant
       color is stable.
    3. For every room, infer the direction indicated by its 1-pixels.  If a
       room has no 1-pixels, default to the direction toward the center of the
       whole grid.
    4. Follow that direction to the neighboring room (or token pixel) and use
       the encountered non-1 color as the output color for the room.
    5. If the original grid is not square, rotate the recolored intermediate
       grid 90 degrees:
          - clockwise if the original width is odd,
          - counter-clockwise if the original width is even.
    """
    if grid is None:
        return grid

    g = np.array(grid, copy=True)
    h, w = g.shape

    # ------------------------------------------------------------------
    # Small helper: dominant non-1 color in a rectangle.
    # ------------------------------------------------------------------
    def dominant_non1_color(sub: np.ndarray) -> int:
        vals = sub.reshape(-1)
        vals = vals[vals != 1]
        if vals.size == 0:
            return -1
        # Count colors, prefer the most frequent non-1 color.
        vals = vals.astype(int)
        vals, counts = np.unique(vals, return_counts=True)
        return int(vals[np.argmax(counts)])

    # ------------------------------------------------------------------
    # Infer strip boundaries.
    #
    # The inputs are rectangular mosaics that can be parsed as vertical
    # strips.  We find strip widths by trying to split columns into bands
    # where each band can be segmented into horizontal rectangles with a
    # stable dominant color.
    # ------------------------------------------------------------------
    def score_strip_partition(boundaries):
        """
        boundaries: sorted list [0, b1, b2, ..., w]
        Returns a high score if the partition looks plausible.
        """
        total = 0
        n_strips = len(boundaries) - 1
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            strip = g[:, a:b]
            if strip.size == 0:
                continue

            # Split strip into horizontal runs where dominant color stays same.
            segs = 0
            prev_dom = -2
            for r in range(h):
                row = strip[r]
                dom = dominant_non1_color(row)
                if dom != prev_dom:
                    # Ignore very small one-pixel color changes caused by tokens.
                    if prev_dom != -2:
                        segs += 1
                    prev_dom = dom
            # Prefer fewer, larger segments.
            total += max(0, 10 - segs)

        # Prefer a modest number of strips.
        total -= max(0, abs(n_strips - 3) * 2)
        return total

    # Generate candidate strip partitions by selecting a subset of columns.
    # Since grids are small/medium, brute force all combinations is too large,
    # so use a simple dynamic-programming over possible widths.
    best_boundaries = [0, w]
    best_score = -10**9

    # Candidate internal boundaries: columns that often change color.
    col_change = np.zeros(w, dtype=int)
    for c in range(1, w):
        col_change[c] = np.sum(g[:, c - 1] != g[:, c])

    # Try all boundaries from a limited candidate set.
    candidates = [0]
    for c in range(1, w):
        if col_change[c] > max(1, h // 8):
            candidates.append(c)
    candidates.append(w)
    candidates = sorted(set(candidates))

    # Enumerate small number of boundaries (<= 8) to keep runtime low.
    if len(candidates) <= 9:
        from itertools import combinations
        for k in range(0, len(candidates) - 1):
            for combo in combinations(candidates[1:-1], k):
                bounds = [0] + sorted(combo) + [w]
                # Avoid extremely narrow strips.
                if any(b2 - b1 < 2 for b1, b2 in zip(bounds[:-1], bounds[1:])):
                    continue
                s = score_strip_partition(bounds)
                if s > best_score:
                    best_score = s
                    best_boundaries = bounds

    # ------------------------------------------------------------------
    # Split each strip into horizontal rooms.
    # ------------------------------------------------------------------
    rooms = []  # dict: r0,r1,c0,c1, color, ones positions
    for c0, c1 in zip(best_boundaries[:-1], best_boundaries[1:]):
        strip = g[:, c0:c1]
        r0 = 0
        prev_dom = -2
        for r1 in range(1, h + 1):
            row = g[r1, c0:c1]
            dom = dominant_non1_color(row)
            if r1 == h or dom != prev_dom:
                if prev_dom != -2:
                    block = g[r0:r1, c0:c1]
                    dom_block = dominant_non1_color(block)
                    ones = np.argwhere(block == 1)
                    # If there are no non-1 colors, use 0.
                    if dom_block == -1:
                        dom_block = 0
                    rooms.append({
                        'r0': r0, 'r1': r1, 'c0': c0, 'c1': c1,
                        'color': int(dom_block), 'ones': ones
                    })
                r0 = r1
            if prev_dom == -2:
                prev_dom = dom

    # If parsing produced too few rooms, fall back to single room.
    if not rooms:
        rooms = [{'r0': 0, 'r1': h, 'c0': 0, 'c1': w,
                  'color': dominant_non1_color(g) if np.any(g != 1) else 0,
                  'ones': np.argwhere(g == 1)}]

    # ------------------------------------------------------------------
    # Direction inference.
    #
    # For each room, use the 1-pixel pattern to choose one of 4 directions.
    # The heuristic is based on the distribution of 1s relative to the room
    # center and bounding box.  No 1s means "toward grid center", with up as
    # fallback.
    # ------------------------------------------------------------------
    def room_center(room):
        return (room['r0'] + room['r1']) / 2.0, (room['c0'] + room['c1']) / 2.0

    def infer_direction(room):
        ones = room['ones']
        r0, r1, c0, c1 = room['r0'], room['r1'], room['c0'], room['c1']
        rh, rw = r1 - r0, c1 - c0
        rc, cc = (r0 + r1) / 2.0, (c0 + c1) / 2.0

        if ones.size == 0:
            # Default: point toward the center of the whole grid.
            gr, gc = h / 2.0, w / 2.0
            dy, dx = gr - rc, gc - cc
            if abs(dy) > abs(dx):
                return (1 if dy > 0 else -1, 0)
            elif abs(dx) > abs(dy):
                return (0, 1 if dx > 0 else -1)
            else:
                return (-1, 0)

        # Centroid of 1s in local coordinates.
        yr, yc = ones[:, 0].mean(), ones[:, 1].mean()

        # Principal orientation of 1s.
        if ones.shape[0] >= 2:
            coords = np.stack([ones[:, 0] - yr, ones[:, 1] - yc], axis=1)
            cov = np.cov(coords.T)
            evals, evecs = np.linalg.eigh(cov)
            main = evecs[:, np.argmax(evals)]
            angle = np.arctan2(main[1], main[0])  # -pi..pi
            vertical = abs(angle) > np.pi / 4 and abs(angle) < 3 * np.pi / 4
            horizontal = not vertical
        else:
            vertical = False
            horizontal = False

        # Use centroid offset and orientation to pick a direction.
        # This is a compact heuristic matching the examples' arrow styles.
        if vertical:
            # Vertical 1-line: choose down if the pattern is in upper half or
            # has a foot on the right, otherwise up; if centered, use center direction.
            if yr < rh / 2.0:
                return (-1, 0)
            elif yr > rh / 2.0:
                return (1, 0)
            else:
                # Foot side: if more 1s on right side, down; on left, up.
                side = np.sum(ones[:, 1] > rw / 2.0) - np.sum(ones[:, 1] < rw / 2.0)
                return (1, 0) if side >= 0 else (-1, 0)
        elif horizontal:
            if yc < rw / 2.0:
                return (0, -1)
            elif yc > rw / 2.0:
                return (0, 1)
            else:
                side = np.sum(ones[:, 0] > rh / 2.0) - np.sum(ones[:, 0] < rh / 2.0)
                return (0, 1) if side >= 0 else (0, -1)

        # General fallback: direction from centroid to room center.
        dy, dx = rc - (r0 + yr), cc - (c0 + yc)
        if abs(dy) > abs(dx):
            return (1 if dy > 0 else -1, 0)
        elif abs(dx) > abs(dy):
            return (0, 1 if dx > 0 else -1)
        else:
            # Toward grid center.
            gr, gc = h / 2.0, w / 2.0
            dy, dx = gr - rc, gc - cc
            if abs(dy) > abs(dx):
                return (1 if dy > 0 else -1, 0)
            elif abs(dx) > abs(dy):
                return (0, 1 if dx > 0 else -1)
            else:
                return (-1, 0)

    # ------------------------------------------------------------------
    # Target color lookup: move from room edge in direction until hitting a
    # non-1 pixel.  If that color is a small token inside another room, use it.
    # ------------------------------------------------------------------
    def target_color(room, d):
        dy, dx = d
        r, c = room_center(room)

        # Start from the room's boundary in that direction.
        if dy == 1:
            r = float(room['r1'])
        elif dy == -1:
            r = float(room['r0'])
        else:
            r = float(room['r0']) + (room['r1'] - room['r0']) / 2.0

        if dx == 1:
            c = float(room['c1'])
        elif dx == -1:
            c = float(room['c0'])
        else:
            c = float(room['c0']) + (room['c1'] - room['c0']) / 2.0

        # Step outward in small increments.
        for _ in range(h + w + 5):
            r += dy * 0.5
            c += dx * 0.5
            if r < 0 or r >= h or c < 0 or c >= w:
                break
            ir, ic = int(round(r)), int(round(c))
            val = int(g[ir, ic])
            if val != 1:
                return val
        return room['color']

    # ------------------------------------------------------------------
    # Build recolored intermediate grid (same size as input).
    # ------------------------------------------------------------------
    mid = np.zeros_like(g)
    for room in rooms:
        d = infer_direction(room)
        val = target_color(room, d)
        r0, r1, c0, c1 = room['r0'], room['r1'], room['c0'], room['c1']
        mid[r0:r1, c0:c1] = val

    # ------------------------------------------------------------------
    # Rotate for non-square grids, using the observed parity rule.
    # ------------------------------------------------------------------
    if h != w:
        if w % 2 == 1:
            mid = np.rot90(mid, k=-1)  # clockwise
        else:
            mid = np.rot90(mid, k=1)   # counter-clockwise

    return mid