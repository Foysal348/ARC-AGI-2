def transform(grid: "np.ndarray") -> "np.ndarray":
    """
    Normalize a set of parallel line segments.

    The rule:
      1. Detect straight 1-pixel-wide segments in horizontal, vertical,
         diagonal (r+c constant), and diagonal (r-c constant) directions.
      2. Group parallel segments that share an endpoint line,
         i.e. they have the same minimum or maximum coordinate along the
         line direction.
      3. For each group, pick the middle segment by position along the
         shared endpoint.
      4. Use that segment's length as the target length, and extend or
         trim all other segments in the group while keeping the shared
         endpoint fixed.
    """
    import numpy as np
    from collections import defaultdict

    if grid is None:
        return np.array([], dtype=int)

    grid = np.array(grid, dtype=int, copy=False)
    out = grid.copy()

    H, W = out.shape
    if H == 0 or W == 0:
        return out

    # Background is the most frequent color. Prefer a corner color if tied.
    vals, counts = np.unique(out.ravel(), return_counts=True)
    max_count = counts.max()
    candidates = vals[counts == max_count]
    corner = int(out[0, 0])
    if corner in candidates.tolist():
        bg = corner
    else:
        bg = int(candidates[0])

    segments = []

    def add_segment(orient, color, q, a0, a1, step):
        """Add a straight segment if it has at least two pixels."""
        if a0 > a1:
            a0, a1 = a1, a0
        if (a1 - a0) % step != 0:
            return
        length = (a1 - a0) // step + 1
        if length < 2:
            return
        segments.append({
            "id": len(segments),
            "orient": orient,
            "color": int(color),
            "q": int(q),
            "a_min": int(a0),
            "a_max": int(a1),
            "step": int(step),
        })

    # Horizontal runs
    for r in range(H):
        c = 0
        while c < W:
            if grid[r, c] == bg:
                c += 1
            else:
                color = grid[r, c]
                start = c
                while c + 1 < W and grid[r, c + 1] == color:
                    c += 1
                add_segment("H", color, r, start, c, 1)
                c += 1

    # Vertical runs
    for c in range(W):
        r = 0
        while r < H:
            if grid[r, c] == bg:
                r += 1
            else:
                color = grid[r, c]
                start = r
                while r + 1 < H and grid[r + 1, c] == color:
                    r += 1
                add_segment("V", color, c, start, r, 1)
                r += 1

    # Diagonal runs with constant r + c
    # Direction parameter is a = r - c, increasing by 2 each step.
    for s in range(H + W - 1):
        r_start = max(0, s - (W - 1))
        r_end = min(H - 1, s)
        if r_start > r_end:
            continue

        r = r_start
        while r <= r_end:
            c = s - r
            if grid[r, c] == bg:
                r += 1
            else:
                color = grid[r, c]
                a0 = r - c
                last_r = r

                while r + 1 <= r_end and grid[r + 1, s - (r + 1)] == color:
                    r += 1
                    last_r = r

                a1 = last_r - (s - last_r)
                add_segment("D1", color, s, a0, a1, 2)
                r += 1

    # Diagonal runs with constant r - c
    # Direction parameter is a = r, increasing by 1 each step.
    for d in range(-(W - 1), H):
        r_start = max(0, d)
        r_end = min(H - 1, W - 1 + d)
        if r_start > r_end:
            continue

        r = r_start
        while r <= r_end:
            c = r - d
            if grid[r, c] == bg:
                r += 1
            else:
                color = grid[r, c]
                a0 = r
                last_r = r

                while r + 1 <= r_end and grid[r + 1, r + 1 - d] == color:
                    r += 1
                    last_r = r

                add_segment("D2", color, d, a0, last_r, 1)
                r += 1

    if not segments:
        return out

    by_orient = defaultdict(list)
    for seg in segments:
        by_orient[seg["orient"]].append(seg)

    groups = []

    for orient, segs in by_orient.items():
        side_vals = {
            "min": defaultdict(list),
            "max": defaultdict(list),
        }

        for seg in segs:
            side_vals["min"][seg["a_min"]].append(seg["id"])
            side_vals["max"][seg["a_max"]].append(seg["id"])

        for side, vals in side_vals.items():
            for value, ids in vals.items():
                if len(ids) < 2:
                    continue

                group = [segments[i] for i in ids]

                # Choose the middle segment in the group by position along
                # the shared endpoint.
                pos_min = min(s["q"] for s in group)
                pos_max = max(s["q"] for s in group)
                mid2 = pos_min + pos_max  # twice the midpoint

                spans = sorted(s["a_max"] - s["a_min"] for s in group)
                med_span = spans[len(spans) // 2]

                def key(s):
                    pos_dist = abs(2 * s["q"] - mid2)
                    span = s["a_max"] - s["a_min"]
                    span_dist = abs(span - med_span)
                    color_tie = 0 if s["color"] == 9 else 1
                    return (pos_dist, span_dist, color_tie, -span, s["id"])

                ref = min(group, key=key)

                if side == "min":
                    anchor_a = value
                    other_a = ref["a_max"]
                else:
                    anchor_a = value
                    other_a = ref["a_min"]

                tmin = min(anchor_a, other_a)
                tmax = max(anchor_a, other_a)

                groups.append({
                    "orient": orient,
                    "side": side,
                    "size": len(ids),
                    "ref_id": ref["id"],
                    "tmin": tmin,
                    "tmax": tmax,
                    "group": group,
                })

    # Apply larger groups first. If a segment appears in multiple groups,
    # the larger group takes priority.
    groups.sort(key=lambda g: (-g["size"], g["orient"], g["side"], g["tmin"], g["tmax"]))

    def coord(orient, q, a):
        if orient == "H":
            return q, a
        if orient == "V":
            return a, q
        if orient == "D1":
            return (q + a) // 2, (q - a) // 2
        if orient == "D2":
            return a, a - q
        return None

    adjusted = set()

    for group in groups:
        tmin = group["tmin"]
        tmax = group["tmax"]

        for seg in group["group"]:
            if seg["id"] in adjusted:
                continue
            adjusted.add(seg["id"])

            orient = seg["orient"]
            q = seg["q"]
            color = seg["color"]
            step = seg["step"]

            # Remove pixels outside the target interval.
            a = seg["a_min"]
            while a <= seg["a_max"]:
                if a < tmin or a > tmax:
                    r, c = coord(orient, q, a)
                    if 0 <= r < H and 0 <= c < W and out[r, c] == color:
                        out[r, c] = bg
                a += step

            # Draw the full target interval.
            a = tmin
            while a <= tmax:
                r, c = coord(orient, q, a)
                if 0 <= r < H and 0 <= c < W:
                    if out[r, c] == bg or out[r, c] == color:
                        out[r, c] = color
                a += step

    return out