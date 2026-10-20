import numpy as np
from collections import Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a grid containing parallel line segments.

    The rule implemented:
    - Find the dominant orientation of the non-background line segments.
    - Group pixels into parallel segments.
    - Determine the common anchor side shared by most/all segments.
    - Use the middle segment (by perpendicular coordinate) as the reference length.
    - Extend shorter segments to the reference length from the anchor.
    - Keep inner segments that are already longer than the reference length.
    - Trim outermost segments that are longer than the reference length.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    if H == 0 or W == 0:
        return g

    # Background is the most frequent color.
    values, counts = np.unique(g.ravel(), return_counts=True)
    bg = int(values[np.argmax(counts)])

    mask = (g != bg)
    if not mask.any():
        return g.copy()

    pixels = np.argwhere(mask)

    # Orientation identifiers:
    # V  : vertical, direction (0, 1)
    # H  : horizontal, direction (1, 0)
    # DL : diagonal down-left, direction (1, -1), invariant p = r + c
    # DR : diagonal down-right, direction (1, 1), invariant p = r - c
    def get_p_t(r, c, orient):
        if orient == "V":
            return c, r          # p = column, t = row
        if orient == "H":
            return r, c          # p = row, t = column
        if orient == "DL":
            return r + c, r - c  # p = r+c, t = r-c
        if orient == "DR":
            return r - c, r + c  # p = r-c, t = r+c

    step_for = {"V": 1, "H": 1, "DL": 2, "DR": 2}

    def score_orientation(orient):
        """
        Score an orientation by summing squares of contiguous non-background runs.
        Also return the p-values that contain at least one run of length >= 2.
        """
        score = 0
        good_p = set()

        if orient == "V":
            for c in range(W):
                r = 0
                while r < H:
                    if mask[r, c]:
                        start = r
                        while r < H and mask[r, c]:
                            r += 1
                        L = r - start
                        if L >= 2:
                            score += L * L
                            good_p.add(c)
                    else:
                        r += 1

        elif orient == "H":
            for r in range(H):
                c = 0
                while c < W:
                    if mask[r, c]:
                        start = c
                        while c < W and mask[r, c]:
                            c += 1
                        L = c - start
                        if L >= 2:
                            score += L * L
                            good_p.add(r)
                    else:
                        c += 1

        elif orient == "DL":
            # r + c = p, moving down-left: r += 1, c -= 1
            for p in range(H + W - 1):
                r = max(0, p - (W - 1))
                c = p - r
                while 0 <= r < H and 0 <= c < W:
                    if mask[r, c]:
                        L = 0
                        while 0 <= r < H and 0 <= c < W and mask[r, c]:
                            L += 1
                            r += 1
                            c -= 1
                        if L >= 2:
                            score += L * L
                            good_p.add(p)
                    else:
                        r += 1
                        c -= 1

        elif orient == "DR":
            # r - c = p, moving down-right: r += 1, c += 1
            for p in range(-(W - 1), H):
                r = max(0, p)
                c = r - p
                while 0 <= r < H and 0 <= c < W:
                    if mask[r, c]:
                        L = 0
                        while 0 <= r < H and 0 <= c < W and mask[r, c]:
                            L += 1
                            r += 1
                            c += 1
                        if L >= 2:
                            score += L * L
                            good_p.add(p)
                    else:
                        r += 1
                        c += 1

        return score, good_p

    best_score = -1
    best_orient = None
    best_good_p = set()

    for orient in ("V", "H", "DL", "DR"):
        sc, gp = score_orientation(orient)
        if sc > best_score:
            best_score = sc
            best_orient = orient
            best_good_p = gp

    # If no clear parallel line structure exists, return input unchanged.
    if best_score <= 0 or best_orient is None:
        return g.copy()

    step = step_for[best_orient]

    # Group pixels into segments by (perpendicular coordinate, color).
    # This allows filling missing interior pixels on the same line.
    groups = {}
    for r, c in pixels:
        r = int(r)
        c = int(c)
        p, t = get_p_t(r, c, best_orient)
        if p not in best_good_p:
            continue
        color = int(g[r, c])
        key = (p, color)
        if key not in groups:
            groups[key] = [t, t]
        else:
            if t < groups[key][0]:
                groups[key][0] = t
            if t > groups[key][1]:
                groups[key][1] = t

    segments = []
    for (p, color), (tmin, tmax) in groups.items():
        if tmin > tmax:
            tmin, tmax = tmax, tmin
        length = (tmax - tmin) // step + 1
        if length < 1:
            length = 1
        segments.append({
            "p": int(p),
            "color": color,
            "tmin": int(tmin),
            "tmax": int(tmax),
            "len": int(length),
        })

    if not segments:
        return g.copy()

    # Determine which endpoint side is the common anchor.
    tmins = [s["tmin"] for s in segments]
    tmaxs = [s["tmax"] for s in segments]

    cmin = Counter(tmins)
    cmax = Counter(tmaxs)

    max_freq_min = max(cmin.values())
    max_freq_max = max(cmax.values())

    if max_freq_max > max_freq_min:
        anchor = "max"
    elif max_freq_min > max_freq_max:
        anchor = "min"
    else:
        range_min = max(tmins) - min(tmins)
        range_max = max(tmaxs) - min(tmaxs)
        anchor = "min" if range_min <= range_max else "max"

    def choose_mode(counter, values):
        """Choose the most frequent value; break ties by closeness to the mean."""
        max_freq = max(counter.values())
        candidates = [v for v, f in counter.items() if f == max_freq]
        avg = sum(values) / len(values)
        return min(candidates, key=lambda v: abs(v - avg))

    if anchor == "min":
        A = choose_mode(cmin, tmins)
        direction = 1
    else:
        A = choose_mode(cmax, tmaxs)
        direction = -1

    # Choose the middle line by perpendicular coordinate as the reference.
    unique_ps = sorted(set(s["p"] for s in segments))
    min_p = unique_ps[0]
    max_p = unique_ps[-1]
    target_p = (min_p + max_p) / 2.0

    p_to_segments = {}
    for s in segments:
        p_to_segments.setdefault(s["p"], []).append(s)

    ref_p = min(unique_ps, key=lambda p: (abs(p - target_p), p))
    ref_segments = p_to_segments[ref_p]
    ref_segment = max(ref_segments, key=lambda s: s["len"])
    target_len = int(ref_segment["len"])
    if target_len < 1:
        target_len = 1

    out = np.full(g.shape, bg, dtype=g.dtype)

    def fill_segment(p, color, t0, t1):
        """Fill one parallel line segment between t0 and t1 inclusive."""
        if t0 > t1:
            t0, t1 = t1, t0

        t = t0
        while t <= t1:
            if best_orient == "V":
                r = t
                c = p
            elif best_orient == "H":
                r = p
                c = t
            elif best_orient == "DL":
                if (p + t) % 2 != 0:
                    t += step
                    continue
                r = (p + t) // 2
                c = (p - t) // 2
            else:  # DR
                if (t + p) % 2 != 0:
                    t += step
                    continue
                r = (t + p) // 2
                c = (t - p) // 2

            if 0 <= r < H and 0 <= c < W:
                out[r, c] = color

            t += step

    for s in segments:
        p = s["p"]
        color = s["color"]
        tmin = s["tmin"]
        tmax = s["tmax"]
        length = s["len"]

        # Inner overlong segments are preserved.
        protected = (
            length > target_len
            and len(unique_ps) > 1
            and min_p < p < max_p
        )

        if protected:
            fill_segment(p, color, tmin, tmax)
        else:
            # Align to the common anchor and force the reference length.
            target_other = A + direction * (target_len - 1) * step
            fill_segment(p, color, A, target_other)

    return out