import numpy as np
from collections import deque


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Convert an ARC grid containing several colored line-grids plus a scattered
    filler/noise color into the compact bar-chart output.

    Rule inferred from the examples:
    - Each structured color draws one or more axis-aligned wall segments.
      These walls form rectangular rooms/cells.
    - A separate filler color is scattered over the input and usually does not
      form long walls.
    - For every structured color, count the number of enclosed background
      rooms created by its walls.
    - Walls may be partially overwritten by the noise color or by walls of
      another color.  The solver reconstructs missing wall pixels using
      nearby collinear evidence and non-zero support.
    - Output rows are sorted by room count (capped at 4), then by color value.
      Each output row has width 4: the first min(count, 4) cells are the
      color, the rest are the filler color.
    """

    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape
    colors = [int(c) for c in np.unique(grid) if int(c) != 0]
    if not colors:
        return np.zeros_like(grid)

    nonzero = grid != 0

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def cluster_positions(pos, max_gap):
        """Cluster sorted integer positions into (start, end) spans."""
        if len(pos) == 0:
            return []
        out = []
        start = prev = int(pos[0])
        for p in pos[1:]:
            p = int(p)
            if p - prev <= max_gap:
                prev = p
            else:
                out.append((start, prev))
                start = prev = p
        out.append((start, prev))
        return out

    def max_consec_positions(pos):
        """Longest run of consecutive integers in sorted pos."""
        if len(pos) == 0:
            return 0
        best = cur = 1
        for i in range(1, len(pos)):
            if pos[i] == pos[i - 1] + 1:
                cur += 1
            else:
                cur = 1
            if cur > best:
                best = cur
        return best

    def max_zero_gap(pos, start, end):
        """Maximum number of consecutive missing positions in [start, end]."""
        if len(pos) == 0:
            return int(end - start + 1)
        pos = [int(p) for p in pos if start <= p <= end]
        if len(pos) == 0:
            return int(end - start + 1)
        prev = start
        best = 0
        for p in pos:
            gap = p - prev - 1
            if gap > best:
                best = gap
            prev = p
        gap = end - prev
        if gap > best:
            best = gap
        return best

    # ------------------------------------------------------------------
    # Wall-segment reconstruction for one color.
    #
    # A wall segment is accepted when it is either:
    #   1. solid enough in the color itself, or
    #   2. mostly supported by non-zero pixels, with at least two pixels of
    #      this color inside the span.
    # ------------------------------------------------------------------
    def find_horizontal_segments(mask):
        segs = []
        min_len = 4

        for y in range(h):
            own = np.where(mask[y])[0]
            nz = np.where(nonzero[y])[0]

            # Solid own-color runs.
            if own.size:
                for a, b in cluster_positions(own, max_gap=2):
                    L = b - a + 1
                    if L < min_len:
                        continue
                    sub = own[(own >= a) & (own <= b)]
                    cnt = len(sub)
                    mr = max_consec_positions(sub)
                    if cnt >= 3 and (cnt / L >= 0.45 or mr >= 4):
                        segs.append((y, int(a), int(b)))

            # Occluded runs: mostly non-zero, but with this color present.
            if nz.size:
                for a, b in cluster_positions(nz, max_gap=2):
                    L = b - a + 1
                    if L < min_len:
                        continue
                    sub_own = own[(own >= a) & (own <= b)]
                    if len(sub_own) < 2:
                        continue

                    x1, x2 = int(sub_own[0]), int(sub_own[-1])
                    span = x2 - x1 + 1
                    if span < min_len:
                        continue

                    sub_nz = nz[(nz >= x1) & (nz <= x2)]
                    nz_cnt = len(sub_nz)
                    if nz_cnt / span < 0.55:
                        continue
                    if max_zero_gap(sub_nz, x1, x2) > 2:
                        continue

                    segs.append((y, x1, x2))

        return segs

    def find_vertical_segments(mask):
        segs = []
        min_len = 4

        for x in range(w):
            own = np.where(mask[:, x])[0]
            nz = np.where(nonzero[:, x])[0]

            # Solid own-color runs.
            if own.size:
                for a, b in cluster_positions(own, max_gap=2):
                    L = b - a + 1
                    if L < min_len:
                        continue
                    sub = own[(own >= a) & (own <= b)]
                    cnt = len(sub)
                    mr = max_consec_positions(sub)
                    if cnt >= 3 and (cnt / L >= 0.45 or mr >= 4):
                        segs.append((x, int(a), int(b)))

            # Occluded runs.
            if nz.size:
                for a, b in cluster_positions(nz, max_gap=2):
                    L = b - a + 1
                    if L < min_len:
                        continue
                    sub_own = own[(own >= a) & (own <= b)]
                    if len(sub_own) < 2:
                        continue

                    y1, y2 = int(sub_own[0]), int(sub_own[-1])
                    span = y2 - y1 + 1
                    if span < min_len:
                        continue

                    sub_nz = nz[(nz >= y1) & (nz <= y2)]
                    nz_cnt = len(sub_nz)
                    if nz_cnt / span < 0.55:
                        continue
                    if max_zero_gap(sub_nz, y1, y2) > 2:
                        continue

                    segs.append((x, y1, y2))

        return segs

    def merge_horizontal(segs):
        by_y = {}
        for y, x1, x2 in segs:
            by_y.setdefault(int(y), []).append((int(x1), int(x2)))
        out = []
        for y, arr in by_y.items():
            arr.sort()
            cur_s, cur_e = arr[0]
            for s, e in arr[1:]:
                if s <= cur_e + 3:
                    cur_e = max(cur_e, e)
                else:
                    out.append((y, cur_s, cur_e))
                    cur_s, cur_e = s, e
            out.append((y, cur_s, cur_e))
        return out

    def merge_vertical(segs):
        by_x = {}
        for x, y1, y2 in segs:
            by_x.setdefault(int(x), []).append((int(y1), int(y2)))
        out = []
        for x, arr in by_x.items():
            arr.sort()
            cur_s, cur_e = arr[0]
            for s, e in arr[1:]:
                if s <= cur_e + 3:
                    cur_e = max(cur_e, e)
                else:
                    out.append((x, cur_s, cur_e))
                    cur_s, cur_e = s, e
            out.append((x, cur_s, cur_e))
        return out

    def supported_extension(nonzero_row_or_col, a, b):
        """Return True if extending from a to b has some non-zero support."""
        if a == b:
            return True
        lo, hi = min(a, b), max(a, b)
        seg = nonzero_row_or_col[lo:hi + 1]
        return bool(np.any(seg))

    def snap_and_extend(h_segs, v_segs, snap=3):
        """Extend wall ends to nearby perpendicular walls when supported."""
        h = [(int(y), int(x1), int(x2)) for y, x1, x2 in h_segs]
        v = [(int(x), int(y1), int(y2)) for x, y1, y2 in v_segs]

        for _ in range(2):
            changed = False

            # Snap horizontal ends to vertical lines.
            nh = []
            for y, x1, x2 in h:
                nx1, nx2 = x1, x2
                row_nz = nonzero[y]

                for vx, vy1, vy2 in v:
                    # vertical must be close to this row or already close enough
                    if vy1 <= y <= vy2:
                        inter = True
                    else:
                        inter = abs(y - min(max(y, vy1), vy2)) <= snap

                    if not inter:
                        continue

                    if vx < x1 and x1 - vx <= snap and supported_extension(row_nz, vx, x1):
                        nx1 = min(nx1, vx)
                    elif vx > x2 and vx - x2 <= snap and supported_extension(row_nz, x2, vx):
                        nx2 = max(nx2, vx)

                if (nx1, nx2) != (x1, x2):
                    changed = True
                nh.append((y, nx1, nx2))
            h = merge_horizontal(nh)

            # Snap vertical ends to horizontal lines.
            nv = []
            for x, y1, y2 in v:
                ny1, ny2 = y1, y2
                col_nz = nonzero[:, x]

                for hy, hx1, hx2 in h:
                    if hx1 <= x <= hx2:
                        inter = True
                    else:
                        inter = abs(x - min(max(x, hx1), hx2)) <= snap

                    if not inter:
                        continue

                    if hy < y1 and y1 - hy <= snap and supported_extension(col_nz, hy, y1):
                        ny1 = min(ny1, hy)
                    elif hy > y2 and hy - y2 <= snap and supported_extension(col_nz, y2, hy):
                        ny2 = max(ny2, hy)

                if (ny1, ny2) != (y1, y2):
                    changed = True
                nv.append((x, ny1, ny2))
            v = merge_vertical(nv)

            if not changed:
                break

        return merge_horizontal(h), merge_vertical(v)

    def count_rooms(h_segs, v_segs):
        """Count 4-connected background components not connected to border."""
        if not h_segs and not v_segs:
            return 0

        wall = np.zeros((h, w), dtype=bool)
        for y, x1, x2 in h_segs:
            wall[y, max(0, int(x1)):min(w, int(x2) + 1)] = True
        for x, y1, y2 in v_segs:
            wall[max(0, int(y1)):min(h, int(y2) + 1), int(x)] = True

        outside = np.zeros((h, w), dtype=bool)
        q = deque()

        for x in range(w):
            if not wall[0, x]:
                outside[0, x] = True
                q.append((0, x))
            if not wall[h - 1, x]:
                outside[h - 1, x] = True
                q.append((h - 1, x))

        for y in range(h):
            if not wall[y, 0]:
                outside[y, 0] = True
                q.append((y, 0))
            if not wall[y, w - 1]:
                outside[y, w - 1] = True
                q.append((y, w - 1))

        while q:
            y, x = q.popleft()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if not wall[ny, nx] and not outside[ny, nx]:
                        outside[ny, nx] = True
                        q.append((ny, nx))

        seen = outside.copy()
        rooms = 0
        for y in range(h):
            for x in range(w):
                if not wall[y, x] and not seen[y, x]:
                    rooms += 1
                    q = deque([(y, x)])
                    seen[y, x] = True
                    while q:
                        cy, cx = q.popleft()
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if not wall[ny, nx] and not seen[ny, nx]:
                                    seen[ny, nx] = True
                                    q.append((ny, nx))
        return rooms

    # ------------------------------------------------------------------
    # Score all colors.
    # ------------------------------------------------------------------
    stats = {}
    for c in colors:
        mask = (grid == c)
        h_segs = merge_horizontal(find_horizontal_segments(mask))
        v_segs = merge_vertical(find_vertical_segments(mask))
        h_segs, v_segs = snap_and_extend(h_segs, v_segs)

        rooms = count_rooms(h_segs, v_segs)
        line_len = sum(int(x2 - x1 + 1) for _, x1, x2 in h_segs) + \
                   sum(int(y2 - y1 + 1) for _, y1, y2 in v_segs)
        px = int(np.count_nonzero(mask))

        stats[c] = {
            "rooms": rooms,
            "line_len": line_len,
            "px": px,
            "h_segs": h_segs,
            "v_segs": v_segs,
        }

    # Choose filler/noise color: lowest wall evidence, then rooms, then pixels.
    filler = min(
        colors,
        key=lambda c: (
            stats[c]["line_len"],
            stats[c]["rooms"],
            -stats[c]["px"],
            c
        )
    )

    # Structured colors are those with positive enclosed rooms and some wall
    # evidence.  If none are found, fall back to the best non-filler colors.
    structured = [
        c for c in colors
        if c != filler and stats[c]["rooms"] > 0 and stats[c]["line_len"] > 0
    ]

    if not structured:
        structured = [
            c for c in colors
            if c != filler and stats[c]["line_len"] > 0
        ]

    if not structured:
        # Very conservative fallback: keep all non-filler colors.
        structured = [c for c in colors if c != filler]

    items = []
    for c in structured:
        cnt = int(stats[c]["rooms"])
        if cnt <= 0:
            cnt = 1
        items.append((min(cnt, 4), c))

    # If no structured color could be identified (should not happen), return a
    # single-row chart with the filler color.
    if not items:
        return np.full((1, 4), filler, dtype=grid.dtype)

    # Sort by visible count (capped), then by color value.
    items.sort(key=lambda t: (t[0], t[1]))

    out = np.full((len(items), 4), filler, dtype=grid.dtype)
    for i, (cnt, c) in enumerate(items):
        n = max(0, min(cnt, 4))
        out[i, :n] = c

    return out