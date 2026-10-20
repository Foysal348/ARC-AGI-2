import numpy as np
from collections import deque


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Convert a grid containing several colored line-grids and one scattered
    noise/filler color into a small bar chart.

    For each meaningful color:
      - reconstruct its wall segments, allowing small gaps and occlusion
        by non-background pixels;
      - count the number of enclosed rooms/cells formed by those walls;
      - display one output row for that color, with `min(count, 4)` cells of
        the color and the remaining cells filled with the noise color.

    Rows are sorted by cell count (ascending), then by color value.
    """

    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape
    colors = [int(c) for c in np.unique(grid) if c != 0]
    if not colors:
        return np.zeros_like(grid)

    def max_run(xs):
        """Longest consecutive run in sorted integer array xs."""
        if len(xs) == 0:
            return 0
        best = cur = 1
        for i in range(1, len(xs)):
            if xs[i] == xs[i - 1] + 1:
                cur += 1
            else:
                cur = 1
            best = max(best, cur)
        return best

    def horizontal_segments(mask, all_nonzero, max_gap=4):
        """
        Detect horizontal wall segments for one color.

        A segment can be:
          - a dense run of its own color;
          - a run supported by other non-zero pixels (occlusion), with no
            long background gaps.
        """
        segs = []
        for y in range(h):
            xs = np.where(mask[y])[0]
            if xs.size == 0:
                continue

            clusters = []
            start = int(xs[0])
            end = int(xs[0])
            last = int(xs[0])
            for x in xs[1:]:
                x = int(x)
                if x - last <= max_gap:
                    end = x
                    last = x
                else:
                    clusters.append((start, end))
                    start = x
                    end = x
                    last = x
            clusters.append((start, end))

            for x1, x2 in clusters:
                L = x2 - x1 + 1
                if L < 4:
                    continue

                sub = xs[(xs >= x1) & (xs <= x2)]
                cnt = int(sub.size)
                run = max_run(sub)

                own_density = cnt / L
                nonzero_density = int(np.count_nonzero(all_nonzero[y, x1:x2 + 1])) / L
                zero_gaps = []
                prev = 0
                for x in sub:
                    zero_gaps.append(int(x) - prev - 1)
                    prev = int(x)
                max_zero_gap = max(zero_gaps, default=0)

                own_ok = L >= 4 and (cnt >= 3 or run >= 3) and (own_density >= 0.35 or run >= 4)
                occluded_ok = (
                    cnt >= 2 and
                    nonzero_density >= 0.55 and
                    max_zero_gap <= 2 and
                    L >= 5
                )

                if own_ok or occluded_ok:
                    segs.append((y, int(x1), int(x2)))
        return segs

    def vertical_segments(mask, all_nonzero, max_gap=4):
        segs = []
        for x in range(w):
            ys = np.where(mask[:, x])[0]
            if ys.size == 0:
                continue

            clusters = []
            start = int(ys[0])
            end = int(ys[0])
            last = int(ys[0])
            for y in ys[1:]:
                y = int(y)
                if y - last <= max_gap:
                    end = y
                    last = y
                else:
                    clusters.append((start, end))
                    start = y
                    end = y
                    last = y
            clusters.append((start, end))

            for y1, y2 in clusters:
                L = y2 - y1 + 1
                if L < 4:
                    continue

                sub = ys[(ys >= y1) & (ys <= y2)]
                cnt = int(sub.size)
                run = max_run(sub)

                own_density = cnt / L
                nonzero_density = int(np.count_nonzero(all_nonzero[y1:y2 + 1, x])) / L
                zero_gaps = []
                prev = 0
                for y in sub:
                    zero_gaps.append(int(y) - prev - 1)
                    prev = int(y)
                max_zero_gap = max(zero_gaps, default=0)

                own_ok = L >= 4 and (cnt >= 3 or run >= 3) and (own_density >= 0.35 or run >= 4)
                occluded_ok = (
                    cnt >= 2 and
                    nonzero_density >= 0.55 and
                    max_zero_gap <= 2 and
                    L >= 5
                )

                if own_ok or occluded_ok:
                    segs.append((x, int(y1), int(y2)))
        return segs

    def count_cells(mask):
        """Count enclosed 4-connected background components after wall reconstruction."""
        all_nonzero = grid != 0
        h_segs = horizontal_segments(mask, all_nonzero)
        v_segs = vertical_segments(mask, all_nonzero)

        if not h_segs and not v_segs:
            return 0

        wall = np.zeros((h, w), dtype=bool)
        for y, x1, x2 in h_segs:
            wall[y, x1:x2 + 1] = True
        for x, y1, y2 in v_segs:
            wall[y1:y2 + 1, x] = True

        outside = np.zeros((h, w), dtype=bool)
        q = deque()

        for x in range(w):
            if not wall[0, x] and not outside[0, x]:
                outside[0, x] = True
                q.append((0, x))
            if not wall[h - 1, x] and not outside[h - 1, x]:
                outside[h - 1, x] = True
                q.append((h - 1, x))

        for y in range(h):
            if not wall[y, 0] and not outside[y, 0]:
                outside[y, 0] = True
                q.append((y, 0))
            if not wall[y, w - 1] and not outside[y, w - 1]:
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

    def structure_score(c):
        mask = (grid == c)
        all_nonzero = grid != 0
        h_segs = horizontal_segments(mask, all_nonzero)
        v_segs = vertical_segments(mask, all_nonzero)
        strong_h = sum(1 for _, x1, x2 in h_segs if x2 - x1 + 1 >= 4)
        strong_v = sum(1 for _, y1, y2 in v_segs if y2 - y1 + 1 >= 4)
        rooms = count_cells(mask)
        return {
            'rooms': rooms,
            'strong_lines': strong_h + strong_v,
            'score': rooms * 100 + strong_lines
        }

    stats = {c: structure_score(c) for c in colors}

    # Structured colors have at least one enclosed cell and some line evidence.
    structured = [c for c in colors if stats[c]['rooms'] > 0 and stats[c]['strong_lines'] >= 2]

    # Fallback: if the strict test finds nothing, keep colors with any rooms.
    if not structured:
        structured = [c for c in colors if stats[c]['rooms'] > 0]

    # Fallback: if still nothing, use all non-background colors.
    if not structured:
        structured = colors.copy()

    noise_candidates = [c for c in colors if c not in structured]
    if noise_candidates:
        filler = min(noise_candidates, key=lambda c: (stats[c]['score'], c))
    else:
        filler = min(colors, key=lambda c: (stats[c]['score'], c))

    items = []
    for c in structured:
        cnt = stats[c]['rooms']
        if cnt <= 0:
            cnt = 1
        items.append((cnt, c))

    # Sort by true count, then by color value.
    items.sort(key=lambda t: (t[0], t[1]))

    n = len(items)
    out = np.full((n, 4), filler, dtype=np.uint8)
    for i, (cnt, c) in enumerate(items):
        out[i, :min(int(cnt), 4)] = c

    return out