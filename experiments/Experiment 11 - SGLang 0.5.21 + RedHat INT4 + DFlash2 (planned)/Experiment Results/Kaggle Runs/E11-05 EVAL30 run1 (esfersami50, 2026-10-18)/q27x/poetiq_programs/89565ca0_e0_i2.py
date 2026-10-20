import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input into a small bar-chart diagram.

    Rule inferred from the examples:
    - The input contains several axis-aligned line grids of different colors.
      A separate color is scattered as noise/filler.
    - For each structured color, count the number of enclosed rectangular
      "cells"/rooms formed by its line segments.  Other colors are ignored
      when counting rooms for a given color; only that color's walls matter.
    - Output one row per structured color.  The row is filled with the
      object's color for `count` cells, then padded with the noise color.
    - Rows are sorted by count (ascending); ties are broken by color value.
    - The output width is fixed at 4 columns, as in all examples.
    """

    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape

    # ------------------------------------------------------------------
    # Helper: detect plausible horizontal/vertical line segments for a color.
    #
    # A real line is dense: it contains many pixels of that color over its
    # span.  Rows/columns containing only intersections of perpendicular
    # lines are sparse and must not be treated as lines.
    # ------------------------------------------------------------------
    def line_segments(mask: np.ndarray, axis: int,
                      gap: int = 3,
                      min_extent: int = 4,
                      min_count: int = 3,
                      min_density: float = 0.55):
        """
        axis=0: horizontal segments (fixed row, variable col)
        axis=1: vertical segments (fixed col, variable row)
        """
        segs = []

        if axis == 0:
            for y in range(mask.shape[0]):
                xs = np.where(mask[y])[0]
                if xs.size == 0:
                    continue

                # Cluster x positions, allowing small occlusion gaps.
                clusters = []
                start = int(xs[0])
                end = int(xs[0])
                last = int(xs[0])
                for x in xs[1:]:
                    x = int(x)
                    if x - last <= gap:
                        end = x
                        last = x
                    else:
                        clusters.append((start, end, [start]))
                        start = x
                        end = x
                        last = x
                clusters.append((start, end, [start]))

                # Redo a lighter pass to keep count/max_run inside cluster.
                for start, end, _ in clusters:
                    sub = xs[(xs >= start) & (xs <= end)]
                    if sub.size == 0:
                        continue
                    extent = int(sub[-1] - sub[0] + 1)
                    count = int(sub.size)
                    if extent < min_extent or count < min_count:
                        continue

                    # Density over the span.
                    if count / extent < min_density:
                        # Accept if there is a long continuous run.
                        max_run = 0
                        cur = 1
                        for i in range(1, sub.size):
                            if sub[i] == sub[i - 1] + 1:
                                cur += 1
                            else:
                                cur = 1
                            max_run = max(max_run, cur)
                        if max_run < 3:
                            continue

                    segs.append((int(sub[0]), int(sub[-1])))

        else:
            for x in range(mask.shape[1]):
                ys = np.where(mask[:, x])[0]
                if ys.size == 0:
                    continue

                clusters = []
                start = int(ys[0])
                end = int(ys[0])
                last = int(ys[0])
                for y in ys[1:]:
                    y = int(y)
                    if y - last <= gap:
                        end = y
                        last = y
                    else:
                        clusters.append((start, end))
                        start = y
                        end = y
                        last = y
                clusters.append((start, end))

                for start, end in clusters:
                    sub = ys[(ys >= start) & (ys <= end)]
                    if sub.size == 0:
                        continue
                    extent = int(sub[-1] - sub[0] + 1)
                    count = int(sub.size)
                    if extent < min_extent or count < min_count:
                        continue

                    if count / extent < min_density:
                        max_run = 0
                        cur = 1
                        for i in range(1, sub.size):
                            if sub[i] == sub[i - 1] + 1:
                                cur += 1
                            else:
                                cur = 1
                            max_run = max(max_run, cur)
                        if max_run < 3:
                            continue

                    segs.append((int(sub[0]), int(sub[-1])))

        return segs

    # ------------------------------------------------------------------
    # Helper: count enclosed rooms for one color from reconstructed walls.
    # ------------------------------------------------------------------
    def count_rooms(h_segs, v_segs):
        if not h_segs and not v_segs:
            return 0

        wall = np.zeros((h, w), dtype=bool)

        for x1, x2 in h_segs:
            # h_segs stored as (start, end); we need y too. This helper is
            # used below with combined segments instead.
            pass

        # This function is not used directly in this form; kept for clarity.
        return 0

    def count_rooms_for_color(mask: np.ndarray) -> int:
        h_pos = np.where(mask.any(axis=1))[0]
        v_pos = np.where(mask.any(axis=0))[0]
        if h_pos.size == 0 or v_pos.size == 0:
            return 0

        h_segs = line_segments(mask, axis=0)
        v_segs = line_segments(mask, axis=1)

        # If a color has no sufficiently line-like structure, it is noise.
        if not h_segs or not v_segs:
            return 0

        wall = np.zeros((h, w), dtype=bool)

        # Add horizontal segments.  line_segments returns (x_start, x_end)
        # for the row it was processing; we need row index, so recompute
        # in a combined way below to keep row/col associated.
        #
        # Instead of the previous loose return, build wall directly here.
        # ------------------------------------------------------------------
        # Re-run detection but store with fixed coordinate.
        # ------------------------------------------------------------------
        def horizontal_segments_with_y(mask):
            segs = []
            for y in range(mask.shape[0]):
                xs = np.where(mask[y])[0]
                if xs.size == 0:
                    continue

                clusters = []
                start = int(xs[0])
                end = int(xs[0])
                last = int(xs[0])
                for x in xs[1:]:
                    x = int(x)
                    if x - last <= 3:
                        end = x
                        last = x
                    else:
                        clusters.append((start, end))
                        start = x
                        end = x
                        last = x
                clusters.append((start, end))

                for x1, x2 in clusters:
                    sub = xs[(xs >= x1) & (xs <= x2)]
                    if sub.size == 0:
                        continue
                    extent = int(sub[-1] - sub[0] + 1)
                    count = int(sub.size)
                    if extent < 4 or count < 3:
                        continue

                    if count / extent < 0.55:
                        max_run = 0
                        cur = 1
                        for i in range(1, sub.size):
                            if sub[i] == sub[i - 1] + 1:
                                cur += 1
                            else:
                                cur = 1
                            max_run = max(max_run, cur)
                        if max_run < 3:
                            continue

                    segs.append((y, int(sub[0]), int(sub[-1])))
            return segs

        def vertical_segments_with_x(mask):
            segs = []
            for x in range(mask.shape[1]):
                ys = np.where(mask[:, x])[0]
                if ys.size == 0:
                    continue

                clusters = []
                start = int(ys[0])
                end = int(ys[0])
                last = int(ys[0])
                for y in ys[1:]:
                    y = int(y)
                    if y - last <= 3:
                        end = y
                        last = y
                    else:
                        clusters.append((start, end))
                        start = y
                        end = y
                        last = y
                clusters.append((start, end))

                for y1, y2 in clusters:
                    sub = ys[(ys >= y1) & (ys <= y2)]
                    if sub.size == 0:
                        continue
                    extent = int(sub[-1] - sub[0] + 1)
                    count = int(sub.size)
                    if extent < 4 or count < 3:
                        continue

                    if count / extent < 0.55:
                        max_run = 0
                        cur = 1
                        for i in range(1, sub.size):
                            if sub[i] == sub[i - 1] + 1:
                                cur += 1
                            else:
                                cur = 1
                            max_run = max(max_run, cur)
                        if max_run < 3:
                            continue

                    segs.append((x, int(sub[0]), int(sub[-1])))
            return segs

        h_segs2 = horizontal_segments_with_y(mask)
        v_segs2 = vertical_segments_with_x(mask)

        if not h_segs2 or not v_segs2:
            return 0

        wall = np.zeros((h, w), dtype=bool)

        for y, x1, x2 in h_segs2:
            wall[y, x1:x2 + 1] = True

        for x, y1, y2 in v_segs2:
            wall[y1:y2 + 1, x] = True

        # The outside of the object is the non-wall component connected to
        # the image border.  All other non-wall components are enclosed cells.
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

        # Count remaining non-wall, non-outside components.
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
    # Score every non-background color and choose structured colors/noise.
    # ------------------------------------------------------------------
    colors = [int(c) for c in np.unique(grid) if c != 0]
    if not colors:
        return np.zeros_like(grid)

    scores = {}
    masks = {}
    for c in colors:
        mask = (grid == c)
        masks[c] = mask
        rooms = count_rooms_for_color(mask)
        # A structured color must have at least one enclosed room.
        scores[c] = rooms

    structured = [c for c in colors if scores[c] > 0]

    # If no color was recognized as structured (should not happen for valid
    # ARC cases), fall back to all non-background colors.
    if not structured:
        structured = colors.copy()

    # Noise/filler color: the non-structured color, or lowest-score color
    # if all colors look structured.
    noise_candidates = [c for c in colors if c not in structured]
    if noise_candidates:
        filler = min(noise_candidates, key=lambda c: (scores.get(c, 0), c))
    else:
        filler = min(colors, key=lambda c: (scores[c], c))

    # ------------------------------------------------------------------
    # Build output: fixed 4 columns, sorted by count ascending.
    # ------------------------------------------------------------------
    items = []
    for c in structured:
        cnt = scores[c]
        if cnt <= 0:
            continue
        items.append((cnt, c))

    # If fallback produced items with zero counts, still include them.
    if not items:
        items = [(1, c) for c in colors]

    items.sort(key=lambda t: (t[0], t[1]))

    out = np.full((len(items), 4), filler, dtype=np.uint8)
    for i, (cnt, c) in enumerate(items):
        n = max(0, min(int(cnt), 4))
        out[i, :n] = c

    return out