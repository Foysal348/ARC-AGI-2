import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Recover the inner panel from the framed input.

    The input contains a rectangular panel with two adjacent one-cell tabs.
    Each tab color matches a long solid color run on one side of the outer
    frame. The matching tells us which outer side each tabbed panel side
    should become in the output. We apply the unique rotation/reflection
    consistent with those two side correspondences.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    if H == 0 or W == 0:
        return g.copy()
    if H < 3 or W < 3:
        return g.copy()

    TOP, RIGHT, BOTTOM, LEFT = 0, 1, 2, 3

    # mapping[source_side] = output_side for each of the 8 dihedral transforms
    TRANSFORMS = [
        ("id",        (TOP, RIGHT, BOTTOM, LEFT)),
        ("cw",        (RIGHT, BOTTOM, LEFT, TOP)),
        ("180",       (BOTTOM, LEFT, TOP, RIGHT)),
        ("ccw",       (LEFT, TOP, RIGHT, BOTTOM)),
        ("fliph",     (TOP, LEFT, BOTTOM, RIGHT)),
        ("flipv",     (BOTTOM, RIGHT, TOP, LEFT)),
        ("transpose", (LEFT, BOTTOM, RIGHT, TOP)),
        ("anti",      (RIGHT, TOP, LEFT, BOTTOM)),
    ]

    ADJACENT = [
        {TOP, RIGHT},
        {RIGHT, BOTTOM},
        {BOTTOM, LEFT},
        {LEFT, TOP},
    ]

    def apply_tf(name, arr):
        if name == "id":
            out = arr
        elif name == "cw":
            out = np.rot90(arr, k=-1)
        elif name == "180":
            out = np.rot90(arr, k=2)
        elif name == "ccw":
            out = np.rot90(arr, k=1)
        elif name == "fliph":
            out = np.fliplr(arr)
        elif name == "flipv":
            out = np.flipud(arr)
        elif name == "transpose":
            out = arr.T
        elif name == "anti":
            out = arr[::-1, ::-1].T
        else:
            out = np.rot90(arr, k=1)
        return np.array(out, copy=True)

    def adjacent(a, b):
        return {a, b} in ADJACENT

    def find_transform(constraints):
        for name, mapping in TRANSFORMS:
            if all(mapping[s] == t for s, t in constraints):
                return name
        return None

    def outer_side_info(color):
        """Return the outer side best matching this color, plus confidence."""
        best = None
        side_lines = [
            (TOP, g[0, :]),
            (BOTTOM, g[-1, :]),
            (LEFT, g[:, 0]),
            (RIGHT, g[:, -1]),
        ]

        for side, line in side_lines:
            mask = (line == color)
            cnt = int(mask.sum())
            if cnt == 0:
                continue

            run = 0
            cur = 0
            for b in mask:
                if b:
                    cur += 1
                    if cur > run:
                        run = cur
                else:
                    cur = 0

            score = (run == line.size, run, cnt)
            if best is None or score > best[0]:
                best = (score, side)

        if best is None:
            return None, 0, 0

        score, side = best
        return side, score[1], score[2]

    def get_tabs(r1, r2, c1, c2, bg):
        """
        Detect solid nonzero tabs just outside the candidate panel.
        Tab tuple:
            (side, color, start, end, line_coord)
        For horizontal tabs, start/end are columns and line_coord is row.
        For vertical tabs, start/end are rows and line_coord is column.
        """
        tabs = []
        bg = int(bg)

        # Top tab
        if r1 - 1 >= 0:
            row = r1 - 1
            seg = g[row, c1:c2 + 1]
            if seg.size > 0 and np.all(seg == seg[0]):
                val = int(seg[0])
                if val != 0 and val != bg:
                    left = c1
                    while left > 0 and g[row, left - 1] == val:
                        left -= 1
                    right = c2
                    while right < W - 1 and g[row, right + 1] == val:
                        right += 1
                    tabs.append((TOP, val, left, right, row))

        # Right tab
        if c2 + 1 < W:
            col = c2 + 1
            seg = g[r1:r2 + 1, col]
            if seg.size > 0 and np.all(seg == seg[0]):
                val = int(seg[0])
                if val != 0 and val != bg:
                    top = r1
                    while top > 0 and g[top - 1, col] == val:
                        top -= 1
                    bottom = r2
                    while bottom < H - 1 and g[bottom + 1, col] == val:
                        bottom += 1
                    tabs.append((RIGHT, val, top, bottom, col))

        # Bottom tab
        if r2 + 1 < H:
            row = r2 + 1
            seg = g[row, c1:c2 + 1]
            if seg.size > 0 and np.all(seg == seg[0]):
                val = int(seg[0])
                if val != 0 and val != bg:
                    left = c1
                    while left > 0 and g[row, left - 1] == val:
                        left -= 1
                    right = c2
                    while right < W - 1 and g[row, right + 1] == val:
                        right += 1
                    tabs.append((BOTTOM, val, left, right, row))

        # Left tab
        if c1 - 1 >= 0:
            col = c1 - 1
            seg = g[r1:r2 + 1, col]
            if seg.size > 0 and np.all(seg == seg[0]):
                val = int(seg[0])
                if val != 0 and val != bg:
                    top = r1
                    while top > 0 and g[top - 1, col] == val:
                        top -= 1
                    bottom = r2
                    while bottom < H - 1 and g[bottom + 1, col] == val:
                        bottom += 1
                    tabs.append((LEFT, val, top, bottom, col))

        return tabs

    def bbox_from_pair(t1, t2):
        """Given one horizontal and one vertical adjacent tab, infer panel bbox."""
        horiz = None
        vert = None

        for t in (t1, t2):
            if t[0] in (TOP, BOTTOM):
                horiz = t
            else:
                vert = t

        if horiz is None or vert is None:
            return None

        hs, he, hline = horiz[2], horiz[3], horiz[4]
        vs, ve, vline = vert[2], vert[3], vert[4]

        r1, r2 = vs, ve
        c1, c2 = hs, he

        if r2 < r1 or c2 < c1:
            return None

        if horiz[0] == TOP:
            if hline != r1 - 1:
                return None
        else:  # BOTTOM
            if hline != r2 + 1:
                return None

        if vert[0] == LEFT:
            if vline != c1 - 1:
                return None
        else:  # RIGHT
            if vline != c2 + 1:
                return None

        return (r1, r2, c1, c2)

    def gen_candidates(strict):
        """
        Generate candidate panel bounding boxes from dense color regions.
        Strict mode avoids regions touching the outer border.
        """
        cands = []

        for color in np.unique(g):
            color = int(color)
            mask = (g == color)
            count = int(mask.sum())

            if count < 5:
                continue

            ys, xs = np.nonzero(mask)
            r1 = int(ys.min())
            r2 = int(ys.max())
            c1 = int(xs.min())
            c2 = int(xs.max())

            if strict and (r1 <= 0 or r2 >= H - 1 or c1 <= 0 or c2 >= W - 1):
                continue

            h = r2 - r1 + 1
            w = c2 - c1 + 1
            if h < 2 or w < 2:
                continue

            area = h * w
            fill = count / area
            score = count * fill

            # Deprioritize zero, which is usually the exterior background.
            if color == 0:
                score *= 0.1

            cands.append((score, fill, count, area, color, r1, r2, c1, c2))

        cands.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
        return cands

    def try_candidate(strict):
        for score, fill, count, area, color, r1, r2, c1, c2 in gen_candidates(strict):
            tabs = get_tabs(r1, r2, c1, c2, color)
            if len(tabs) < 2:
                continue

            best = None
            n = len(tabs)

            for i in range(n):
                for j in range(i + 1, n):
                    t1, t2 = tabs[i], tabs[j]

                    if not adjacent(t1[0], t2[0]):
                        continue

                    bbox = bbox_from_pair(t1, t2)
                    if bbox is None:
                        continue

                    fr1, fr2, fc1, fc2 = bbox
                    if fr1 < 0 or fr2 >= H or fc1 < 0 or fc2 >= W:
                        continue

                    sub = g[fr1:fr2 + 1, fc1:fc2 + 1]
                    sub_count = int(np.sum(sub == color))

                    # The candidate color should mostly fill the final panel.
                    if sub_count < 4 or sub_count / sub.size < 0.1:
                        continue

                    target1, run1, cnt1 = outer_side_info(t1[1])
                    target2, run2, cnt2 = outer_side_info(t2[1])

                    if target1 is None or target2 is None or target1 == target2:
                        continue

                    tf = find_transform([(t1[0], target1), (t2[0], target2)])
                    if tf is None:
                        continue

                    conf = (run1 + run2, cnt1 + cnt2)
                    if best is None or conf > best[0]:
                        best = (conf, tf, bbox)

            if best is not None:
                conf, tf, bbox = best
                fr1, fr2, fc1, fc2 = bbox
                panel = g[fr1:fr2 + 1, fc1:fc2 + 1]
                return apply_tf(tf, panel)

        return None

    out = try_candidate(True)
    if out is not None:
        return out

    out = try_candidate(False)
    if out is not None:
        return out

    # Fallback: use the best-looking rectangle and, if possible, apply a valid
    # transformation. Otherwise rotate counterclockwise as a safe default.
    cands = gen_candidates(True)
    if not cands:
        cands = gen_candidates(False)

    if cands:
        _, _, _, _, color, r1, r2, c1, c2 = cands[0]
        panel = g[r1:r2 + 1, c1:c2 + 1]
        tabs = get_tabs(r1, r2, c1, c2, color)

        for i in range(len(tabs)):
            for j in range(i + 1, len(tabs)):
                t1, t2 = tabs[i], tabs[j]

                if not adjacent(t1[0], t2[0]):
                    continue

                bbox = bbox_from_pair(t1, t2)
                if bbox is None:
                    continue

                fr1, fr2, fc1, fc2 = bbox
                if fr1 < 0 or fr2 >= H or fc1 < 0 or fc2 >= W:
                    continue

                panel = g[fr1:fr2 + 1, fc1:fc2 + 1]

                target1, _, _ = outer_side_info(t1[1])
                target2, _, _ = outer_side_info(t2[1])

                if target1 is not None and target2 is not None and target1 != target2:
                    tf = find_transform([(t1[0], target1), (t2[0], target2)])
                    if tf is not None:
                        return apply_tf(tf, panel)

        return np.rot90(panel, k=1)

    return g.copy()