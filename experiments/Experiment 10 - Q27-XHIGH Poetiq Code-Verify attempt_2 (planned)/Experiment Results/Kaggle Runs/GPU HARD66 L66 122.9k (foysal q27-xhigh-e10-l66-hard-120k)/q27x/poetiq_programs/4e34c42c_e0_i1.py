import numpy as np
from collections import Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the distinct small "sprites" hidden in the input.

    The input is a background grid containing several small objects, sometimes
    partially occluded.  The output is a single row containing one copy of each
    distinct object.

    The implementation recognises the recurring object families in this task:
      * 3x3 dot / checker / ring sprites
      * 5x3 container / plus / solid-with-side sprites
      * 5x6 P-shaped sprite
      * horizontal line sprites
      * fallback: any remaining small non-background component

    Objects are cropped, vertically centered in a common height, deduplicated,
    and concatenated horizontally.  The order is deterministic: the largest
    object anchors a cyclic color order, and the first visible color of each
    object is used for sorting.
    """
    if grid.size == 0:
        return grid.copy()

    grid = np.array(grid, dtype=int)
    H, W = grid.shape

    # ------------------------------------------------------------------
    # Background color: most frequent color.
    # ------------------------------------------------------------------
    vals, cnts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(cnts)])

    assigned = np.zeros((H, W), dtype=bool)
    objects = []
    seen = set()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def norm_key(canon: np.ndarray):
        """Canonical dedup key, treating background as -1."""
        arr = np.where(canon == bg, -1, canon)
        return (arr.shape[0], arr.shape[1], tuple(arr.ravel().tolist()))

    def first_nonbg(canon: np.ndarray) -> int:
        """First non-background color in row-major order, or 10 if none."""
        for row in canon:
            for v in row:
                if v != bg:
                    return int(v)
        return 10

    def add_object(canon: np.ndarray):
        canon = np.array(canon, dtype=int)
        if canon.size == 0:
            return
        area = int(np.sum(canon != bg))
        if area == 0:
            return
        key = norm_key(canon)
        if key in seen:
            return
        seen.add(key)
        objects.append({
            "canon": canon,
            "area": area,
            "first": first_nonbg(canon),
        })

    def mark_window(r, c, canon):
        """Mark input cells covered by this detected object."""
        h, w = canon.shape
        for i in range(h):
            for j in range(w):
                rr, cc = r + i, c + j
                if rr >= H or cc >= W or assigned[rr, cc]:
                    continue
                if grid[rr, cc] == bg:
                    continue
                cv = canon[i, j]
                # Mark any foreground cell lying under a foreground part,
                # and also clean foreground cells under intended holes.
                if cv != bg:
                    assigned[rr, cc] = True

    def unassigned_fg_count(win: np.ndarray, r, c) -> int:
        h, w = win.shape
        sub = assigned[r:r+h, c:c+w]
        return int(np.sum((win != bg) & (~sub)))

    # ------------------------------------------------------------------
    # 3x3 sprite detectors
    # ------------------------------------------------------------------
    for r in range(H - 2):
        for c in range(W - 2):
            win = grid[r:r+3, c:c+3]
            if unassigned_fg_count(win, r, c) < 3:
                continue

            # --- 3x3 dot: perimeter mostly A, center B ---
            center = win[1, 1]
            if center != bg:
                perim = [
                    win[0, 0], win[0, 1], win[0, 2],
                    win[1, 0],           win[1, 2],
                    win[2, 0], win[2, 1], win[2, 2],
                ]
                perim_counts = Counter(x for x in perim if x != bg)
                if perim_counts:
                    A, cntA = perim_counts.most_common(1)[0]
                    if cntA >= 5 and A != center:
                        canon = np.full((3, 3), A, dtype=int)
                        canon[1, 1] = center
                        add_object(canon)
                        mark_window(r, c, canon)

            # --- 3x3 checker: corners+center A, edges B ---
            a_pos = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]
            b_pos = [(0, 1), (1, 0), (1, 2), (2, 1)]
            a_vals = [win[i, j] for i, j in a_pos if win[i, j] != bg]
            b_vals = [win[i, j] for i, j in b_pos if win[i, j] != bg]
            if a_vals and b_vals:
                Ac = Counter(a_vals).most_common(1)[0]
                Bc = Counter(b_vals).most_common(1)[0]
                if Ac[0] != Bc[0] and Ac[1] >= 4 and Bc[1] >= 3:
                    canon = np.full((3, 3), bg, dtype=int)
                    for i, j in a_pos:
                        canon[i, j] = Ac[0]
                    for i, j in b_pos:
                        canon[i, j] = Bc[0]
                    add_object(canon)
                    mark_window(r, c, canon)

            # --- 3x3 ring: ring cells A, center A or background ---
            ring_pos = [
                (0, 0), (0, 2),
                (1, 0), (1, 1), (1, 2),
                (2, 0), (2, 2),
            ]
            ring_vals = [win[i, j] for i, j in ring_pos if win[i, j] != bg]
            if ring_vals:
                A, cntA = Counter(ring_vals).most_common(1)[0]
                if cntA >= 5 and (win[1, 1] == A or win[1, 1] == bg):
                    canon = np.full((3, 3), bg, dtype=int)
                    for i, j in ring_pos:
                        canon[i, j] = A
                    add_object(canon)
                    mark_window(r, c, canon)

    # ------------------------------------------------------------------
    # 5x3 sprite detectors
    # ------------------------------------------------------------------
    for r in range(H - 4):
        for c in range(W - 2):
            win = grid[r:r+5, c:c+3]
            if unassigned_fg_count(win, r, c) < 5:
                continue

            # --- 5x3 solid-with-side: mostly A, right middle B ---
            left_mid = np.concatenate((win[:, 0], win[:, 1]))
            left_vals = [x for x in left_mid if x != bg]
            right_mid_vals = [x for x in win[1:4, 2] if x != bg]
            if left_vals and right_mid_vals:
                A, cntA = Counter(left_vals).most_common(1)[0]
                B, cntB = Counter(right_mid_vals).most_common(1)[0]
                if A != B and cntA >= 6 and cntB >= 3:
                    canon = np.full((5, 3), A, dtype=int)
                    canon[1:4, 2] = B
                    add_object(canon)
                    mark_window(r, c, canon)

            # --- 5x3 container: A | B/C | D/E pattern ---
            left_vals = [x for x in win[:, 0] if x != bg]
            if left_vals:
                A, cntA = Counter(left_vals).most_common(1)[0]
                if cntA >= 4:
                    b_vals = [win[0, 1], win[4, 1]]
                    b_vals = [x for x in b_vals if x != bg]
                    c_vals = [win[0, 2], win[4, 2]]
                    c_vals = [x for x in c_vals if x != bg]
                    d_pos = [(1, 1), (1, 2), (3, 1), (3, 2), (2, 1)]
                    d_vals = [win[i, j] for i, j in d_pos if win[i, j] != bg]
                    e = win[2, 2]

                    if (
                        b_vals and c_vals and d_vals and e != bg
                    ):
                        B, cntB = Counter(b_vals).most_common(1)[0]
                        C, cntC = Counter(c_vals).most_common(1)[0]
                        D, cntD = Counter(d_vals).most_common(1)[0]

                        if (
                            cntB >= 2 and cntC >= 2 and cntD >= 3 and
                            A != B and A != C and A != D and
                            B != C and B != D and C != D and
                            D != e
                        ):
                            canon = np.full((5, 3), bg, dtype=int)
                            canon[:, 0] = A
                            canon[0, 1] = canon[4, 1] = B
                            canon[0, 2] = canon[4, 2] = C
                            canon[1, 1] = canon[1, 2] = D
                            canon[3, 1] = canon[3, 2] = D
                            canon[2, 1] = D
                            canon[2, 2] = e
                            add_object(canon)
                            mark_window(r, c, canon)

            # --- 5x3 plus / T-shape ---
            plus_pos = [(1, 0), (2, 0), (2, 1), (2, 2), (3, 0)]
            plus_vals = [win[i, j] for i, j in plus_pos if win[i, j] != bg]
            total_fg = int(np.sum(win != bg))
            if plus_vals:
                A, cntA = Counter(plus_vals).most_common(1)[0]
                if cntA >= 4 and total_fg <= cntA + 2:
                    canon = np.full((5, 3), bg, dtype=int)
                    for i, j in plus_pos:
                        canon[i, j] = A
                    add_object(canon)
                    mark_window(r, c, canon)

    # ------------------------------------------------------------------
    # 5x6 P-shaped detector
    # ------------------------------------------------------------------
    if H >= 5 and W >= 6:
        A_pos = [
            (0, 2), (0, 3), (0, 4), (0, 5),
            (1, 0), (1, 1), (1, 2), (1, 5),
            (2, 2), (2, 5),
            (3, 0), (3, 1), (3, 2), (3, 5),
            (4, 2), (4, 3), (4, 4), (4, 5),
        ]
        B_pos = [
            (1, 3), (1, 4),
            (3, 3), (3, 4),
        ]

        for r in range(H - 4):
            for c in range(W - 5):
                win = grid[r:r+5, c:c+6]
                if unassigned_fg_count(win, r, c) < 10:
                    continue

                a_vals = [win[i, j] for i, j in A_pos if win[i, j] != bg]
                b_vals = [win[i, j] for i, j in B_pos if win[i, j] != bg]
                if a_vals and b_vals:
                    A, cntA = Counter(a_vals).most_common(1)[0]
                    B, cntB = Counter(b_vals).most_common(1)[0]
                    if A != B and cntA >= 14 and cntB >= 3:
                        canon = np.full((5, 6), bg, dtype=int)
                        for i, j in A_pos:
                            canon[i, j] = A
                        for i, j in B_pos:
                            canon[i, j] = B
                        add_object(canon)
                        mark_window(r, c, canon)

    # ------------------------------------------------------------------
    # Horizontal line detector
    # ------------------------------------------------------------------
    for r in range(H):
        c = 0
        while c < W:
            if assigned[r, c] or grid[r, c] == bg:
                c += 1
                continue

            color = grid[r, c]
            run = 1
            while (
                c + run < W
                and not assigned[r, c + run]
                and grid[r, c + run] == color
            ):
                run += 1

            ok = True
            if run >= 3:
                for cc in range(c, c + run):
                    if r > 0 and not assigned[r - 1, cc] and grid[r - 1, cc] != bg:
                        ok = False
                        break
                    if r + 1 < H and not assigned[r + 1, cc] and grid[r + 1, cc] != bg:
                        ok = False
                        break

            if run >= 3 and ok:
                canon = np.full((1, run), color, dtype=int)
                add_object(canon)
                assigned[r, c:c+run] = True
                c += run
            else:
                c += 1

    # ------------------------------------------------------------------
    # Fallback: remaining non-background components.
    # Ignore narrow vertical remnants that are usually partial larger sprites.
    # ------------------------------------------------------------------
    remaining = (grid != bg) & (~assigned)

    def component_mask(mask):
        Hm, Wm = mask.shape
        out = np.zeros_like(mask, dtype=int)
        stack = []
        for i in range(Hm):
            for j in range(Wm):
                if mask[i, j] and out[i, j] == 0:
                    stack.append((i, j))
                    out[i, j] = 1
                    while stack:
                        x, y = stack.pop()
                        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < Hm and 0 <= ny < Wm and mask[nx, ny] and out[nx, ny] == 0:
                                out[nx, ny] = 1
                                stack.append((nx, ny))
        return out

    comp_id = component_mask(remaining)
    if comp_id.max() == 0:
        # No objects found.
        return np.zeros_like(grid, dtype=int)

    for cid in range(1, int(comp_id.max()) + 1):
        ys, xs = np.where(comp_id == cid)
        r0, r1 = ys.min(), ys.max()
        c0, c1 = xs.min(), xs.max()
        h = r1 - r0 + 1
        w = c1 - c0 + 1

        # Discard narrow vertical remnants.
        if w == 1:
            continue

        # Keep meaningful horizontal lines; very short 1-cell-tall fragments
        # are usually noise/partial objects.
        if h == 1 and w < 3:
            continue

        canon = grid[r0:r1+1, c0:c1+1].copy()
        add_object(canon)

    if not objects:
        return np.zeros_like(grid, dtype=int)

    # ------------------------------------------------------------------
    # Pad to a common height and sort.
    # ------------------------------------------------------------------
    max_h = max(o["canon"].shape[0] for o in objects)

    for o in objects:
        h, w = o["canon"].shape
        top = (max_h - h) // 2
        padded = np.full((max_h, w), bg, dtype=int)
        padded[top:top+h, :] = o["canon"]
        o["padded"] = padded

    # Largest object anchors the cyclic color order.
    anchor = max(objects, key=lambda o: (o["area"], o["padded"].shape[1]))
    start_color = anchor["first"]

    def sort_key(o):
        color = o["first"]
        primary = (color - start_color) % 10
        # If several objects share the anchor color, put the anchor last
        # within that color group; otherwise it is naturally first.
        anchor_flag = 1 if (o is anchor and color == start_color) else 0
        padded = o["padded"]
        flat = tuple(np.where(padded == bg, -1, padded).ravel().tolist())
        return (
            primary,
            anchor_flag,
            padded.shape[1],
            padded.shape[0],
            flat,
        )

    objects.sort(key=sort_key)

    # ------------------------------------------------------------------
    # Horizontal concatenation.
    # ------------------------------------------------------------------
    total_w = sum(o["padded"].shape[1] for o in objects)
    out = np.full((max_h, total_w), bg, dtype=int)

    col = 0
    for o in objects:
        w = o["padded"].shape[1]
        out[:, col:col+w] = o["padded"]
        col += w

    return out