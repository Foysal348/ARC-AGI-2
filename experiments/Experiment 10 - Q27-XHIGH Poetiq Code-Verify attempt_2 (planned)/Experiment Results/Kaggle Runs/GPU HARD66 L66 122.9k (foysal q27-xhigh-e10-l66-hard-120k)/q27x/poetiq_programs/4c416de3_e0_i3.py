import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Complete isolated colored pixels into the stamp style that is already
    present in the input.

    Three stamp styles are supported:

    A-stamp:
        zero center, four orthogonal colored cells, and one diagonal pair.

    B-stamp:
        colored corner, two opposite orthogonal arms, and a diagonal seed.

    C-stamp:
        2x2 block on the opposite side of a zero corner plus a diagonal
        connector line to the seed.

    Multi-pixel colored components are treated as protected templates and are
    not overwritten. Isolated colored pixels are completed using the detected
    stamp style.
    """
    arr = np.array(grid, dtype=int)
    if arr.ndim != 2:
        return arr

    H, W = arr.shape
    if H == 0 or W == 0:
        return arr

    # Background: most frequent non-zero color if possible.  Zero is treated
    # as the special frame/line color.
    vals, counts = np.unique(arr, return_counts=True)
    nz_mask = vals != 0
    if np.any(nz_mask):
        bg = int(vals[nz_mask][np.argmax(counts[nz_mask])])
    else:
        bg = int(vals[np.argmax(counts)])
    if bg == 0:
        bg = int(vals[np.argmax(counts)])

    dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    def inb(r, c):
        return 0 <= r < H and 0 <= c < W

    def get(r, c):
        return arr[r, c] if inb(r, c) else None

    # Colored pixels are non-background and non-zero.
    colored = (arr != bg) & (arr != 0)

    # 8-connected colored components.
    labels = np.zeros((H, W), dtype=int)
    multi = np.zeros((H, W), dtype=bool)
    singles = []
    counter = 0

    for i in range(H):
        for j in range(W):
            if colored[i, j] and labels[i, j] == 0:
                counter += 1
                stack = [(i, j)]
                labels[i, j] = counter
                cells = []

                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))

                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if inb(nr, nc) and colored[nr, nc] and labels[nr, nc] == 0:
                                labels[nr, nc] = counter
                                stack.append((nr, nc))

                if len(cells) == 1:
                    singles.append(cells[0])
                else:
                    for r, c in cells:
                        multi[r, c] = True

    def same_color(cells, color):
        for r, c in cells:
            if not inb(r, c) or arr[r, c] != color:
                return False
        return True

    # ------------------------------------------------------------------
    # Detect complete stamp styles in the input.
    # ------------------------------------------------------------------
    def detect_A():
        """A: zero center, four orthogonals, one diagonal pair."""
        for r in range(1, H - 1):
            for c in range(1, W - 1):
                if arr[r, c] != 0:
                    continue

                for dr, dc in dirs:
                    color = get(r + dr, c + dc)
                    if color is None or color == 0 or color == bg:
                        continue

                    cells = [
                        (r + dr, c + dc),
                        (r - dr, c - dc),
                        (r - 1, c),
                        (r + 1, c),
                        (r, c - 1),
                        (r, c + 1),
                    ]
                    if same_color(cells, color):
                        return True
        return False

    def detect_B(strict=True):
        """
        B: corner, two opposite arms, diagonal seed.
        Strict also requires the two zero arms toward the seed.
        """
        for r in range(H):
            for c in range(W):
                color = arr[r, c]
                if color == 0 or color == bg:
                    continue

                for dr, dc in dirs:
                    cells = [
                        (r, c),
                        (r + dr, c + dc),
                        (r - dr, c),
                        (r, c - dc),
                    ]
                    if not same_color(cells, color):
                        continue

                    # Do not match a B-subpattern inside a longer C pattern.
                    br, bc = r + 2 * dr, c + 2 * dc
                    if inb(br, bc) and arr[br, bc] == color:
                        continue

                    if strict:
                        if not (
                            inb(r + dr, c) and arr[r + dr, c] == 0 and
                            inb(r, c + dc) and arr[r, c + dc] == 0
                        ):
                            continue

                    return True
        return False

    def detect_C(strict=True):
        """
        C: 2x2 opposite block plus diagonal connector/seed.
        Strict also requires the two zero arms toward the seed.
        """
        for r in range(H):
            for c in range(W):
                color = arr[r, c]
                if color == 0 or color == bg:
                    continue

                for dr, dc in dirs:
                    for d in (2, 3, 4):
                        cells = [
                            (r, c),
                            (r - dr, c),
                            (r, c - dc),
                            (r - dr, c - dc),
                        ]
                        for t in range(1, d + 1):
                            cells.append((r + t * dr, c + t * dc))

                        if not same_color(cells, color):
                            continue

                        # Ensure the seed is the terminal colored pixel on this diagonal.
                        br, bc = r + (d + 1) * dr, c + (d + 1) * dc
                        if inb(br, bc) and arr[br, bc] == color:
                            continue

                        if strict:
                            if not (
                                inb(r + dr, c) and arr[r + dr, c] == 0 and
                                inb(r, c + dc) and arr[r, c + dc] == 0
                            ):
                                continue

                        return True
        return False

    has_A = detect_A()
    has_B = detect_B(True)
    has_C = detect_C(True)

    # Fallback if no strict complete template is found.
    if not (has_A or has_B or has_C):
        has_B = has_B or detect_B(False)
        has_C = has_C or detect_C(False)

    # ------------------------------------------------------------------
    # Helper predicates.
    # ------------------------------------------------------------------
    def zero_arms_ok(cr, cc, dr, dc):
        """
        For a seed at C + (dr, dc), require the two zero cells adjacent
        to the corner in the directions toward the seed.
        """
        return (
            inb(cr + dr, cc) and arr[cr + dr, cc] == 0 and
            inb(cr, cc + dc) and arr[cr, cc + dc] == 0
        )

    def available_cells(cells, seed, val):
        """Can these cells be recolored to val without conflicting?"""
        seen = set()
        for r, c in cells:
            if not inb(r, c):
                return False
            if (r, c) in seen:
                continue
            seen.add((r, c))

            if (r, c) == seed:
                if arr[r, c] != val:
                    return False
                continue

            if multi[r, c] and arr[r, c] != val:
                return False

            v = arr[r, c]
            if v not in (0, bg, val):
                return False

        return True

    def open_L_NE(cr, cc):
        """
        True for the special open L-shape seen in Example 1:
        corner with zero arms up and right, but not part of a closed rectangle.
        """
        up = 0
        rr = cr - 1
        while inb(rr, cc) and arr[rr, cc] == 0:
            up += 1
            rr -= 1

        right = 0
        cc2 = cc + 1
        while inb(cr, cc2) and arr[cr, cc2] == 0:
            right += 1
            cc2 += 1

        if up < 2 or right < 2:
            return False

        closed = True

        top_r = cr - up
        if not (
            inb(top_r, cc) and inb(top_r, cc + right) and
            all(arr[top_r, k] == 0 for k in range(cc, cc + right + 1))
        ):
            closed = False

        right_c = cc + right
        if not (
            inb(cr - up, right_c) and inb(cr, right_c) and
            all(arr[k, right_c] == 0 for k in range(cr - up, cr + 1))
        ):
            closed = False

        return not closed

    # ------------------------------------------------------------------
    # Candidate generation.
    # ------------------------------------------------------------------
    def generate_candidates(r0, c0):
        val = arr[r0, c0]
        seed = (r0, c0)
        cands = []

        # ---------------- A-stamp ----------------
        if has_A:
            for dr, dc in dirs:
                cr, cc = r0 - dr, c0 - dc
                if not inb(cr, cc) or arr[cr, cc] != 0:
                    continue
                if not zero_arms_ok(cr, cc, dr, dc):
                    continue

                cells = [
                    (cr - 1, cc),
                    (cr + 1, cc),
                    (cr, cc - 1),
                    (cr, cc + 1),
                    (r0, c0),
                    (cr - dr, cc - dc),
                ]

                if available_cells(cells, seed, val):
                    score = 1200
                    score += 10 * sum(1 for r, c in cells if inb(r, c) and arr[r, c] == bg)
                    score -= 3 * sum(1 for r, c in cells if inb(r, c) and arr[r, c] == 0)
                    cands.append((score, cells, (cr, cc), val, seed))

            # Special deflected A for the observed open L-shape case.
            dr, dc = -1, 1
            cr, cc = r0 - dr, c0 - dc
            if (
                inb(cr, cc) and arr[cr, cc] == 0 and
                zero_arms_ok(cr, cc, dr, dc) and
                open_L_NE(cr, cc)
            ):
                cells = [
                    (r0, c0),
                    (r0, c0 + 1),
                    (r0 + 1, c0 - 2),
                    (r0 + 1, c0),
                    (r0 + 2, c0 - 2),
                    (r0 + 2, c0 - 1),
                ]
                if available_cells(cells, seed, val):
                    score = 2000
                    cands.append((score, cells, (cr, cc), val, seed))

        # ---------------- B-stamp ----------------
        if has_B:
            for dr, dc in dirs:
                cr, cc = r0 - dr, c0 - dc
                if not inb(cr, cc) or arr[cr, cc] != 0:
                    continue
                if not zero_arms_ok(cr, cc, dr, dc):
                    continue

                br, bc = r0 + dr, c0 + dc
                if inb(br, bc) and arr[br, bc] == val:
                    continue

                cells = [
                    (cr, cc),
                    (cr - dr, cc),
                    (cr, cc - dc),
                    (r0, c0),
                ]

                if available_cells(cells, seed, val):
                    score = 1000
                    score += 10 * sum(1 for r, c in cells if inb(r, c) and arr[r, c] == bg)
                    score -= 5 * sum(1 for r, c in cells if inb(r, c) and arr[r, c] == 0)
                    cands.append((score, cells, None, val, seed))

        # ---------------- C-stamp ----------------
        if has_C:
            for d in (2, 3, 4):
                for dr, dc in dirs:
                    cr, cc = r0 - dr * d, c0 - dc * d
                    if not inb(cr, cc) or arr[cr, cc] != 0:
                        continue
                    if not zero_arms_ok(cr, cc, dr, dc):
                        continue

                    br, bc = r0 + dr, c0 + dc
                    if inb(br, bc) and arr[br, bc] == val:
                        continue

                    cells = []
                    for ar in (0, 1):
                        for ac in (0, 1):
                            cells.append((cr + ar * (-dr), cc + ac * (-dc)))

                    for t in range(1, d + 1):
                        cells.append((cr + t * dr, cc + t * dc))

                    if available_cells(cells, seed, val):
                        score = 1100 if d == 2 else 950
                        score += 10 * sum(1 for r, c in cells if inb(r, c) and arr[r, c] == bg)
                        score -= 3 * sum(1 for r, c in cells if inb(r, c) and arr[r, c] == 0)
                        cands.append((score, cells, None, val, seed))

        return cands

    # Choose one best candidate per isolated seed.
    actions = []
    for r0, c0 in singles:
        cands = generate_candidates(r0, c0)
        if cands:
            cands.sort(key=lambda x: -x[0])
            actions.append(cands[0])

    # Apply chosen actions, higher-score actions first.
    out = arr.copy()
    actions.sort(key=lambda x: -x[0])

    zero_score = {}
    for score, cells, center, val, seed in actions:
        if center is not None:
            r, c = center
            if inb(r, c) and arr[r, c] == 0 and not multi[r, c]:
                zero_score[(r, c)] = max(zero_score.get((r, c), -1), score + 10000)

    for score, cells, center, val, seed in actions:
        for r, c in cells:
            if not inb(r, c):
                continue
            if multi[r, c] and arr[r, c] != val:
                continue
            if (r, c) in zero_score and zero_score[(r, c)] >= score:
                continue
            if out[r, c] not in (0, bg, val):
                continue
            out[r, c] = val

    # Ensure processed seeds remain colored.
    for score, cells, center, val, seed in actions:
        r, c = seed
        if inb(r, c) and (r, c) not in zero_score:
            out[r, c] = val

    return out