import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Complete single-pixel colored stamps near zero frames/corners.

    The function protects multi-pixel colored components and completes isolated
    colored pixels into one of three pattern styles:

      A: 3x3 with a zero/background center, four side cells, and one opposite
         diagonal pair.
      B: a corner cell plus two outward orthogonal cells plus a diagonal seed.
      C: a 2x2 opposite block plus diagonal connector(s) and seed.

    Pattern type and local shift are inferred from complete components and the
    size/orientation of the zero component containing the anchor corner.
    """
    arr = np.array(grid, dtype=int)
    if arr.ndim != 2:
        return arr

    H, W = arr.shape
    if H == 0 or W == 0:
        return arr

    # Background is the most frequent color.
    vals, counts = np.unique(arr, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    def inb(r, c):
        return 0 <= r < H and 0 <= c < W

    # ------------------------------------------------------------------
    # 8-connected components of colored pixels.
    # ------------------------------------------------------------------
    colored = (arr != bg) & (arr != 0)
    labels = np.zeros((H, W), dtype=int)
    sizes = [0]
    multi = np.zeros((H, W), dtype=bool)
    singles = []

    eight = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),           (0, 1),
             (1, -1),  (1, 0),  (1, 1)]

    for i in range(H):
        for j in range(W):
            if colored[i, j] and labels[i, j] == 0:
                lab = len(sizes)
                sizes.append(0)
                stack = [(i, j)]
                labels[i, j] = lab
                cells = []

                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))
                    sizes[lab] += 1

                    for dr, dc in eight:
                        nr, nc = r + dr, c + dc
                        if inb(nr, nc) and colored[nr, nc] and labels[nr, nc] == 0:
                            labels[nr, nc] = lab
                            stack.append((nr, nc))

                if sizes[lab] > 1:
                    for r, c in cells:
                        multi[r, c] = True
                else:
                    singles.append(cells[0])

    # ------------------------------------------------------------------
    # 4-connected zero components and their vertical heights.
    # ------------------------------------------------------------------
    zero_labels = np.zeros((H, W), dtype=int)
    zero_height = {}
    zlab = 0
    four = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(H):
        for j in range(W):
            if arr[i, j] == 0 and zero_labels[i, j] == 0:
                zlab += 1
                stack = [(i, j)]
                zero_labels[i, j] = zlab
                min_r = max_r = i

                while stack:
                    r, c = stack.pop()
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)

                    for dr, dc in four:
                        nr, nc = r + dr, c + dc
                        if inb(nr, nc) and arr[nr, nc] == 0 and zero_labels[nr, nc] == 0:
                            zero_labels[nr, nc] = zlab
                            stack.append((nr, nc))

                zero_height[zlab] = max_r - min_r + 1

    def comp_size(r, c):
        lab = labels[r, c]
        if lab == 0:
            return 0
        return sizes[lab]

    def get(r, c):
        if inb(r, c):
            return arr[r, c]
        return None

    # ------------------------------------------------------------------
    # Detect complete pattern styles in the input.
    # ------------------------------------------------------------------
    nonbg_colors = [int(v) for v in vals if v not in (0, bg)]
    diag_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    def detect_A():
        # A complete A-stamp: 6 colored cells around a non-colored center.
        for r in range(1, H - 1):
            for c in range(1, W - 1):
                if arr[r, c] not in (0, bg):
                    continue
                for color in nonbg_colors:
                    orth_ok = (
                        get(r - 1, c) == color and
                        get(r + 1, c) == color and
                        get(r, c - 1) == color and
                        get(r, c + 1) == color
                    )
                    if not orth_ok:
                        continue

                    if (get(r - 1, c - 1) == color and
                        get(r + 1, c + 1) == color):
                        lab = labels[r - 1, c]
                        if lab and sizes[lab] >= 6:
                            return True

                    if (get(r - 1, c + 1) == color and
                        get(r + 1, c - 1) == color):
                        lab = labels[r - 1, c + 1]
                        if lab and sizes[lab] >= 6:
                            return True
        return False

    def detect_B():
        # Complete B-stamp: corner + two outward orthogonals + diagonal seed.
        # Require zero arms toward the seed, and keep it for relatively small
        # components to avoid false positives from C-stamps.
        for r in range(H):
            for c in range(W):
                color = arr[r, c]
                if color == 0 or color == bg:
                    continue

                for dr, dc in diag_dirs:
                    if not (inb(r + dr, c) and arr[r + dr, c] == 0 and
                            inb(r, c + dc) and arr[r, c + dc] == 0):
                        continue

                    if (get(r, c) == color and
                        get(r - dr, c) == color and
                        get(r, c - dc) == color and
                        get(r + dr, c + dc) == color):
                        lab = labels[r, c]
                        if lab == 0 or sizes[lab] <= 5:
                            return True
        return False

    def detect_C():
        # Complete C-stamp: 2x2 opposite block + connector + seed at distance 2.
        for r in range(H):
            for c in range(W):
                color = arr[r, c]
                if color == 0 or color == bg:
                    continue

                for dr, dc in diag_dirs:
                    if not (inb(r + dr, c) and arr[r + dr, c] == 0 and
                            inb(r, c + dc) and arr[r, c + dc] == 0):
                        continue

                    if (get(r, c) == color and
                        get(r - dr, c) == color and
                        get(r, c - dc) == color and
                        get(r - dr, c - dc) == color and
                        get(r + dr, c + dc) == color and
                        get(r + 2 * dr, c + 2 * dc) == color):
                        lab = labels[r, c]
                        if lab == 0 or sizes[lab] >= 6:
                            return True
        return False

    has_A = detect_A()
    has_B = detect_B()
    has_C = detect_C()

    # ------------------------------------------------------------------
    # Candidate generation helpers.
    # ------------------------------------------------------------------
    def cells_ok(cells, val):
        for r, c in cells:
            if not inb(r, c):
                return False
            if multi[r, c] and arr[r, c] != val:
                return False
            v = arr[r, c]
            if v not in (0, bg, val):
                return False
        return True

    def center_ok(center, val):
        if center is None:
            return True
        r, c = center
        if not inb(r, c):
            return False
        if multi[r, c]:
            return False
        return arr[r, c] in (0, bg, val)

    def score_candidate(typ, shift, d, dr, zero_h, ncells):
        score = 0

        if typ == "A":
            score += 110 if has_A else -100
        elif typ == "B":
            score += 100 if has_B else -100
        else:  # C
            score += 120 if has_C else -100

        if typ in ("A", "B") and d == 1:
            score += 50
        if typ == "C" and d == 2:
            score += 50

        # Heuristic shift expectations derived from the examples.
        if typ == "A":
            expected_shift = zero_h <= 5
        elif typ == "B":
            expected_shift = (dr == -1 and zero_h >= 6)
        else:
            expected_shift = False

        if shift == expected_shift:
            score += 15
        else:
            score -= 15

        if has_C and typ == "C" and d == 2:
            score += 20

        # Slight preference for smaller candidate shapes on ties.
        score -= ncells // 10

        return score

    def make_candidate(typ, shift, r0, c0, val, Cr, Cc, dr, dc, d, zero_h):
        """Return (score, cells, center, erase_seed) or None."""

        if typ == "A":
            # A is only meaningful for distance 1 anchor in these examples.
            if d != 1:
                return None

            # Shift is vertical downward in the observed cases.
            Or = Cr + (1 if shift else 0)
            Oc = Cc

            cells = [
                (Or - 1, Oc),
                (Or + 1, Oc),
                (Or, Oc - 1),
                (Or, Oc + 1),
                (Or + dr, Oc + dc),
                (Or - dr, Oc - dc),
            ]
            center = (Or, Oc)
            erase_seed = ((r0, c0) not in cells)

        elif typ == "B":
            if d != 1:
                return None

            if not shift:
                cells = [
                    (Cr, Cc),
                    (Cr - dr, Cc),
                    (Cr, Cc - dc),
                    (r0, c0),
                ]
            else:
                # Shifted B observed for bottom corners of taller frames.
                cells = [
                    (r0, c0),
                    (Cr + 1, Cc),
                    (Cr + 2, Cc),
                    (Cr + 1, Cc - dc),
                ]

            center = None
            erase_seed = False

        elif typ == "C":
            if d < 2:
                return None

            # 2x2 block on the side opposite the seed, plus diagonal connectors.
            cells = [
                (Cr, Cc),
                (Cr - dr, Cc),
                (Cr, Cc - dc),
                (Cr - dr, Cc - dc),
            ]

            # For the examples d is exactly 2. If d were larger, add all
            # intermediate diagonal cells up to the seed.
            for t in range(1, d):
                cells.append((Cr + dr * t, Cc + dc * t))

            cells.append((r0, c0))
            center = None
            erase_seed = False

        else:
            return None

        if not center_ok(center, val):
            return None
        if not cells_ok(cells, val):
            return None

        score = score_candidate(typ, shift, d, dr, zero_h, len(cells))
        return score, cells, center, erase_seed

    # ------------------------------------------------------------------
    # Find the best completion for every isolated seed.
    # ------------------------------------------------------------------
    actions = []

    for r0, c0 in singles:
        val = arr[r0, c0]
        best = None
        best_score = -10**9
        best_len = 10**9

        # Search diagonal zero anchors at distances 1 and 2.
        for d in (1, 2):
            for dr, dc in diag_dirs:
                Cr = r0 - dr * d
                Cc = c0 - dc * d

                if not inb(Cr, Cc) or arr[Cr, Cc] != 0:
                    continue

                # The anchor should be a zero corner opening toward the seed.
                if not (inb(Cr + dr, Cc) and arr[Cr + dr, Cc] == 0 and
                        inb(Cr, Cc + dc) and arr[Cr, Cc + dc] == 0):
                    continue

                zlabel = zero_labels[Cr, Cc]
                zero_h = zero_height.get(zlabel, 1)

                # For distance 1, try A and B. For distance >= 2, try C.
                if d == 1:
                    for typ in ("A", "B"):
                        for shift in (False, True):
                            cand = make_candidate(
                                typ, shift, r0, c0, val,
                                Cr, Cc, dr, dc, d, zero_h
                            )
                            if cand is None:
                                continue

                            score, cells, center, erase_seed = cand
                            key = (score, -len(cells))
                            if (best is None or
                                key[0] > best_score or
                                (key[0] == best_score and key[1] > -best_len)):
                                best = (score, cells, center, erase_seed, (r0, c0), val)
                                best_score = score
                                best_len = len(cells)

                else:
                    cand = make_candidate(
                        "C", False, r0, c0, val,
                        Cr, Cc, dr, dc, d, zero_h
                    )
                    if cand is None:
                        continue

                    score, cells, center, erase_seed = cand
                    if best is None or score > best_score or (score == best_score and len(cells) < best_len):
                        best = (score, cells, center, erase_seed, (r0, c0), val)
                        best_score = score
                        best_len = len(cells)

        if best is not None:
            actions.append(best)

    # ------------------------------------------------------------------
    # Apply selected actions with simple conflict handling.
    # ------------------------------------------------------------------
    out = arr.copy()
    fill_map = {}
    zero_map = {}

    for score, cells, center, erase_seed, seed, val in actions:
        if center is not None:
            if center not in zero_map or score > zero_map[center][0]:
                zero_map[center] = (score, 0)

        for cell in cells:
            if (cell not in fill_map or
                    score > fill_map[cell][0] or
                    (score == fill_map[cell][0] and fill_map[cell][1] == val)):
                fill_map[cell] = (score, val)

    # Place zero centers first.
    for (r, c), (score, _) in zero_map.items():
        if inb(r, c) and not multi[r, c]:
            out[r, c] = 0

    # Place colored cells.
    for (r, c), (score, val) in fill_map.items():
        if not inb(r, c):
            continue

        # Never overwrite protected components of a different color.
        if multi[r, c] and arr[r, c] != val:
            continue

        # Respect higher-priority zero centers.
        if (r, c) in zero_map and zero_map[(r, c)][0] >= score:
            continue

        out[r, c] = val

    # Erase seeds that were only triggers, not part of the final stamp.
    for score, cells, center, erase_seed, seed, val in actions:
        if not erase_seed:
            continue

        r, c = seed
        if not inb(r, c):
            continue
        if multi[r, c]:
            continue
        if arr[r, c] != val:
            continue
        if (r, c) in fill_map or (r, c) in zero_map:
            continue

        out[r, c] = bg

    return out