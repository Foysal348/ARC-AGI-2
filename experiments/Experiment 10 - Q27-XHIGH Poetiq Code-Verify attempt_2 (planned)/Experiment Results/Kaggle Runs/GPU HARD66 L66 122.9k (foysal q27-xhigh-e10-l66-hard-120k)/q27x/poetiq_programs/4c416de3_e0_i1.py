import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Complete single-pixel colored markers near zero frames/corners.

    The examples contain:
      * a background color,
      * zero pixels forming frames / line corners,
      * colored non-background, non-zero markers.

    Multi-pixel colored components are treated as already-complete templates
    and are left unchanged. Single-pixel markers are completed according to
    the pattern style present in the input:

    A-style:
        3x3 pattern with a zero center, four orthogonal cells, and one full
        diagonal (the one containing the seed). The center zero is kept.

    B-style:
        For a seed diagonally adjacent to a zero corner, fill the zero corner
        plus the two orthogonal cells on the opposite side of the corner.

    C-style:
        For a seed farther away on a diagonal, fill the 2x2 square on the
        opposite side of the zero corner and the diagonal cells connecting
        the corner to the seed.

    A small special case handles a seed orthogonally adjacent to a horizontal
    zero line endpoint, as in Example 1 with the 6.
    """
    arr = np.array(grid, dtype=int)
    if arr.size == 0:
        return arr

    H, W = arr.shape

    # Background is the most frequent color.
    bg = int(np.bincount(arr.flatten()).argmax())

    # Colored pixels are non-background and non-zero.
    colored = (arr != bg) & (arr != 0)

    # ---------------------------------------------------------------
    # 8-connected component labeling for colored pixels.
    # Components with more than one pixel are protected templates.
    # ---------------------------------------------------------------
    labels = np.zeros((H, W), dtype=int)
    multi = np.zeros((H, W), dtype=bool)
    singles = []
    lab = 0

    def in_bounds(r, c):
        return 0 <= r < H and 0 <= c < W

    for i in range(H):
        for j in range(W):
            if colored[i, j] and labels[i, j] == 0:
                lab += 1
                stack = [(i, j)]
                labels[i, j] = lab
                cells = []

                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if in_bounds(nr, nc) and colored[nr, nc] and labels[nr, nc] == 0:
                                labels[nr, nc] = lab
                                stack.append((nr, nc))

                if len(cells) > 1:
                    for r, c in cells:
                        multi[r, c] = True
                else:
                    singles.append(cells[0])

    # ---------------------------------------------------------------
    # Detect which template styles are present.
    # ---------------------------------------------------------------
    def detect_a_template():
        """A-template: center 0, 4 orthogonals + one diagonal pair."""
        for r in range(1, H - 1):
            for c in range(1, W - 1):
                if arr[r, c] != 0:
                    continue

                orth = [
                    (r - 1, c),
                    (r + 1, c),
                    (r, c - 1),
                    (r, c + 1),
                ]
                vals = [arr[x, y] for x, y in orth]
                if len(set(vals)) != 1 or vals[0] in (0, bg):
                    continue
                color = vals[0]

                diag_main = (
                    arr[r - 1, c - 1] == color and
                    arr[r + 1, c + 1] == color and
                    arr[r - 1, c + 1] != color and
                    arr[r + 1, c - 1] != color
                )
                diag_anti = (
                    arr[r - 1, c + 1] == color and
                    arr[r + 1, c - 1] == color and
                    arr[r - 1, c - 1] != color and
                    arr[r + 1, c + 1] != color
                )

                if diag_main or diag_anti:
                    return True
        return False

    def detect_b_template():
        """B-template: four same-color cells in the B corner pattern."""
        for r in range(H):
            for c in range(W):
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        sr, sc = r + sx, c + sy
                        if not in_bounds(sr, sc):
                            continue
                        if arr[sr, sc] in (0, bg):
                            continue
                        color = arr[sr, sc]

                        # The corner cell in an input B-template is colored.
                        if arr[r, c] != color:
                            continue

                        a = (r - sx, c)
                        b = (r, c - sy)
                        if (
                            in_bounds(a[0], a[1]) and
                            in_bounds(b[0], b[1]) and
                            arr[a[0], a[1]] == color and
                            arr[b[0], b[1]] == color
                        ):
                            return True
        return False

    def detect_c_template():
        """C-template: larger component with a 2x2 block and diagonal tail."""
        unique_labels = np.unique(labels)
        for lab_id in unique_labels:
            if lab_id == 0:
                continue
            comp = labels == lab_id
            size = int(comp.sum())
            if size < 5:
                continue

            rows = np.any(comp, axis=1)
            cols = np.any(comp, axis=0)
            if int(rows.sum()) < 3 or int(cols.sum()) < 3:
                continue

            # Has a full 2x2 block of this component?
            for r in range(H - 1):
                for c in range(W - 1):
                    if comp[r, c] and comp[r, c + 1] and comp[r + 1, c] and comp[r + 1, c + 1]:
                        return True
        return False

    has_a = detect_a_template()
    has_b = detect_b_template()
    has_c = detect_c_template()

    out = arr.copy()

    # ---------------------------------------------------------------
    # Helper: safe cell setter.
    # Never overwrites protected multi-pixel templates.
    # Never overwrites initially colored pixels except the current seed.
    # ---------------------------------------------------------------
    def set_cell(r, c, value, seed):
        if not in_bounds(r, c):
            return
        if multi[r, c]:
            return
        if (
            arr[r, c] not in (0, bg) and
            not (r == seed[0] and c == seed[1])
        ):
            return
        out[r, c] = value

    # ---------------------------------------------------------------
    # Special orthogonal line-end case (the 6 in Example 1).
    # Seed is directly above a zero line, with a zero corner to its
    # lower-left and the line continuing to the right.
    # ---------------------------------------------------------------
    def orthogonal_special(r0, c0):
        r, c = r0, c0
        if r + 1 < H and c - 1 >= 0 and c + 1 < W:
            if (
                arr[r + 1, c] == 0 and
                arr[r + 1, c - 1] == 0 and
                arr[r, c - 1] == 0 and
                arr[r + 1, c + 1] == 0
            ):
                zr, zc = r + 1, c
                return [
                    (zr - 1, zc),   # N
                    (zr + 1, zc),   # S
                    (zr, zc - 1),   # W
                    (zr, zc + 1),   # E
                    (zr - 1, zc - 1),  # NW
                    (zr + 1, zc - 1),  # SW
                ]
        return None

    # ---------------------------------------------------------------
    # Find the best diagonal zero anchor for a single seed.
    # Prefer anchors where the two first steps toward the seed are zeros
    # (a zero corner opening toward the seed).
    # ---------------------------------------------------------------
    def find_diagonal_anchor(r0, c0):
        best = None
        best_score = -10**9
        max_d = min(H, W, 4)  # examples need distances 1 or 2

        for d in range(1, max_d + 1):
            corner_found = False
            for sx in (-1, 1):
                for sy in (-1, 1):
                    zr = r0 - sx * d
                    zc = c0 - sy * d

                    if not in_bounds(zr, zc) or arr[zr, zc] != 0:
                        continue

                    score = -d

                    # First step toward the seed in row and column.
                    toward_r = zr + sx
                    toward_c1 = zc
                    toward_r2 = zr
                    toward_c2 = zc + sy

                    tz1 = in_bounds(toward_r, toward_c1) and arr[toward_r, toward_c1] == 0
                    tz2 = in_bounds(toward_r2, toward_c2) and arr[toward_r2, toward_c2] == 0

                    if tz1 and tz2:
                        score += 100
                        corner_found = True
                    elif tz1 or tz2:
                        score += 10

                    if score > best_score:
                        best_score = score
                        best = (zr, zc, sx, sy, d)

            if corner_found and d <= 2:
                break

        if best is None or best_score < 10:
            return None
        return best

    # ---------------------------------------------------------------
    # Process each single-pixel marker.
    # ---------------------------------------------------------------
    for r0, c0 in singles:
        val = arr[r0, c0]

        # 1) Special orthogonal case.
        special_cells = orthogonal_special(r0, c0)
        if special_cells is not None:
            for r, c in special_cells:
                set_cell(r, c, val, (r0, c0))
            out[r0, c0] = val
            continue

        # 2) Diagonal corner / line anchor.
        anchor = find_diagonal_anchor(r0, c0)
        if anchor is None:
            out[r0, c0] = val
            continue

        zr, zc, sx, sy, d = anchor

        # Choose style.
        if has_a and d == 1:
            # A: plus shape + same diagonal, center remains zero.
            cells = [
                (zr - 1, zc),
                (zr + 1, zc),
                (zr, zc - 1),
                (zr, zc + 1),
                (zr - sx, zc - sy),
                (zr + sx, zc + sy),
            ]
        elif d == 1:
            # B: corner + two opposite orthogonal cells.
            cells = [
                (zr, zc),
                (zr - sx, zc),
                (zr, zc - sy),
                (zr + sx, zc + sy),
            ]
        else:
            # C: opposite 2x2 square + diagonal connector.
            cells = []

            # 2x2 square on the side opposite the seed.
            for dr in (0, -sx):
                for dc in (0, -sy):
                    cells.append((zr + dr, zc + dc))

            # Diagonal cells between anchor and seed.
            for t in range(1, d):
                cells.append((zr + sx * t, zc + sy * t))

            # Seed itself.
            cells.append((zr + sx * d, zc + sy * d))

        for r, c in cells:
            set_cell(r, c, val, (r0, c0))

        # Ensure the seed itself remains.
        out[r0, c0] = val

    return out