import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct intended orthogonal color paths.

    The input contains several colored 1-pixel-wide orthogonal paths.  Some
    true path cells are missing (0), some are marked with hint colors, and
    some path cells are overwritten by another path.  The output restores the
    full intended paths:
      * fill short orthogonal gaps between same-colored segments,
      * extend short gaps when the two visible ends are aligned,
      * remove isolated short decoy strokes,
      * at intersections, the smaller numeric path color is shown.

    The implementation is deliberately conservative: it only adds pixels when
    there is strong collinear evidence from the same color or hint pixels.
    """
    g = np.array(grid, dtype=int, copy=True)
    if g.ndim != 2 or g.size == 0:
        return g

    H, W = g.shape

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def is_boundary(r, c):
        return r == 0 or r == H - 1 or c == 0 or c == W - 1

    def rect(r, c, radius=2):
        r0 = max(0, r - radius)
        r1 = min(H, r + radius + 1)
        c0 = max(0, c - radius)
        c1 = min(W, c + radius + 1)
        return r0, r1, c0, c1

    def color_counts(mask):
        return int(np.count_nonzero(mask))

    def longest_orthogonal_run(mask):
        best = 0
        for r in range(mask.shape[0]):
            run = 0
            for x in mask[r]:
                if x:
                    run += 1
                    if run > best:
                        best = run
                else:
                    run = 0
        for c in range(mask.shape[1]):
            run = 0
            for x in mask[:, c]:
                if x:
                    run += 1
                    if run > best:
                        best = run
                else:
                    run = 0
        return best

    def horizontal_component_size(mask, r, c):
        if not (0 <= r < H and 0 <= c < W) or not mask[r, c]:
            return 0
        left = c
        while left > 0 and mask[r, left - 1]:
            left -= 1
        right = c
        while right + 1 < W and mask[r, right + 1]:
            right += 1
        return right - left + 1

    def vertical_component_size(mask, r, c):
        if not (0 <= r < H and 0 <= c < W) or not mask[r, c]:
            return 0
        up = r
        while up > 0 and mask[up - 1, c]:
            up -= 1
        down = r
        while down + 1 < H and mask[down + 1, c]:
            down += 1
        return down - up + 1

    def component_size(mask, r, c):
        if not (0 <= r < H and 0 <= c < W) or not mask[r, c]:
            return 0
        seen = np.zeros_like(mask, dtype=bool)
        stack = [(r, c)]
        seen[r, c] = True
        size = 0
        while stack:
            rr, cc = stack.pop()
            size += 1
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        return size

    def all_mask_components(mask):
        visited = np.zeros_like(mask, dtype=bool)
        comps = []
        for r in range(H):
            for c in range(W):
                if mask[r, c] and not visited[r, c]:
                    seen = np.zeros_like(mask, dtype=bool)
                    stack = [(r, c)]
                    seen[r, c] = True
                    cells = []
                    while stack:
                        rr, cc = stack.pop()
                        cells.append((rr, cc))
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True
                                stack.append((nr, nc))
                    visited |= seen
                    comps.append((len(cells), cells))
        return comps

    # ------------------------------------------------------------------
    # Identify main path colors and hint colors.
    # Main colors are the most frequent non-zero colors with a visible line.
    # ------------------------------------------------------------------
    vals, counts = np.unique(g, return_counts=True)
    nonzero = {int(v): int(cnt) for v, cnt in zip(vals, counts) if int(v) != 0}
    if not nonzero:
        return np.zeros_like(g, dtype=int)

    candidate_main = []
    for c, cnt in sorted(nonzero.items(), key=lambda x: (-x[1], x[0])):
        mask = (g == c)
        if cnt >= 4 and longest_orthogonal_run(mask) >= 2:
            candidate_main.append(c)

    # Usually 2-4 main colors.  Keep top frequent line-like colors.
    main_colors = candidate_main[:4]
    if not main_colors:
        main_colors = sorted(nonzero, key=lambda x: -nonzero[x])[:2]

    main_set = set(main_colors)

    # Hint colors are low-frequency non-main colors, typically 2 or 4.
    hint_colors = [c for c in nonzero if c not in main_set]

    # ------------------------------------------------------------------
    # For each main color, build a support mask including likely hint pixels.
    # A hint pixel is assigned to a main color if that color is close to it
    # and the hint lies on a line with two visible same-color supports.
    # ------------------------------------------------------------------
    support = {}
    for c in main_colors:
        E = (g == c)
        M = np.zeros_like(E, dtype=bool)

        for hr in range(H):
            for hc in range(W):
                if g[hr, hc] not in hint_colors:
                    continue

                r0, r1, c0, c1 = rect(hr, hc, radius=2)
                if not np.any(E[r0:r1, c0:c1]):
                    continue

                # Prefer horizontal assignment if there are E pixels to both
                # sides on the same row.
                left = E[hr, :hc].any() if hc > 0 else False
                right = E[hr, hc + 1:].any() if hc + 1 < W else False
                up = E[:hr, hc].any() if hr > 0 else False
                down = E[hr + 1:, hc].any() if hr + 1 < H else False

                if left and right:
                    M[hr, hc] = True
                elif up and down:
                    M[hr, hc] = True
                elif (left or right) and np.any(E[hr, max(0, hc - 3):hc + 4]):
                    M[hr, hc] = True
                elif (up or down) and np.any(E[max(0, hr - 3):hr + 4, hc]):
                    M[hr, hc] = True

        support[c] = E | M

    # ------------------------------------------------------------------
    # Add pixels to main colors in increasing order.  Smaller colors are final
    # at intersections, so draw larger colors first and smaller colors later.
    # ------------------------------------------------------------------
    out = np.zeros((H, W), dtype=int)

    def set_color(r, c, color):
        if 0 <= r < H and 0 <= c < W:
            if out[r, c] == 0 or color < out[r, c]:
                out[r, c] = color

    def segment_clear(row, c1, c2):
        # True if the open interval can be filled without overriding an already
        # drawn smaller-color pixel.
        if c2 <= c1:
            return True
        seg = out[row, c1 + 1:c2]
        return not np.any((seg > 0) & (seg < row if False else 0))  # always true; placeholder
        # We will not actually use this; color priority is handled by set_color.

    def fill_horizontal(row, c1, c2):
        a, b = sorted((c1, c2))
        for c in range(a, b + 1):
            set_color(row, c, row_color)

    def fill_vertical(col, r1, r2):
        a, b = sorted((r1, r2))
        for r in range(a, b + 1):
            set_color(r, col, row_color)

    for row_color in sorted(main_colors, reverse=True):
        E = (g == row_color)
        M = support[row_color]
        S = M.copy()

        changed = True
        max_iter = 4 * max(H, W)
        it = 0
        while changed and it < max_iter:
            changed = False
            it += 1

            # Fill horizontal gaps of length up to 4.
            for r in range(H):
                cols = np.where(S[r])[0]
                if len(cols) < 2:
                    continue
                for a, b in zip(cols, cols[1:]):
                    gap = b - a - 1
                    if 0 < gap <= 4:
                        # Require endpoints to look like line ends: not a
                        # suspicious isolated pixel.
                        left_ok = horizontal_component_size(S, r, a) >= 2 or is_boundary(r, a) or np.any(S[r, max(0, a - 2):a + 1])
                        right_ok = horizontal_component_size(S, r, b) >= 2 or is_boundary(r, b) or np.any(S[r, b:min(W, b + 3)])
                        if left_ok and right_ok:
                            for c in range(a, b + 1):
                                if not S[r, c]:
                                    S[r, c] = True
                                    changed = True
                                    set_color(r, c, row_color)

            # Fill vertical gaps of length up to 4.
            for c in range(W):
                rows = np.where(S[:, c])[0]
                if len(rows) < 2:
                    continue
                for a, b in zip(rows, rows[1:]):
                    gap = b - a - 1
                    if 0 < gap <= 4:
                        up_ok = vertical_component_size(S, a, c) >= 2 or is_boundary(a, c) or np.any(S[max(0, a - 2):a + 1, c])
                        down_ok = vertical_component_size(S, b, c) >= 2 or is_boundary(b, c) or np.any(S[b:min(H, b + 3), c])
                        if up_ok and down_ok:
                            for r in range(a, b + 1):
                                if not S[r, c]:
                                    S[r, c] = True
                                    changed = True
                                    set_color(r, c, row_color)

        # Extend straight runs that stop just before a hint or a same-color
        # endpoint with a small aligned continuation.  This handles corners
        # where one arm is partially hinted.
        for r in range(H):
            for c in range(W):
                if not S[r, c]:
                    continue
                # Horizontal right extension through hint/zero.
                if c + 1 < W and not S[r, c + 1]:
                    # Need another support within 3 cells to the right.
                    if c + 4 < W and S[r, c + 4]:
                        # Check no strong perpendicular color blocks it.
                        ok = True
                        for cc in range(c + 1, c + 4):
                            val = g[r, cc]
                            if val != 0 and val != row_color and val not in hint_colors and val in main_set:
                                ok = False
                                break
                        if ok:
                            for cc in range(c + 1, c + 4):
                                if not S[r, cc]:
                                    S[r, cc] = True
                                    set_color(r, cc, row_color)

                # Horizontal left extension.
                if c - 1 >= 0 and not S[r, c - 1]:
                    if c - 4 >= 0 and S[r, c - 4]:
                        ok = True
                        for cc in range(c - 4, c):
                            val = g[r, cc]
                            if val != 0 and val != row_color and val not in hint_colors and val in main_set:
                                ok = False
                                break
                        if ok:
                            for cc in range(c - 4, c):
                                if not S[r, cc]:
                                    S[r, cc] = True
                                    set_color(r, cc, row_color)

                # Vertical down extension.
                if r + 1 < H and not S[r + 1, c]:
                    if r + 4 < H and S[r + 4, c]:
                        ok = True
                        for rr in range(r + 1, r + 4):
                            val = g[rr, c]
                            if val != 0 and val != row_color and val not in hint_colors and val in main_set:
                                ok = False
                                break
                        if ok:
                            for rr in range(r + 1, r + 4):
                                if not S[rr, c]:
                                    S[rr, c] = True
                                    set_color(rr, c, row_color)

                # Vertical up extension.
                if r - 1 >= 0 and not S[r - 1, c]:
                    if r - 4 >= 0 and S[r - 4, c]:
                        ok = True
                        for rr in range(r - 4, r):
                            val = g[rr, c]
                            if val != 0 and val != row_color and val not in hint_colors and val in main_set:
                                ok = False
                                break
                        if ok:
                            for rr in range(r - 4, r):
                                if not S[rr, c]:
                                    S[rr, c] = True
                                    set_color(rr, c, row_color)

        # Remove tiny decoy components of this color if a much larger
        # component exists and the tiny one does not cross another path
        # usefully.  This is conservative: only remove 1-2 pixel components.
        comps = all_mask_components(S)
        if len(comps) >= 2:
            comps.sort(key=lambda x: -x[0])
            largest = comps[0][0]
            for size, cells in comps[1:]:
                if size >= 1 and size <= 2 and largest >= 6:
                    # Keep boundary-connected small components.
                    if any(is_boundary(r, c) for r, c in cells):
                        continue
                    # Keep if adjacent to a hint-colored pixel that is plausibly
                    # part of a line.
                    keep = False
                    for r, c in cells:
                        r0, r1, c0, c1 = rect(r, c, radius=1)
                        if np.any((g[r0:r1, c0:c1] == row_color) | (g[r0:r1, c0:c1].isin(hint_colors))):
                            keep = True
                            break
                    if not keep:
                        for r, c in cells:
                            S[r, c] = False
                            if out[r, c] == row_color:
                                out[r, c] = 0

    # ------------------------------------------------------------------
    # Final pass: preserve original main-color pixels that were not explained
    # but are part of a visible line.  This avoids deleting legitimate short
    # segments that the gap-filler did not use.
    # ------------------------------------------------------------------
    for c in main_colors:
        mask = (g == c)
        for r in range(H):
            for cc in range(W):
                if mask[r, cc] and out[r, cc] == 0:
                    # Keep if it has same-color support horizontally or
                    # vertically within distance 2.
                    r0, r1, c0, c1 = rect(r, cc, radius=2)
                    same = mask[r0:r1, c0:c1]
                    same[r, cc] = False
                    if np.any(same):
                        # Do not place a pixel if a smaller color already
                        # correctly covers it.
                        if out[r, cc] == 0:
                            out[r, cc] = c
        # If a visible line is completely erased except endpoints, fill again
        # directly from original color only.
        for r in range(H):
            cols = np.where(mask[r])[0]
            if len(cols) >= 2:
                for a, b in zip(cols, cols[1:]):
                    if b - a <= 4:
                        for cc in range(a, b + 1):
                            if out[r, cc] == 0:
                                out[r, cc] = c
        for cc in range(W):
            rows = np.where(mask[:, cc])[0]
            if len(rows) >= 2:
                for a, b in zip(rows, rows[1:]):
                    if b - a <= 4:
                        for r in range(a, b + 1):
                            if out[r, cc] == 0:
                                out[r, cc] = c

    # ------------------------------------------------------------------
    # Ensure hint colors are not left in output.
    # ------------------------------------------------------------------
    for r in range(H):
        for c in range(W):
            if g[r, c] in hint_colors and out[r, c] == 0:
                out[r, c] = 0

    return out