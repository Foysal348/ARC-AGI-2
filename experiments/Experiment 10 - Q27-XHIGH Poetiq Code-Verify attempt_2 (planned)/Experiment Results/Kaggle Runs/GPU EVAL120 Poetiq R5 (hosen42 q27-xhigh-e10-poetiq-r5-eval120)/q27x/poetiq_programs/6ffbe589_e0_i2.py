import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Recover the compact assembled object.

    The rule inferred from the examples:
      1. Ignore small stray pixels that lie outside the main square object.
      2. The main object has three square layers (outer/middle/inner), each of a
         different non-zero color, centered in the grid.
      3. For each color layer, the input layer has been transformed by one of the
         8 dihedral symmetries (identity, horizontal flip, vertical flip, 180
         rotation, 90 CW, 90 CCW, transpose, anti-transpose) inside its own
         square bounding box.
      4. The correct symmetry for a color layer is the one that makes the layer
         maximally consistent with a compact nested ARC object.  In practice,
         this is the symmetry that maximizes overlap with the other already
         recovered layers' expected nested square positions while minimizing
         unnatural color intersections.

    The implementation below identifies the main square, extracts each color's
    square layer, tries all 8 symmetries for that layer, and keeps the symmetry
    with the best heuristic score.  The final output is the tight bounding box
    of the reconstructed main object.
    """
    if grid is None or grid.size == 0:
        return grid.copy()

    H, W = grid.shape
    nz = grid > 0

    # ------------------------------------------------------------------
    # Connected components on the dilated non-zero mask to find the main body.
    # ------------------------------------------------------------------
    def dilate(mask: np.ndarray, iters: int = 1) -> np.ndarray:
        out = mask.copy()
        for _ in range(iters):
            out = (
                out.copy()
                | np.roll(out, 1, axis=0)
                | np.roll(out, -1, axis=0)
                | np.roll(out, 1, axis=1)
                | np.roll(out, -1, axis=1)
            )
        return out

    dilated = dilate(nz, 1)

    seen = np.zeros_like(dilated, dtype=bool)
    comps = []

    for r in range(H):
        for c in range(W):
            if not dilated[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cur = []
            while stack:
                y, x = stack.pop()
                cur.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and dilated[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            comps.append(cur)

    if not comps:
        return np.zeros_like(grid)

    # Choose component with the largest original non-zero pixel count.
    best_comp = None
    best_score = -1
    for comp in comps:
        ys = [p[0] for p in comp]
        xs = [p[1] for p in comp]
        r0, r1 = min(ys), max(ys)
        c0, c1 = min(xs), max(xs)
        score = int(np.sum(nz[r0:r1 + 1, c0:c1 + 1]))
        if score > best_score:
            best_score = score
            best_comp = (r0, r1, c0, c1)

    r0, r1, c0, c1 = best_comp
    main = grid[r0:r1 + 1, c0:c1 + 1].copy()

    # The main object is usually square. If not, pad/crop to a square using
    # the larger dimension and later crop tightly.
    mh, mw = main.shape
    side = max(mh, mw)
    sq = np.zeros((side, side), dtype=main.dtype)
    y0 = (mh - side) // 2
    x0 = (mw - side) // 2
    sq[y0:y0 + mh, x0:x0 + mw] = main
    main = sq

    # ------------------------------------------------------------------
    # Extract color layers.
    # ------------------------------------------------------------------
    colors = [int(v) for v in np.unique(main) if v != 0]
    if not colors:
        return np.zeros_like(main)

    layers = {}
    for col in colors:
        mask = (main == col)
        ys, xs = np.where(mask)
        if ys.size == 0:
            layers[col] = (None, None, None, None)
            continue
        rA, rB = int(ys.min()), int(ys.max())
        cA, cB = int(xs.min()), int(xs.max())
        layers[col] = (col, mask, rA, rB, cA, cB)

    # ------------------------------------------------------------------
    # Dihedral symmetries.  Each is applied to the square bounding box of a
    # color layer, then written back to the same square bbox in the output.
    # ------------------------------------------------------------------
    def apply_sym(arr: np.ndarray, sym: int) -> np.ndarray:
        # sym:
        # 0: identity
        # 1: H flip
        # 2: V flip
        # 3: 180
        # 4: 90 CW
        # 5: 90 CCW
        # 6: transpose
        # 7: anti-transpose
        if sym == 0:
            return arr.copy()
        if sym == 1:
            return arr[:, ::-1].copy()
        if sym == 2:
            return arr[::-1, :].copy()
        if sym == 3:
            return arr[::-1, ::-1].copy()
        if sym == 4:
            return np.rot90(arr, k=-1).copy()
        if sym == 5:
            return np.rot90(arr, k=1).copy()
        if sym == 6:
            return arr.T.copy()
        if sym == 7:
            return arr.T[:, ::-1].copy()
        raise ValueError("bad symmetry")

    def bbox_of_mask(mask: np.ndarray):
        ys, xs = np.where(mask)
        if ys.size == 0:
            return None
        return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())

    # Heuristic score for a candidate reconstructed grid:
    # - Prefer non-overlapping colors (ARC layers usually do not overlap).
    # - Prefer each non-outer layer's bbox centered inside the outer layer.
    # - Prefer layers whose bbox side lengths are close to the nested pattern
    #   (outer largest, inner smaller), with square bboxes.
    def score_candidate(out_grid: np.ndarray, color_order):
        score = 0.0

        # No overlap score
        overlaps = 0
        nonzero_masks = []
        for col in color_order:
            m = out_grid == col
            if m.any():
                nonzero_masks.append(m)
        for i in range(len(nonzero_masks)):
            for j in range(i + 1, len(nonzero_masks)):
                overlaps += int(np.sum(nonzero_masks[i] & nonzero_masks[j]))
        score -= 100.0 * overlaps

        # Squareness and nesting score
        boxes = []
        for col in color_order:
            m = out_grid == col
            b = bbox_of_mask(m)
            if b is not None:
                boxes.append((col, b))

        if not boxes:
            return -1e9

        # Sort by bbox area descending.
        boxes.sort(key=lambda t: (
            (t[1][1] - t[1][0] + 1) * (t[1][3] - t[1][2] + 1)
        ), reverse=True)

        for col, (a, b, c, d) in boxes:
            h = b - a + 1
            w = d - c + 1
            # Prefer square boxes.
            score -= 20.0 * abs(h - w)

            # Prefer boxes roughly centered in the grid.
            ch = (a + b) / 2.0
            cw = (c + d) / 2.0
            gh, gw = out_grid.shape
            score -= 5.0 * abs(ch - (gh - 1) / 2.0)
            score -= 5.0 * abs(cw - (gw - 1) / 2.0)

        # Nesting: for each pair, smaller should be inside larger.
        n = len(boxes)
        for i in range(n):
            for j in range(i + 1, n):
                _, (a1, b1, c1, d1) = boxes[i]
                _, (a2, b2, c2, d2) = boxes[j]
                area1 = (b1 - a1 + 1) * (d1 - c1 + 1)
                area2 = (b2 - a2 + 1) * (d2 - c2 + 1)
                outer, inner = boxes[i], boxes[j]
                if area1 < area2:
                    outer, inner = boxes[j], boxes[i]
                oa, ob, oc, od = outer[1]
                ia, ib, ic, idd = inner[1]
                if ia >= oa and ib <= ob and ic >= oc and idd <= od:
                    score += 10.0
                else:
                    score -= 30.0

        return score

    # ------------------------------------------------------------------
    # Try all symmetry combinations.  There are at most 3 colors, so 8^3 = 512
    # candidates, which is tiny.
    # ------------------------------------------------------------------
    sym_choices = [0, 1, 2, 3, 4, 5, 6, 7]
    best_out = None
    best_score = -1e100

    def recurse(idx: int, current_sym: dict, current_out: np.ndarray):
        nonlocal best_out, best_score

        if idx == len(colors):
            s = score_candidate(current_out, colors)
            if s > best_score:
                best_score = s
                best_out = current_out.copy()
            return

        col = colors[idx]
        info = layers[col]
        if info[1] is None:
            recurse(idx + 1, current_sym, current_out)
            return

        col, mask, rA, rB, cA, cB = info
        h = rB - rA + 1
        w = cB - cA + 1

        # For a square layer, use its square bbox.  If non-square, embed in a
        # square to allow rotations.
        side = max(h, w)
        box = np.zeros((side, side), dtype=main.dtype)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        sub = (mask[rA:rB + 1, cA:cB + 1]).astype(main.dtype)
        box[y0:y0 + h, x0:x0 + w] = sub

        for sym in sym_choices:
            transformed_box = apply_sym(box, sym)

            # Place back into the same overall location as the original bbox.
            # If the layer was embedded in a larger square, place the whole
            # embedded square back into the original bbox's top-left square
            # position as closely as possible.
            trial = current_out.copy()
            base_r = rA + y0
            base_c = cA + x0

            # Clip to grid if needed.
            rr0 = max(0, base_r)
            cc0 = max(0, base_c)
            rr1 = min(trial.shape[0], base_r + side)
            cc1 = min(trial.shape[1], base_c + side)

            if rr1 <= rr0 or cc1 <= cc0:
                recurse(idx + 1, current_sym, trial)
                continue

            # Only write non-zero pixels.
            region = trial[rr0:rr1, cc0:cc1]
            box_region = transformed_box[rr0 - base_r:rr1 - base_r,
                                         cc0 - base_c:cc1 - base_c]
            region[box_region > 0] = col

            recurse(idx + 1, current_sym, trial)

    initial = np.zeros_like(main)
    # Start with zeros; each color is written in the order discovered.
    # To reduce symmetry ambiguity, process larger layers first.
    colors_sorted = sorted(colors, key=lambda c: int(np.sum(main == c)), reverse=True)
    # But the recursion uses `colors`; replace it for this search.
    colors = colors_sorted

    recurse(0, {}, initial)

    if best_out is None:
        best_out = main

    # Tight crop to non-zero pixels.
    if not best_out.any():
        return np.zeros_like(best_out)

    ys, xs = np.where(best_out > 0)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    out = best_out[y0:y1 + 1, x0:x1 + 1]

    return out