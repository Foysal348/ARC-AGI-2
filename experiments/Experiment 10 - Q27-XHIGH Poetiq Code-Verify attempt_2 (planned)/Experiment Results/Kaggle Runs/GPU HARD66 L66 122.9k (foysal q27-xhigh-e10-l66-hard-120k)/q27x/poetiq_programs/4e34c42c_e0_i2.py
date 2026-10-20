import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the set of valid small scenes and concatenate them into one row.

    The input is a background grid containing several small non-background
    components.  The transformation:

    1. Uses the most frequent color as background.
    2. Keeps only meaningful pieces:
       - 5-row-tall non-background connected components (the "big" objects),
       - valid 3x3 dots (uniform non-background perimeter, different center),
       - isolated horizontal line segments of length at least 6.
    3. Removes obvious artifacts:
       - components touching the outer border,
       - components containing color 7 (used as noise in this task family),
       - duplicate exact objects,
       - any object that appears as an exact subpattern of a larger object,
       - a special 5x? border-column artifact when the same column appears
         in more than one 5-row object and the wider object is redundant.
    4. Orders the remaining objects by a cyclic color order:
       - the shortest / narrowest object is the anchor,
       - other objects follow by first non-background color in cyclic
         numeric order,
       - duplicate anchor colors are placed at the end.
    5. Concatenates the objects vertically centered in one row.
       Adjacent objects overlap by one column when their touching columns
       are compatible and share at least one non-background pixel.
    """
    if grid.size == 0:
        return grid.copy()

    grid = np.array(grid, dtype=int)
    H, W = grid.shape

    vals, cnts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(cnts)])

    if not np.any(grid != bg):
        return np.full((1, 1), bg, dtype=int)

    # ------------------------------------------------------------------
    # Small utilities
    # ------------------------------------------------------------------
    def component_bboxes(mask):
        """Return bounding boxes of 4-connected components in mask."""
        Hm, Wm = mask.shape
        seen = np.zeros_like(mask, dtype=bool)
        boxes = []

        for i in range(Hm):
            for j in range(Wm):
                if not mask[i, j] or seen[i, j]:
                    continue

                stack = [(i, j)]
                seen[i, j] = True
                r0 = r1 = i
                c0 = c1 = j

                while stack:
                    r, c = stack.pop()
                    r0 = min(r0, r)
                    r1 = max(r1, r)
                    c0 = min(c0, c)
                    c1 = max(c1, c)

                    for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                        if 0 <= nr < Hm and 0 <= nc < Wm and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))

                boxes.append((r0, r1, c0, c1))

        return boxes

    def trim(arr):
        """Crop away all-background rows/columns."""
        mask = arr != bg
        if not np.any(mask):
            return np.empty((0, 0), dtype=int)
        ys, xs = np.where(mask)
        return arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()

    def canon(arr):
        """Canonical pattern with background replaced by -1."""
        return np.where(arr == bg, -1, arr)

    def flat_key(arr):
        return (arr.shape[0], arr.shape[1], tuple(canon(arr).ravel().tolist()))

    def is_subpattern(small, big):
        """True if small's exact colored pattern appears in big."""
        hs, ws = small.shape
        hb, wb = big.shape
        if hb < hs or wb < ws:
            return False

        S = canon(small)
        B = canon(big)

        for y in range(hb - hs + 1):
            for x in range(wb - ws + 1):
                if np.array_equal(S, B[y:y + hs, x:x + ws]):
                    return True
        return False

    def first_nonbg(arr):
        for v in arr.ravel():
            if v != bg:
                return int(v)
        return bg

    objects = []
    seen_keys = set()

    def add_object(arr):
        """Trim, filter, de-duplicate and store an object."""
        arr = trim(arr)
        if arr.size == 0:
            return

        h, w = arr.shape
        if h not in (1, 3, 5):
            return
        if w < 1:
            return
        if int(np.sum(arr != bg)) < 3:
            return
        if bg != 7 and np.any(arr == 7):
            return

        key = flat_key(arr)
        if key in seen_keys:
            return
        seen_keys.add(key)
        objects.append(arr)

    # ------------------------------------------------------------------
    # 1) Big 5-row connected components
    # ------------------------------------------------------------------
    nonbg = grid != bg
    boxes = component_bboxes(nonbg)

    big_crops = []
    for r0, r1, c0, c1 in boxes:
        h = r1 - r0 + 1
        if h != 5:
            continue

        # Ignore edge-touching pieces; they are almost always distractors.
        if r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1:
            continue

        crop = grid[r0:r1 + 1, c0:c1 + 1].copy()
        if bg != 7 and np.any(crop == 7):
            continue
        if int(np.sum(crop != bg)) < 5:
            continue

        big_crops.append(crop)

    # Prune a repeated short border column of the form:
    # background, c, c, c, background
    # from the wider of the containing 5-row objects.
    def pruneable_col(col):
        if len(col) != 5:
            return None
        nonbg_idx = [i for i, v in enumerate(col) if v != bg]
        if len(nonbg_idx) < 2:
            return None
        if 0 in nonbg_idx or 4 in nonbg_idx:
            return None
        if max(nonbg_idx) - min(nonbg_idx) + 1 != len(nonbg_idx):
            return None
        vals = {int(col[i]) for i in nonbg_idx}
        if len(vals) != 1:
            return None
        return tuple(int(v) for v in col)

    col_occurrences = {}
    for idx, crop in enumerate(big_crops):
        left = pruneable_col(crop[:, 0])
        if left is not None:
            col_occurrences.setdefault(left, []).append((idx, "left"))
        right = pruneable_col(crop[:, -1])
        if right is not None:
            col_occurrences.setdefault(right, []).append((idx, "right"))

    prune = [set() for _ in big_crops]

    for pat, occ in col_occurrences.items():
        if len(occ) < 2:
            continue

        widths = [big_crops[idx].shape[1] for idx, _ in occ]
        min_w = min(widths)

        for idx, side in occ:
            if big_crops[idx].shape[1] > min_w:
                prune[idx].add(side)

    for idx, crop in enumerate(big_crops):
        new_crop = crop
        if "left" in prune[idx]:
            new_crop = new_crop[:, 1:]
        if "right" in prune[idx]:
            new_crop = new_crop[:, :-1]
        add_object(new_crop)

    # ------------------------------------------------------------------
    # 2) Small objects: valid 3x3 dots and isolated horizontal lines
    # ------------------------------------------------------------------
    marked = np.zeros((H, W), dtype=bool)

    # 3x3 dots: uniform non-background perimeter, different non-background center.
    for r in range(H - 2):
        for c in range(W - 2):
            win = grid[r:r + 3, c:c + 3]
            a = int(win[0, 0])
            if a == bg or (bg != 7 and a == 7):
                continue

            perimeter = [
                int(win[0, 0]), int(win[0, 1]), int(win[0, 2]),
                int(win[1, 0]),                 int(win[1, 2]),
                int(win[2, 0]), int(win[2, 1]), int(win[2, 2]),
            ]
            if any(v != a for v in perimeter):
                continue

            b = int(win[1, 1])
            if b == bg or b == a or (bg != 7 and b == 7):
                continue

            marked[r:r + 3, c:c + 3] = True

    # Isolated horizontal lines: same color, length >= 6, background above/below.
    for r in range(1, H - 1):
        c = 0
        while c < W:
            v = int(grid[r, c])
            if v == bg or (bg != 7 and v == 7):
                c += 1
                continue

            run = 1
            while c + run < W and int(grid[r, c + run]) == v:
                run += 1

            if run >= 6:
                ok = True
                for cc in range(c, c + run):
                    if grid[r - 1, cc] != bg or grid[r + 1, cc] != bg:
                        ok = False
                        break
                if ok:
                    marked[r, c:c + run] = True
                    c += run
                    continue

            c += 1

    for r0, r1, c0, c1 in component_bboxes(marked):
        # Ignore marked pieces that touch the outer border.
        if r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1:
            continue
        add_object(grid[r0:r1 + 1, c0:c1 + 1].copy())

    if not objects:
        return np.full((1, 1), bg, dtype=int)

    # ------------------------------------------------------------------
    # 3) Remove objects that are exact subpatterns of larger objects.
    #    This removes duplicate dots/lines that are already contained
    #    in a larger reconstructed object.
    # ------------------------------------------------------------------
    canon_objects = [canon(arr) for arr in objects]
    areas = [int(np.sum(arr != bg)) for arr in objects]

    keep = [True] * len(objects)
    for i in range(len(objects)):
        if not keep[i]:
            continue
        hi, wi = objects[i].shape
        for j in range(len(objects)):
            if i == j or not keep[j]:
                continue
            hj, wj = objects[j].shape
            if hj < hi or wj < wi:
                continue
            if areas[j] <= areas[i]:
                continue
            if is_subpattern(objects[i], objects[j]):
                keep[i] = False
                break

    objects = [arr for arr, k in zip(objects, keep) if k]

    if not objects:
        return np.full((1, 1), bg, dtype=int)

    # ------------------------------------------------------------------
    # 4) Order the objects.
    #    Anchor: shortest height, then narrowest width, then smallest area.
    #    Others: cyclic order of first non-background color from anchor hue.
    #    Objects sharing the anchor hue but not the anchor go to the end.
    # ------------------------------------------------------------------
    infos = []
    for idx, arr in enumerate(objects):
        h, w = arr.shape
        area = int(np.sum(arr != bg))
        hue = first_nonbg(arr)
        flat = tuple(canon(arr).ravel().tolist())
        infos.append((idx, arr, h, w, area, hue, flat))

    base = min(infos, key=lambda x: (x[2], x[3], x[4], x[5]))
    base_hue = base[5]
    base_idx = base[0]

    def sort_key(info):
        idx, arr, h, w, area, hue, flat = info
        primary = (hue - base_hue) % 10
        if primary == 0 and idx != base_idx:
            primary = 100
        return (primary, w, h, area, flat)

    infos.sort(key=sort_key)

    # ------------------------------------------------------------------
    # 5) Concatenate with compatible one-column overlaps.
    # ------------------------------------------------------------------
    max_h = max(info[2] for info in infos)

    def pad_v(arr):
        h, w = arr.shape
        top = (max_h - h) // 2
        out = np.full((max_h, w), bg, dtype=int)
        out[top:top + h, :] = arr
        return out

    def compatible(left_col, right_col):
        """
        Compatible if no conflicting non-background pixels,
        and at least one actual non-background match.
        """
        matched = False
        for a, b in zip(left_col, right_col):
            an = a != bg
            bn = b != bg
            if an and bn:
                if a != b:
                    return False
                matched = True
        return matched

    padded = [pad_v(info[1]) for info in infos]

    out = padded[0].copy()
    last_col = out[:, -1].copy()

    for p in padded[1:]:
        w = p.shape[1]
        left_col = p[:, 0]

        overlap = 1 if compatible(last_col, left_col) else 0
        new_w = out.shape[1] + w - overlap
        new_out = np.full((max_h, new_w), bg, dtype=int)
        new_out[:, :out.shape[1]] = out

        start_col = out.shape[1] - overlap
        for j in range(w):
            col = p[:, j]
            rows = np.where(col != bg)[0]
            if rows.size:
                new_out[rows, start_col + j] = col[rows]

        out = new_out
        last_col = out[:, -1].copy()

    return out