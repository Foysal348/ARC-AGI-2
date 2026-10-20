import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the framed block grid according to the color templates and
    3x4 mapping keys found outside the frame.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    if H == 0 or W == 0:
        return g.copy()

    def components(mask):
        """Return list of 4-connected components as lists of (r, c)."""
        labels = np.zeros(mask.shape, dtype=int)
        comps = []
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] and labels[i, j] == 0:
                    lab = len(comps) + 1
                    stack = [(i, j)]
                    labels[i, j] = lab
                    pixels = []
                    while stack:
                        x, y = stack.pop()
                        pixels.append((x, y))
                        if x > 0 and mask[x - 1, y] and labels[x - 1, y] == 0:
                            labels[x - 1, y] = lab
                            stack.append((x - 1, y))
                        if x + 1 < mask.shape[0] and mask[x + 1, y] and labels[x + 1, y] == 0:
                            labels[x + 1, y] = lab
                            stack.append((x + 1, y))
                        if y > 0 and mask[x, y - 1] and labels[x, y - 1] == 0:
                            labels[x, y - 1] = lab
                            stack.append((x, y - 1))
                        if y + 1 < mask.shape[1] and mask[x, y + 1] and labels[x, y + 1] == 0:
                            labels[x, y + 1] = lab
                            stack.append((x, y + 1))
                    comps.append(pixels)
        return comps

    def is_connected(mask):
        """Check whether all True cells in a small 2D mask are 4-connected."""
        if not mask.any():
            return False
        seen = np.zeros(mask.shape, dtype=bool)
        i0, j0 = np.argwhere(mask)[0]
        stack = [(int(i0), int(j0))]
        seen[i0, j0] = True
        count = 1
        total = int(mask.sum())
        while stack:
            x, y = stack.pop()
            if x > 0 and mask[x - 1, y] and not seen[x - 1, y]:
                seen[x - 1, y] = True
                count += 1
                stack.append((x - 1, y))
            if x + 1 < mask.shape[0] and mask[x + 1, y] and not seen[x + 1, y]:
                seen[x + 1, y] = True
                count += 1
                stack.append((x + 1, y))
            if y > 0 and mask[x, y - 1] and not seen[x, y - 1]:
                seen[x, y - 1] = True
                count += 1
                stack.append((x, y - 1))
            if y + 1 < mask.shape[1] and mask[x, y + 1] and not seen[x, y + 1]:
                seen[x, y + 1] = True
                count += 1
                stack.append((x, y + 1))
        return count == total

    def find_frame():
        """
        Find the largest hollow rectangular frame. Returns
        (frame_color, r0, c0, r1, c1) or None.
        """
        best = None

        # Component-based detection is more robust to stray same-colored pixels.
        for color in np.unique(g):
            mask = (g == color)
            if mask.sum() < 10:
                continue
            for pixels in components(mask):
                if len(pixels) < 10:
                    continue
                rs = [p[0] for p in pixels]
                cs = [p[1] for p in pixels]
                r0, r1 = min(rs), max(rs)
                c0, c1 = min(cs), max(cs)
                h, w = r1 - r0 + 1, c1 - c0 + 1
                if h < 3 or w < 3:
                    continue
                if not (
                    np.all(g[r0, c0:c1 + 1] == color) and
                    np.all(g[r1, c0:c1 + 1] == color) and
                    np.all(g[r0:r1 + 1, c0] == color) and
                    np.all(g[r0:r1 + 1, c1] == color)
                ):
                    continue
                if r1 > r0 + 1 and c1 > c0 + 1:
                    if np.all(g[r0 + 1:r1, c0 + 1:c1] == color):
                        continue
                area = h * w
                if best is None or area > best[0]:
                    best = (area, color, r0, c0, r1, c1)

        if best is not None:
            return best[1], best[2], best[3], best[4], best[5]

        # Fallback: bounding box of all pixels of a color.
        for color in np.unique(g):
            ys, xs = np.where(g == color)
            if ys.size == 0:
                continue
            r0, r1 = ys.min(), ys.max()
            c0, c1 = xs.min(), xs.max()
            h, w = r1 - r0 + 1, c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            if not (
                np.all(g[r0, c0:c1 + 1] == color) and
                np.all(g[r1, c0:c1 + 1] == color) and
                np.all(g[r0:r1 + 1, c0] == color) and
                np.all(g[r0:r1 + 1, c1] == color)
            ):
                continue
            if r1 > r0 + 1 and c1 > c0 + 1:
                if np.all(g[r0 + 1:r1, c0 + 1:c1] == color):
                    continue
            area = h * w
            if best is None or area > best[0]:
                best = (area, color, r0, c0, r1, c1)

        if best is not None:
            return best[1], best[2], best[3], best[4], best[5]
        return None

    frame_info = find_frame()
    if frame_info is None:
        return g.copy()

    frame_color, r0, c0, r1, c1 = frame_info

    # Background inside the frame: use the immediate interior border.
    if r1 > r0 + 1 and c1 > c0 + 1:
        perim = np.concatenate((
            g[r0 + 1, c0 + 1:c1],
            g[r1 - 1, c0 + 1:c1],
            g[r0 + 1:r1 - 1, c0 + 1],
            g[r0 + 1:r1 - 1, c1 - 1]
        ))
        if perim.size > 0:
            vals, counts = np.unique(perim, return_counts=True)
            bg = int(vals[np.argmax(counts)])
        else:
            bg = 0
    else:
        bg = 0

    inside_frame = np.zeros((H, W), dtype=bool)
    inside_frame[r0:r1 + 1, c0:c1 + 1] = True

    # Detect solid square blocks inside the frame.
    inside = np.zeros((H, W), dtype=bool)
    inside[r0 + 1:r1, c0 + 1:c1] = True
    nonbg = (g != bg) & inside

    blocks = []
    sizes = []
    for pixels in components(nonbg):
        if not pixels:
            continue
        rs = [p[0] for p in pixels]
        cs = [p[1] for p in pixels]
        rr0, rr1 = min(rs), max(rs)
        cc0, cc1 = min(cs), max(cs)
        h, w = rr1 - rr0 + 1, cc1 - cc0 + 1
        if h != w:
            continue
        sub = g[rr0:rr1 + 1, cc0:cc1 + 1]
        if sub.size == 0:
            continue
        if np.all(sub == sub[0, 0]):
            color = int(sub[0, 0])
            if color != bg:
                blocks.append((rr0, cc0, h, color))
                sizes.append(h)

    if not blocks:
        out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        out[0, :] = frame_color
        out[-1, :] = frame_color
        out[:, 0] = frame_color
        out[:, -1] = frame_color
        return out

    size_vals, size_counts = np.unique(sizes, return_counts=True)
    K = int(size_vals[np.argmax(size_counts)])
    blocks = [b for b in blocks if b[2] == K]
    if not blocks:
        blocks = [b for b in blocks if b[2] == K] or blocks[:1]

    source_colors = {b[3] for b in blocks}

    # Detect 3x4 mapping keys:
    # source color appears 7 times, target appears 5 times in fixed positions.
    def find_keys():
        expected_target = np.zeros((3, 4), dtype=bool)
        expected_target[0, 2] = True
        expected_target[0, 3] = True
        expected_target[1, 3] = True
        expected_target[2, 2] = True
        expected_target[2, 3] = True

        mapping = {}
        key_rects = []

        for r in range(H - 2):
            for c in range(W - 3):
                if np.any(inside_frame[r:r + 3, c:c + 4]):
                    continue
                sub = g[r:r + 3, c:c + 4]
                vals, counts = np.unique(sub, return_counts=True)
                if len(vals) != 2:
                    continue
                if sorted(counts.tolist()) != [5, 7]:
                    continue
                src = int(vals[counts == 7][0])
                tgt = int(vals[counts == 5][0])
                if src not in source_colors:
                    continue
                if not np.array_equal(sub == tgt, expected_target):
                    continue

                # Record the key region so templates are not confused with keys.
                key_rects.append((r, c, 3, 4))
                if tgt != bg and src not in mapping:
                    mapping[src] = tgt

        return mapping, key_rects

    mapping, key_rects = find_keys()

    # Find the best KxK template for each source color outside the frame.
    def find_template(src_color):
        best_score = None
        best_mask = None
        solid_pos = None

        if K <= 0:
            return None

        for r in range(H - K + 1):
            for c in range(W - K + 1):
                if np.any(inside_frame[r:r + K, c:c + K]):
                    continue

                overlap = False
                for kr, kc, kh, kw in key_rects:
                    if not (r + K <= kr or kr + kh <= r or c + K <= kc or kc + kw <= c):
                        overlap = True
                        break
                if overlap:
                    continue

                sub = g[r:r + K, c:c + K]
                vals = np.unique(sub)

                if len(vals) == 1:
                    if int(vals[0]) == int(src_color):
                        if solid_pos is None or (r, c) < solid_pos:
                            solid_pos = (r, c)
                    continue

                if len(vals) != 2:
                    continue
                if int(src_color) not in vals or int(bg) not in vals:
                    continue

                mask = (sub == src_color)
                count = int(np.sum(mask))
                if count == 0:
                    continue

                if count == K * K:
                    if solid_pos is None or (r, c) < solid_pos:
                        solid_pos = (r, c)
                    continue

                connected = is_connected(mask)
                # Prefer connected templates, then densest, then top-left.
                score = (1 if connected else 0, count, -r, -c)
                if best_score is None or score > best_score:
                    best_score = score
                    best_mask = mask.copy()

        if best_mask is not None:
            return best_mask
        if solid_pos is not None:
            return np.ones((K, K), dtype=bool)
        return None

    templates = {}
    for src in source_colors:
        templates[src] = find_template(src)

    # Build output frame.
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    out[0, :] = frame_color
    out[-1, :] = frame_color
    out[:, 0] = frame_color
    out[:, -1] = frame_color

    for br, bc, kb, color in blocks:
        rr, cc = br - r0, bc - c0
        target = int(mapping.get(color, color))
        mask = templates.get(color)

        region = out[rr:rr + kb, cc:cc + kb]
        if mask is not None and mask.shape == (kb, kb):
            region[mask] = target
            region[~mask] = bg
        elif mask is not None and mask.size == kb * kb:
            m = mask.reshape(kb, kb)
            region[m] = target
            region[~m] = bg
        else:
            region[:] = target

    return out