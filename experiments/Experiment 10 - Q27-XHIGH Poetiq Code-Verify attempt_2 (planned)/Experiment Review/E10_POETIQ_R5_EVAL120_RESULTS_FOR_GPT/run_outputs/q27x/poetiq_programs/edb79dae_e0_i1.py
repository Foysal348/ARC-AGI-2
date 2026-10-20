import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by:
      1. Cropping to the outer rectangular frame.
      2. Replacing interior solid blocks with recolored pattern templates
         found outside the frame.
      3. Determining recoloring from small directed 3x4 color keys of the form:
             a a b b
             a a a b
             a a b b
         meaning a -> b, followed transitively until a terminal color.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2:
        return grid.copy()

    H, W = grid.shape

    def connected_components(mask):
        """4-connected component labeling for a boolean mask."""
        h, w = mask.shape
        labels = np.zeros((h, w), dtype=int)
        next_label = 0

        for i in range(h):
            for j in range(w):
                if mask[i, j] and labels[i, j] == 0:
                    next_label += 1
                    labels[i, j] = next_label
                    stack = [(i, j)]

                    while stack:
                        r, c = stack.pop()
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = r + dr, c + dc
                            if (
                                0 <= nr < h
                                and 0 <= nc < w
                                and mask[nr, nc]
                                and labels[nr, nc] == 0
                            ):
                                labels[nr, nc] = next_label
                                stack.append((nr, nc))

        return labels, next_label

    def find_frame(g):
        """
        Find the largest rectangular border of a single color.
        Returns (color, r0, r1, c0, c1) or None.
        """
        best = None
        for color in np.unique(g):
            mask = g == color
            if not mask.any():
                continue

            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            if len(rows) == 0 or len(cols) == 0:
                continue

            r0, r1 = int(rows[0]), int(rows[-1])
            c0, c1 = int(cols[0]), int(cols[-1])

            # Need at least a 3x3 rectangle to have an interior.
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue

            # Check that the perimeter is completely this color.
            if not (
                mask[r0, c0:c1 + 1].all()
                and mask[r1, c0:c1 + 1].all()
                and mask[r0:r1 + 1, c0].all()
                and mask[r0:r1 + 1, c1].all()
            ):
                continue

            score = (r1 - r0 + 1) * (c1 - c0 + 1)
            if best is None or score > best[0]:
                best = (score, color, r0, r1, c0, c1)

        if best is None:
            return None
        return best[1], best[2], best[3], best[4], best[5]

    def find_directed_color_pairs(g, bg, frame_color, r0, r1, c0, c1):
        """
        Find all directed color pairs from 3x4 keys outside the frame:
            a a b b
            a a a b
            a a b b
        Returns dict: a -> b.
        """
        mapping = {}
        outside = np.ones((H, W), dtype=bool)
        outside[r0:r1 + 1, c0:c1 + 1] = False

        for r in range(H - 2):
            for c in range(W - 3):
                if not outside[r:r + 3, c:c + 4].all():
                    continue

                a = g[r, c]
                if a == bg or a == frame_color:
                    continue

                b = g[r, c + 2]
                if b == a or b == bg or b == frame_color:
                    continue

                if (
                    g[r, c + 1] == a
                    and g[r, c + 3] == b
                    and g[r + 1, c] == a
                    and g[r + 1, c + 1] == a
                    and g[r + 1, c + 2] == a
                    and g[r + 1, c + 3] == b
                    and g[r + 2, c] == a
                    and g[r + 2, c + 1] == a
                    and g[r + 2, c + 2] == b
                    and g[r + 2, c + 3] == b
                ):
                    mapping[int(a)] = int(b)

        return mapping

    def fit_mask(mask, h, w):
        """
        Crop/pad a boolean mask to shape (h, w), centered.
        This is a fallback for slight size mismatches.
        """
        out = np.zeros((h, w), dtype=bool)
        if h <= 0 or w <= 0:
            return out

        ch, cw = mask.shape

        if ch > h:
            src_r0 = (ch - h) // 2
            src_h = h
            out_r0 = 0
        else:
            src_r0 = 0
            src_h = ch
            out_r0 = (h - ch) // 2

        if cw > w:
            src_c0 = (cw - w) // 2
            src_w = w
            out_c0 = 0
        else:
            src_c0 = 0
            src_w = cw
            out_c0 = (w - cw) // 2

        out[out_r0:out_r0 + src_h, out_c0:out_c0 + src_w] = (
            mask[src_r0:src_r0 + src_h, src_c0:src_c0 + src_w]
        )
        return out

    def choose_template(candidates, h, w):
        """
        Choose the best template mask for a source color and fit it to (h, w).
        """
        if not candidates:
            return np.ones((h, w), dtype=bool)

        best_mask = None
        best_score = None

        for m in candidates:
            ch, cw = m.shape
            area = int(np.sum(m))
            solid_penalty = 1 if area == ch * cw else 0

            # Prefer:
            #   1. close bounding-box size
            #   2. non-solid pattern over solid rectangle
            #   3. larger area
            #   4. larger minimum dimension
            score = (
                abs(ch - h) + abs(cw - w),
                solid_penalty,
                -area,
                -min(ch, cw),
            )

            if best_score is None or score < best_score:
                best_score = score
                best_mask = m

        return fit_mask(best_mask, h, w)

    # --- Main transform ---

    frame_info = find_frame(grid)
    if frame_info is None:
        return grid.copy()

    frame_color, r0, r1, c0, c1 = frame_info
    frame = grid[r0:r1 + 1, c0:c1 + 1].copy()
    fh, fw = frame.shape

    # Determine interior background.
    bg = 0
    if fh > 2 and fw > 2:
        interior = frame[1:-1, 1:-1]
        vals, counts = np.unique(interior, return_counts=True)
        if vals.size:
            # Prefer the most common color that is not the frame color.
            order = np.argsort(-counts)
            for idx in order:
                if vals[idx] != frame_color:
                    bg = int(vals[idx])
                    break
            else:
                bg = int(vals[0])

    # Start output as an empty frame.
    out = np.full((fh, fw), bg, dtype=grid.dtype)
    if fh > 0:
        out[0, :] = frame_color
        out[-1, :] = frame_color
    if fw > 0:
        out[:, 0] = frame_color
        out[:, -1] = frame_color

    if fh < 3 or fw < 3:
        return out

    # Directed color-change mapping, followed transitively.
    mapping = find_directed_color_pairs(grid, bg, frame_color, r0, r1, c0, c1)

    def final_target(color):
        cur = int(color)
        seen = set()
        for _ in range(12):  # Safety guard against accidental cycles.
            if cur in mapping and cur not in seen:
                seen.add(cur)
                cur = mapping[cur]
            else:
                break
        return cur

    # Find interior objects (solid blocks) by connected components.
    interior = frame[1:-1, 1:-1]
    interior_objects = []

    for color in np.unique(interior):
        if color == bg or color == frame_color:
            continue

        mask = interior == color
        labels, n = connected_components(mask)

        for lab in range(1, n + 1):
            comp = labels == lab
            rows = np.where(comp.any(axis=1))[0]
            cols = np.where(comp.any(axis=0))[0]
            if len(rows) == 0 or len(cols) == 0:
                continue

            ri0, ri1 = int(rows[0]), int(rows[-1])
            ci0, ci1 = int(cols[0]), int(cols[-1])

            # Convert to frame coordinates (interior starts at offset 1).
            interior_objects.append(
                (
                    int(color),
                    ri0 + 1,
                    ri1 + 1,
                    ci0 + 1,
                    ci1 + 1,
                    ri1 - ri0 + 1,
                    ci1 - ci0 + 1,
                )
            )

    # Collect possible template components outside the frame.
    outside = np.ones((H, W), dtype=bool)
    outside[r0:r1 + 1, c0:c1 + 1] = False
    candidates_by_color = {}

    if outside.any():
        for color in np.unique(grid[outside]):
            if color == bg or color == frame_color:
                continue

            mask = (grid == color) & outside
            labels, n = connected_components(mask)

            cands = []
            for lab in range(1, n + 1):
                comp = labels == lab
                rows = np.where(comp.any(axis=1))[0]
                cols = np.where(comp.any(axis=0))[0]
                if len(rows) == 0 or len(cols) == 0:
                    continue

                rr0, rr1 = int(rows[0]), int(rows[-1])
                cc0, cc1 = int(cols[0]), int(cols[-1])
                sub = comp[rr0:rr1 + 1, cc0:cc1 + 1]
                if sub.size > 0:
                    cands.append(sub)

            if cands:
                candidates_by_color[int(color)] = cands

    # Replace each interior object by its template in the final target color.
    for color, fr0, fr1, fc0, fc1, h, w in interior_objects:
        target = final_target(color)
        template = choose_template(candidates_by_color.get(color, []), h, w)

        # Clear the slot first, then paint the template.
        out[fr0:fr1 + 1, fc0:fc1 + 1] = bg
        slot = out[fr0:fr1 + 1, fc0:fc1 + 1]
        slot[template] = target

    return out