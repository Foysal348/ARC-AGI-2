import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input by extracting the frame template, selecting component
    masks, rotating non-frame masks to face the center, and packing them into
    the smallest square anchored by the template positions.
    """
    if grid.size == 0:
        return grid.copy()

    rows, cols = grid.shape

    # ------------------------------------------------------------
    # Connected components by exact color, 4-connected
    # ------------------------------------------------------------
    visited = np.zeros((rows, cols), dtype=bool)
    components = []

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0 and not visited[r, c]:
                color = int(grid[r, c])
                stack = [(r, c)]
                visited[r, c] = True
                pixels = []

                while stack:
                    rr, cc = stack.pop()
                    pixels.append((rr, cc))

                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if (
                            0 <= nr < rows
                            and 0 <= nc < cols
                            and not visited[nr, nc]
                            and grid[nr, nc] == color
                        ):
                            visited[nr, nc] = True
                            stack.append((nr, nc))

                components.append((color, pixels))

    if not components:
        return np.zeros_like(grid)

    # ------------------------------------------------------------
    # Find frame component: bounding box contains most distinct other colors
    # ------------------------------------------------------------
    best = None
    for color, pixels in components:
        min_r = min(p[0] for p in pixels)
        max_r = max(p[0] for p in pixels)
        min_c = min(p[1] for p in pixels)
        max_c = max(p[1] for p in pixels)

        sub = grid[min_r:max_r + 1, min_c:max_c + 1]
        other_colors = set(int(x) for x in sub[sub != 0].tolist())
        other_colors.discard(0)
        other_colors.discard(color)

        key = (
            len(other_colors),
            len(pixels),
            (max_r - min_r + 1) * (max_c - min_c + 1),
        )

        if best is None or key > best[0]:
            best = (key, color, min_r, max_r, min_c, max_c)

    if best is None or best[0][0] == 0:
        # Fallback for malformed input: return a copy.
        return grid.copy()

    _, frame_color, fr0, fr1, fc0, fc1 = best

    frame_mask = (grid[fr0:fr1 + 1, fc0:fc1 + 1] != 0).astype(bool)

    # ------------------------------------------------------------
    # Extract the small color template inside the frame
    # ------------------------------------------------------------
    frame_sub = grid[fr0:fr1 + 1, fc0:fc1 + 1].astype(int)
    nonframe_mask = (frame_sub != 0) & (frame_sub != frame_color)

    if np.any(nonframe_mask):
        row_has = np.any(nonframe_mask, axis=1)
        col_has = np.any(nonframe_mask, axis=0)

        pr0 = int(np.argmax(row_has))
        pr1 = int(len(row_has) - np.argmax(row_has[::-1]) - 1)
        pc0 = int(np.argmax(col_has))
        pc1 = int(len(col_has) - np.argmax(col_has[::-1]) - 1)

        pattern = grid[
            fr0 + pr0:fr0 + pr1 + 1,
            fc0 + pc0:fc0 + pc1 + 1
        ].astype(int)
    else:
        pattern = np.array([[frame_color]], dtype=int)

    pat_h, pat_w = pattern.shape

    # ------------------------------------------------------------
    # Collect other components, excluding anything inside frame bbox
    # ------------------------------------------------------------
    others = []

    for color, pixels in components:
        min_r = min(p[0] for p in pixels)
        max_r = max(p[0] for p in pixels)
        min_c = min(p[1] for p in pixels)
        max_c = max(p[1] for p in pixels)

        # If bboxes do not intersect, no need to check pixels.
        inside = False
        if max_r >= fr0 and min_r <= fr1 and max_c >= fc0 and min_c <= fc1:
            for r, c in pixels:
                if fr0 <= r <= fr1 and fc0 <= c <= fc1:
                    inside = True
                    break

        if inside:
            continue

        # Ignore tiny stray pixels outside the frame.
        if len(pixels) < 2:
            continue

        h = max_r - min_r + 1
        w = max_c - min_c + 1
        mask = np.zeros((h, w), dtype=bool)
        for r, c in pixels:
            mask[r - min_r, c - min_c] = True

        others.append(
            {
                "mask": mask,
                "area": len(pixels),
                "h": h,
                "w": w,
                "color": color,
            }
        )

    # Discard the largest other component, then sort the rest by area.
    others.sort(
        key=lambda x: (x["area"], x["h"] * x["w"], x["color"]),
        reverse=True,
    )
    if others:
        others = others[1:]

    # ------------------------------------------------------------
    # Assign masks to non-zero template cells in row-major order
    # ------------------------------------------------------------
    pat_cells = []
    for i in range(pat_h):
        for j in range(pat_w):
            if pattern[i, j] != 0:
                pat_cells.append((i, j, int(pattern[i, j])))

    all_masks = [frame_mask] + [o["mask"] for o in others]

    pos_info = []
    colors = []
    orig_masks = []

    for idx, (i, j, col) in enumerate(pat_cells):
        if idx >= len(all_masks):
            break
        pos_info.append((i, j))
        colors.append(col)
        orig_masks.append(all_masks[idx])

    if not pos_info:
        return np.zeros((1, 1), dtype=int)

    # Background color:
    # - 0 if template has a zero and all non-zero cells are filled.
    # - otherwise the last non-zero template color (e.g. bottom-right in 2x2).
    has_zero = bool(np.any(pattern == 0))
    unfilled_count = max(0, len(pat_cells) - len(all_masks))
    if has_zero and unfilled_count == 0:
        bg_color = 0
    else:
        bg_color = pat_cells[-1][2] if pat_cells else 0

    # ------------------------------------------------------------
    # Helpers for rotation and placement
    # ------------------------------------------------------------
    def unique_rotations(mask: np.ndarray):
        rots = [
            mask,
            np.rot90(mask, k=-1),
            np.rot90(mask, k=2),
            np.rot90(mask, k=1),
        ]
        unique = []
        for r in rots:
            r = r.copy()
            if not any(np.array_equal(r, u) for u in unique):
                unique.append(r)
        return unique

    def hole_score(mask: np.ndarray, dr: int, dc: int) -> float:
        """
        Higher means the empty cells inside the mask's bbox point more toward
        the direction (dr, dc) in local mask coordinates.
        """
        h, w = mask.shape
        holes = ~mask
        if not np.any(holes):
            return 0.0

        rr, cc = np.nonzero(holes)
        hr = float(rr.mean())
        hc = float(cc.mean())

        hr_norm = hr / (h - 1) if h > 1 else 0.5
        hc_norm = hc / (w - 1) if w > 1 else 0.5

        return float(dr) * hr_norm + float(dc) * hc_norm

    def local_cost_scalar(mask: np.ndarray, i: int, j: int) -> float:
        h, w = mask.shape
        top_bottom = (i == 0 or i == pat_h - 1)
        left_right = (j == 0 or j == pat_w - 1)

        cost = 0.0
        if top_bottom:
            cost += float(w)
        if left_right:
            cost += float(h)
        cost += 0.1 * (h + w)
        return cost

    def local_cost_key(mask: np.ndarray, i: int, j: int):
        h, w = mask.shape
        top_bottom = (i == 0 or i == pat_h - 1)
        left_right = (j == 0 or j == pat_w - 1)

        cost = 0.0
        if top_bottom:
            cost += float(w)
        if left_right:
            cost += float(h)
        cost += 0.1 * (h + w)

        return (
            cost,
            float(w) if top_bottom else 0.0,
            float(h) if left_right else 0.0,
            h,
            w,
        )

    def direction_to_center(i: int, j: int):
        if i < (pat_h - 1) / 2.0:
            dr = 1
        elif i > (pat_h - 1) / 2.0:
            dr = -1
        else:
            dr = 0

        if j < (pat_w - 1) / 2.0:
            dc = 1
        elif j > (pat_w - 1) / 2.0:
            dc = -1
        else:
            dc = 0

        return dr, dc

    # Choose rotation candidates for each assigned mask.
    cand_lists = []
    for idx, (i, j) in enumerate(pos_info):
        mask = orig_masks[idx]

        # The frame itself is kept in its original orientation.
        if idx == 0:
            cand_lists.append([mask])
            continue

        dr, dc = direction_to_center(i, j)
        rots = unique_rotations(mask)

        if np.all(mask):
            # No holes: orientation score is neutral.
            cands = rots
        else:
            scores = [hole_score(m, dr, dc) for m in rots]
            best_score = max(scores)
            cands = [
                m for m, s in zip(rots, scores)
                if s >= best_score - 1e-9
            ]

        cands.sort(key=lambda m: local_cost_key(m, i, j))

        # Keep the search small; local cost already breaks ties.
        if len(cands) > 3:
            cands = cands[:3]

        cand_lists.append(cands)

    def anchor(N: int, h: int, w: int, i: int, j: int):
        if i == 0 and j == 0:
            return 0, 0
        if i == 0 and j == pat_w - 1:
            return N - w, 0
        if i == pat_h - 1 and j == 0:
            return 0, N - h
        if i == pat_h - 1 and j == pat_w - 1:
            return N - w, N - h
        if i == 0:
            return (N - w) // 2, 0
        if i == pat_h - 1:
            return (N - w) // 2, N - h
        if j == 0:
            return 0, (N - h) // 2
        if j == pat_w - 1:
            return N - w, (N - h) // 2

        # Inner/non-perimeter fallback: center.
        return (N - w) // 2, (N - h) // 2

    def feasible(N: int, combo) -> bool:
        occ = np.zeros((N, N), dtype=bool)

        for k, mask in enumerate(combo):
            h, w = mask.shape
            i, j = pos_info[k]
            x, y = anchor(N, h, w, i, j)

            if x < 0 or y < 0 or x + w > N or y + h > N:
                return False

            if np.any(occ[y:y + h, x:x + w] & mask):
                return False

            occ[y:y + h, x:x + w] |= mask

        return True

    def compute_min_N(combo) -> int:
        if not combo:
            return 1

        lower = 1
        for mask in combo:
            h, w = mask.shape
            lower = max(lower, h, w)

        upper = max(
            lower,
            10,
            sum(max(m.shape[0], m.shape[1]) for m in combo) * 2 + 10,
        )

        # Expand upper bound until feasible, if needed.
        while not feasible(upper, combo):
            upper *= 2
            if upper > 2000:
                return upper

        low, high = lower, upper
        while low < high:
            mid = (low + high) // 2
            if feasible(mid, combo):
                high = mid
            else:
                low = mid + 1

        return low

    # ------------------------------------------------------------
    # Choose rotations: exhaustive if small, otherwise coordinate descent
    # ------------------------------------------------------------
    product_count = 1
    for cl in cand_lists:
        product_count *= len(cl)
        if product_count > 256:
            break

    if product_count <= 256:
        from itertools import product

        best_combo = None
        best_N = None
        best_cost = None

        for combo in product(*cand_lists):
            N = compute_min_N(combo)
            cost = sum(
                local_cost_scalar(m, pos_info[k][0], pos_info[k][1])
                for k, m in enumerate(combo)
            )

            if (
                best_N is None
                or N < best_N
                or (N == best_N and (best_cost is None or cost < best_cost))
            ):
                best_N = N
                best_combo = list(combo)
                best_cost = cost
    else:
        best_combo = [cl[0] for cl in cand_lists]
        best_N = compute_min_N(best_combo)
        best_cost = sum(
            local_cost_scalar(m, pos_info[k][0], pos_info[k][1])
            for k, m in enumerate(best_combo)
        )

        # Coordinate descent over rotation choices.
        for _ in range(3):
            improved = False
            for idx, cl in enumerate(cand_lists):
                if len(cl) < 2:
                    continue

                for m in cl:
                    if np.array_equal(m, best_combo[idx]):
                        continue

                    test_combo = list(best_combo)
                    test_combo[idx] = m

                    N = compute_min_N(test_combo)
                    cost = sum(
                        local_cost_scalar(
                            test_combo[k],
                            pos_info[k][0],
                            pos_info[k][1],
                        )
                        for k in range(len(test_combo))
                    )

                    if (
                        N < best_N
                        or (N == best_N and cost < best_cost)
                    ):
                        best_N = N
                        best_combo = test_combo
                        best_cost = cost
                        improved = True

            if not improved:
                break

    N = int(best_N)

    # ------------------------------------------------------------
    # Build final output
    # ------------------------------------------------------------
    out = np.full((N, N), bg_color, dtype=int)

    for k, mask in enumerate(best_combo):
        h, w = mask.shape
        i, j = pos_info[k]
        color = colors[k]
        x, y = anchor(N, h, w, i, j)

        if 0 <= x and 0 <= y and x + w <= N and y + h <= N:
            out[y:y + h, x:x + w][mask] = color

    return out