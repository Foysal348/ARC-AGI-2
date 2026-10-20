import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform ARC grids containing 0-frames, 9-blocks, and 5/7 fuel cells.

    For each rectangular 0-frame that contains at least one 5:
      * Find the adjacent solid 9-block, if any.
      * If another 9-block exists farther in the same direction, the frame and
        adjacent 9 move toward it until the 9s collide.
      * Otherwise, the frame moves into the adjacent 9, eroding it from the
        side facing the frame.
      * Movement distance is capped by the number of 5 cells inside the frame.
      * That many 5 cells are changed to 7 in a side-specific scan order:
          - up:    top-to-bottom, left-to-right
          - right: right-to-left, top-to-bottom
          - down:  bottom-to-top, right-to-left
          - left:  left-to-right, bottom-to-top
    """
    g = np.array(grid, dtype=int)
    if g.ndim != 2:
        return g

    H, W = g.shape
    if H == 0 or W == 0:
        return g

    # Background is normally 6, but determine it robustly.
    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    if bg == 0 and np.any(g == 6):
        bg = 6

    # ------------------------------------------------------------
    # Prefix-sum helpers
    # ------------------------------------------------------------
    def make_ps(mask):
        ps = np.zeros((H + 1, W + 1), dtype=np.int32)
        if H and W:
            ps[1:, 1:] = mask.astype(np.int32).cumsum(axis=0).cumsum(axis=1)
        return ps

    def ps_sum(ps, r0, c0, r1, c1):
        """Inclusive-ish rectangle sum using half-open [r0:r1, c0:c1]."""
        if r0 >= r1 or c0 >= c1:
            return 0
        return int(ps[r1, c1] - ps[r0, c1] - ps[r1, c0] + ps[r0, c0])

    ps5 = make_ps(g == 5)
    ps9 = make_ps(g == 9)

    # Exclude background inside frames, unless background is 0 (should not happen).
    exclude_bg = (bg != 0)
    psbg = make_ps(g == bg) if exclude_bg else None

    # Row/column zero prefixes for fast perimeter checks.
    row_zero = np.zeros((H, W + 1), dtype=np.int32)
    for r in range(H):
        row_zero[r, 1:] = np.cumsum((g[r] == 0).astype(np.int32))

    col_zero = np.zeros((W, H + 1), dtype=np.int32)
    for c in range(W):
        col_zero[c, 1:] = np.cumsum((g[:, c] == 0).astype(np.int32))

    # ------------------------------------------------------------
    # Find candidate 0-frames:
    #   - rectangular perimeter all 0
    #   - interior contains at least one 5
    #   - interior does not contain background or 9
    # Interior zeros are allowed.
    # ------------------------------------------------------------
    frames = []
    seen_frames = set()

    for r0 in range(H - 2):
        for r1 in range(r0 + 2, H):
            h = r1 - r0 + 1
            for c0 in range(W - 2):
                if g[r0, c0] != 0 or g[r1, c0] != 0:
                    continue

                for c1 in range(c0 + 2, W):
                    if g[r0, c1] != 0 or g[r1, c1] != 0:
                        continue

                    w = c1 - c0 + 1

                    # Top/bottom borders.
                    if row_zero[r0, c1 + 1] - row_zero[r0, c0] != w:
                        continue
                    if row_zero[r1, c1 + 1] - row_zero[r1, c0] != w:
                        continue

                    # Left/right borders, excluding corners.
                    if col_zero[c0, r1] - col_zero[c0, r0 + 1] != h - 2:
                        continue
                    if col_zero[c1, r1] - col_zero[c1, r0 + 1] != h - 2:
                        continue

                    # Interior checks.
                    i5 = ps_sum(ps5, r0 + 1, c0 + 1, r1, c1)
                    if i5 <= 0:
                        continue

                    if exclude_bg:
                        ibg = ps_sum(psbg, r0 + 1, c0 + 1, r1, c1)
                        if ibg != 0:
                            continue

                    i9 = ps_sum(ps9, r0 + 1, c0 + 1, r1, c1)
                    if i9 != 0:
                        continue

                    bbox = (r0, r1, c0, c1)
                    if bbox not in seen_frames:
                        seen_frames.add(bbox)
                        frames.append({
                            "bbox": bbox,
                            "interior": g[r0 + 1:r1, c0 + 1:c1].copy(),
                            "fives": int(i5)
                        })

    # ------------------------------------------------------------
    # Find solid rectangular 9-components.
    # ------------------------------------------------------------
    nine_comps = []
    labels = np.zeros((H, W), dtype=int)
    DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
    cid = 0

    for r in range(H):
        for c in range(W):
            if g[r, c] != 9 or labels[r, c] != 0:
                continue

            cid += 1
            labels[r, c] = cid
            stack = [(r, c)]
            rmin = rmax = r
            cmin = cmax = c
            count = 0

            while stack:
                rr, cc = stack.pop()
                count += 1

                if rr < rmin:
                    rmin = rr
                if rr > rmax:
                    rmax = rr
                if cc < cmin:
                    cmin = cc
                if cc > cmax:
                    cmax = cc

                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < H and 0 <= nc < W
                        and g[nr, nc] == 9
                        and labels[nr, nc] == 0
                    ):
                        labels[nr, nc] = cid
                        stack.append((nr, nc))

            area = (rmax - rmin + 1) * (cmax - cmin + 1)
            if count == area:
                nine_comps.append({
                    "id": cid,
                    "bbox": (rmin, rmax, cmin, cmax)
                })

    # ------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------
    def overlap_rows(a0, a1, b0, b1):
        return max(0, min(a1, b1) - max(a0, b0) + 1)

    def overlap_cols(a0, a1, b0, b1):
        return max(0, min(a1, b1) - max(a0, b0) + 1)

    def shift_bbox(bbox, d, side):
        r0, r1, c0, c1 = bbox
        if side == "left":
            return (r0, r1, c0 - d, c1 - d)
        if side == "right":
            return (r0, r1, c0 + d, c1 + d)
        if side == "up":
            return (r0 - d, r1 - d, c0, c1)
        if side == "down":
            return (r0 + d, r1 + d, c0, c1)
        return bbox

    def erode_bbox(bbox, d, side):
        """
        Erode a 9-rectangle from the side facing the frame.
        """
        r0, r1, c0, c1 = bbox
        if side == "left":
            # 9 is to the left of frame: erode its right edge.
            return (r0, r1, c0, c1 - d)
        if side == "right":
            # 9 is to the right of frame: erode its left edge.
            return (r0, r1, c0 + d, c1)
        if side == "up":
            # 9 is above frame: erode its bottom edge.
            return (r0, r1 - d, c0, c1)
        if side == "down":
            # 9 is below frame: erode its top edge.
            return (r0 + d, r1, c0, c1)
        return bbox

    def max_move_for_frame(bbox, side):
        r0, r1, c0, c1 = bbox
        if side == "up":
            return r0
        if side == "down":
            return H - 1 - r1
        if side == "left":
            return c0
        if side == "right":
            return W - 1 - c1
        return 0

    def find_far(near, side):
        """
        Find the nearest farther 9-rectangle in the same direction as the
        frame movement, requiring overlap with the near 9.
        """
        nr0, nr1, nc0, nc1 = near["bbox"]
        best = None
        best_dist = None

        for n in nine_comps:
            if n is near:
                continue

            gr0, gr1, gc0, gc1 = n["bbox"]

            if side == "left":
                if gc1 < nc0:
                    ov = overlap_rows(gr0, gr1, nr0, nr1)
                    if ov > 0:
                        dist = nc0 - gc1
                        if dist > 0 and (best_dist is None or dist < best_dist):
                            best = n
                            best_dist = dist

            elif side == "right":
                if gc0 > nc1:
                    ov = overlap_rows(gr0, gr1, nr0, nr1)
                    if ov > 0:
                        dist = gc0 - nc1
                        if dist > 0 and (best_dist is None or dist < best_dist):
                            best = n
                            best_dist = dist

            elif side == "up":
                if gr1 < nr0:
                    ov = overlap_cols(gc0, gc1, nc0, nc1)
                    if ov > 0:
                        dist = nr0 - gr1
                        if dist > 0 and (best_dist is None or dist < best_dist):
                            best = n
                            best_dist = dist

            elif side == "down":
                if gr0 > nr1:
                    ov = overlap_cols(gc0, gc1, nc0, nc1)
                    if ov > 0:
                        dist = gr0 - nr1
                        if dist > 0 and (best_dist is None or dist < best_dist):
                            best = n
                            best_dist = dist

        return best

    def desired_distance(near, far, side):
        nr0, nr1, nc0, nc1 = near["bbox"]

        if far is not None:
            fr0, fr1, fc0, fc1 = far["bbox"]
            if side == "left":
                return nc0 - fc1
            if side == "right":
                return fc0 - nc1
            if side == "up":
                return nr0 - fr1
            if side == "down":
                return fr0 - nr1

        # No farther 9: erode until the 9 rectangle is one cell thick.
        if side in ("left", "right"):
            thickness = nc1 - nc0 + 1
        else:
            thickness = nr1 - nr0 + 1
        return max(0, thickness - 1)

    def convert_interior(interior, side, d):
        """
        Convert exactly d 5-cells to 7.

        Equivalent to rotating the interior so the side adjacent to the 9 is
        on top, then scanning row-major.
        """
        out = interior.copy()
        if d <= 0:
            return out

        ih, iw = out.shape
        added = 0

        if side == "left":
            for c in range(iw):
                for r in range(ih - 1, -1, -1):
                    if out[r, c] == 5:
                        out[r, c] = 7
                        added += 1
                        if added == d:
                            return out

        elif side == "right":
            for c in range(iw - 1, -1, -1):
                for r in range(ih):
                    if out[r, c] == 5:
                        out[r, c] = 7
                        added += 1
                        if added == d:
                            return out

        elif side == "up":
            for r in range(ih):
                for c in range(iw):
                    if out[r, c] == 5:
                        out[r, c] = 7
                        added += 1
                        if added == d:
                            return out

        elif side == "down":
            for r in range(ih - 1, -1, -1):
                for c in range(iw - 1, -1, -1):
                    if out[r, c] == 5:
                        out[r, c] = 7
                        added += 1
                        if added == d:
                            return out

        return out

    # ------------------------------------------------------------
    # Compute operations for each frame.
    # ------------------------------------------------------------
    frame_ops = []
    nine_actions = {}

    for frame in frames:
        five_count = frame["fives"]
        if five_count <= 0:
            continue

        fr0, fr1, fc0, fc1 = frame["bbox"]

        # Collect all adjacent 9 blocks.
        candidates = []
        for n in nine_comps:
            nr0, nr1, nc0, nc1 = n["bbox"]

            if nr1 == fr0 - 1:
                ov = overlap_cols(nc0, nc1, fc0, fc1)
                if ov > 0:
                    candidates.append(("up", n, ov))

            if nc0 == fc1 + 1:
                ov = overlap_rows(nr0, nr1, fr0, fr1)
                if ov > 0:
                    candidates.append(("right", n, ov))

            if nr0 == fr1 + 1:
                ov = overlap_cols(nc0, nc1, fc0, fc1)
                if ov > 0:
                    candidates.append(("down", n, ov))

            if nc1 == fc0 - 1:
                ov = overlap_rows(nr0, nr1, fr0, fr1)
                if ov > 0:
                    candidates.append(("left", n, ov))

        if not candidates:
            continue

        # Choose the active side. Prefer sides that actually allow movement,
        # then larger overlap, then larger possible movement.
        best = None
        best_ov = -1
        best_d = -1

        for side, near, ov in candidates:
            far = find_far(near, side)
            desired = desired_distance(near, far, side)
            d = min(desired, five_count, max_move_for_frame(frame["bbox"], side))
            d = max(0, int(d))

            if d <= 0:
                continue

            if (
                best is None
                or ov > best_ov
                or (ov == best_ov and d > best_d)
            ):
                best = (side, near, far, desired, d)
                best_ov = ov
                best_d = d

        if best is None:
            continue

        side, near, far, desired, d = best
        if d <= 0:
            continue

        new_frame_bbox = shift_bbox(frame["bbox"], d, side)
        new_interior = convert_interior(frame["interior"], side, d)

        frame_ops.append({
            "bbox": frame["bbox"],
            "new_bbox": new_frame_bbox,
            "interior": new_interior
        })

        # Decide what happens to the near 9.
        if far is not None:
            if desired > 0 and d == desired:
                action = "erase"
                new_nine_bbox = None
            else:
                action = "shift"
                new_nine_bbox = shift_bbox(near["bbox"], d, side)
        else:
            action = "erode"
            new_nine_bbox = erode_bbox(near["bbox"], d, side)

        info = {
            "action": action,
            "new_bbox": new_nine_bbox,
            "d": d
        }

        old = nine_actions.get(near["id"])
        if old is None or d > old.get("d", -1):
            nine_actions[near["id"]] = info

    # ------------------------------------------------------------
    # Build the output grid.
    # ------------------------------------------------------------
    out = g.copy()

    # Clear old frame locations.
    for op in frame_ops:
        r0, r1, c0, c1 = op["bbox"]
        out[r0:r1 + 1, c0:c1 + 1] = bg

    # Clear old 9 locations that will move/erode/disappear.
    for comp in nine_comps:
        info = nine_actions.get(comp["id"])
        if info is not None and info["action"] in ("shift", "erode", "erase"):
            r0, r1, c0, c1 = comp["bbox"]
            out[r0:r1 + 1, c0:c1 + 1] = bg

    # Draw final 9 rectangles.
    for comp in nine_comps:
        info = nine_actions.get(comp["id"])

        if info is not None and info["action"] == "erase":
            continue

        if info is not None and info["action"] in ("shift", "erode"):
            bbox = info["new_bbox"]
            if bbox is None:
                continue
        else:
            bbox = comp["bbox"]

        r0, r1, c0, c1 = bbox
        out[r0:r1 + 1, c0:c1 + 1] = 9

    # Draw final frames on top.
    for op in frame_ops:
        r0, r1, c0, c1 = op["new_bbox"]
        out[r0:r1 + 1, c0:c1 + 1] = 0
        out[r0 + 1:r1, c0 + 1:c1] = op["interior"]

    return out