import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid according to the rules inferred from the examples.

    The rule
    -------
    1. Find all rectangular 0-borders ("frames") whose interiors contain 5/7.
    2. Each frame has an adjacent 9-rectangle on one side.
       - If there is another 9-rectangle farther in the same direction, the frame
         and its adjacent 9-rectangle move toward that far 9-rectangle.
         The desired movement is the distance until the moving 9-rectangle
         collides with the far 9-rectangle.
       - If there is no farther 9-rectangle, the frame moves into the adjacent
         9-rectangle by at most (thickness_of_9 - 1), eroding the 9-rectangle.
    3. The actual movement distance is capped by the number of 5-cells inside
       the frame, because that many 5-cells will be turned into 7-cells.
    4. Inside each frame, starting from the side that was adjacent to the 9,
       change exactly d 5-cells to 7 in a fixed scan order:
         left   : columns left-to-right, rows bottom-to-top
         right  : columns right-to-left, rows top-to-bottom
         top    : rows top-to-bottom,   columns left-to-right
         bottom : rows bottom-to-top,   columns right-to-left
       Existing 7-cells are skipped.
    """
    g = np.array(grid, dtype=int)
    if g.ndim != 2:
        return g

    H, W = g.shape
    if H == 0 or W == 0:
        return g

    # Background color is the most frequent color (usually 6).
    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    DIRS4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def connected_components(mask):
        """Return 4-connected components of a boolean mask."""
        labels = np.zeros((H, W), dtype=int)
        comps = []
        cid = 0

        for r0 in range(H):
            for c0 in range(W):
                if not mask[r0, c0] or labels[r0, c0] != 0:
                    continue

                cid += 1
                stack = [(r0, c0)]
                labels[r0, c0] = cid
                pixels = []

                while stack:
                    r, c = stack.pop()
                    pixels.append((r, c))
                    for dr, dc in DIRS4:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if mask[nr, nc] and labels[nr, nc] == 0:
                                labels[nr, nc] = cid
                                stack.append((nr, nc))

                rs = [p[0] for p in pixels]
                cs = [p[1] for p in pixels]
                comps.append({
                    "id": cid,
                    "bbox": (min(rs), max(rs), min(cs), max(cs)),
                    "pixels": pixels
                })

        return comps, labels

    # ------------------------------------------------------------------
    # Find all 0-frames: rectangular borders made only of 0.
    # ------------------------------------------------------------------
    zero_comps, zero_labels = connected_components(g == 0)
    frames = []

    for comp in zero_comps:
        r0, r1, c0, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1

        if h < 3 or w < 3:
            continue

        expected_perimeter = 2 * h + 2 * w - 4
        if len(comp["pixels"]) != expected_perimeter:
            continue

        # All boundary cells must belong to this 0-component.
        ok = True
        for r in (r0, r1):
            for c in range(c0, c1 + 1):
                if zero_labels[r, c] != comp["id"]:
                    ok = False
                    break
            if not ok:
                break

        if ok:
            for r in range(r0 + 1, r1):
                if zero_labels[r, c0] != comp["id"] or zero_labels[r, c1] != comp["id"]:
                    ok = False
                    break

        if not ok:
            continue

        # Interior must not contain 0.
        if r1 > r0 + 1 and c1 > c0 + 1:
            if np.any(g[r0 + 1:r1, c0 + 1:c1] == 0):
                continue

        frames.append({
            "bbox": (r0, r1, c0, c1),
            "interior": g[r0 + 1:r1, c0 + 1:c1].copy()
        })

    # ------------------------------------------------------------------
    # Find all 9-components.
    # ------------------------------------------------------------------
    nine_comps, _ = connected_components(g == 9)

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
        """Erode a 9-rectangle from the side facing the frame."""
        r0, r1, c0, c1 = bbox
        if side == "left":
            # 9 is to the left of the frame: erode its right edge.
            return (r0, r1, c0, c1 - d)
        if side == "right":
            # 9 is to the right of the frame: erode its left edge.
            return (r0, r1, c0 + d, c1)
        if side == "up":
            # 9 is above the frame: erode its bottom edge.
            return (r0, r1 - d, c0, c1)
        if side == "down":
            # 9 is below the frame: erode its top edge.
            return (r0 + d, r1, c0, c1)
        return bbox

    def transform_interior(interior, side, d):
        """Convert d 5-cells to 7 using the side-dependent scan order."""
        if d <= 0:
            return interior.copy()

        out = interior.copy()
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
        elif side == "top":
            for r in range(ih):
                for c in range(iw):
                    if out[r, c] == 5:
                        out[r, c] = 7
                        added += 1
                        if added == d:
                            return out
        elif side == "bottom":
            for r in range(ih - 1, -1, -1):
                for c in range(iw - 1, -1, -1):
                    if out[r, c] == 5:
                        out[r, c] = 7
                        added += 1
                        if added == d:
                            return out

        return out

    # ------------------------------------------------------------------
    # For each frame, compute its movement and the action for its 9.
    # ------------------------------------------------------------------
    frame_ops = []
    nine_actions = {}

    for frame in frames:
        fr0, fr1, fc0, fc1 = frame["bbox"]
        interior = frame["interior"]

        # Find an adjacent 9-rectangle.
        near = None
        near_side = None
        best_overlap = -1

        for n in nine_comps:
            nr0, nr1, nc0, nc1 = n["bbox"]

            candidates = []

            # Left of frame
            if nc1 == fc0 - 1:
                ov = min(nr1, fr1) - max(nr0, fr0) + 1
                if ov > 0:
                    candidates.append(("left", n, ov))

            # Right of frame
            if nc0 == fc1 + 1:
                ov = min(nr1, fr1) - max(nr0, fr0) + 1
                if ov > 0:
                    candidates.append(("right", n, ov))

            # Above frame
            if nr1 == fr0 - 1:
                ov = min(nc1, fc1) - max(nc0, fc0) + 1
                if ov > 0:
                    candidates.append(("up", n, ov))

            # Below frame
            if nr0 == fr1 + 1:
                ov = min(nc1, fc1) - max(nc0, fc0) + 1
                if ov > 0:
                    candidates.append(("down", n, ov))

            for side, comp, ov in candidates:
                if ov > best_overlap:
                    best_overlap = ov
                    near = comp
                    near_side = side

        if near is None:
            # No associated 9: leave the frame unchanged.
            frame_ops.append({
                "new_bbox": frame["bbox"],
                "interior": interior.copy(),
                "near_id": None,
                "near_action": None
            })
            continue

        # Find the farther 9 in the same direction, if any.
        far = None
        best_far_dist = None

        nr0, nr1, nc0, nc1 = near["bbox"]

        for n in nine_comps:
            if n is near:
                continue

            gr0, gr1, gc0, gc1 = n["bbox"]

            if near_side == "left":
                # Farther left.
                if gc1 < nc0:
                    ov = min(gr1, fr1) - max(gr0, fr0) + 1
                    if ov > 0:
                        dist = nc0 - gc1
                        if dist > 0 and (best_far_dist is None or dist < best_far_dist):
                            best_far_dist = dist
                            far = n

            elif near_side == "right":
                # Farther right.
                if gc0 > nc1:
                    ov = min(gr1, fr1) - max(gr0, fr0) + 1
                    if ov > 0:
                        dist = gc0 - nc1
                        if dist > 0 and (best_far_dist is None or dist < best_far_dist):
                            best_far_dist = dist
                            far = n

            elif near_side == "up":
                # Farther up.
                if gr1 < nr0:
                    ov = min(gc1, fc1) - max(gc0, fc0) + 1
                    if ov > 0:
                        dist = nr0 - gr1
                        if dist > 0 and (best_far_dist is None or dist < best_far_dist):
                            best_far_dist = dist
                            far = n

            elif near_side == "down":
                # Farther down.
                if gr0 > nr1:
                    ov = min(gc1, fc1) - max(gc0, fc0) + 1
                    if ov > 0:
                        dist = gr0 - nr1
                        if dist > 0 and (best_far_dist is None or dist < best_far_dist):
                            best_far_dist = dist
                            far = n

        # Desired movement distance.
        if far is not None:
            fr0f, fr1f, fc0f, fc1f = far["bbox"]
            if near_side == "left":
                desired = nc0 - fc1f
            elif near_side == "right":
                desired = fc0f - nc1
            elif near_side == "up":
                desired = nr0 - fr1f
            elif near_side == "down":
                desired = fr0f - nr1
            else:
                desired = 0
        else:
            # No far 9: move into the adjacent 9 until it becomes a line.
            if near_side in ("left", "right"):
                thickness = nc1 - nc0 + 1
            else:
                thickness = nr1 - nr0 + 1
            desired = max(0, thickness - 1)

        desired = max(0, int(desired))

        # Actual distance is capped by the number of 5-cells inside.
        five_count = int(np.sum(interior == 5))
        d = min(desired, five_count)

        new_frame_bbox = shift_bbox(frame["bbox"], d, near_side)
        new_interior = transform_interior(interior, near_side, d)

        near_id = near["id"]

        if far is not None:
            if desired > 0 and d == desired:
                near_action = "erase"
                new_near_bbox = None
            else:
                near_action = "shift"
                new_near_bbox = shift_bbox(near["bbox"], d, near_side)
        else:
            if d > 0:
                near_action = "erode"
                new_near_bbox = erode_bbox(near["bbox"], d, near_side)
            else:
                near_action = "none"
                new_near_bbox = near["bbox"]

        nine_actions[near_id] = {
            "action": near_action,
            "new_bbox": new_near_bbox
        }

        frame_ops.append({
            "new_bbox": new_frame_bbox,
            "interior": new_interior,
            "near_id": near_id,
            "near_action": near_action
        })

    # ------------------------------------------------------------------
    # Build the output.
    # ------------------------------------------------------------------
    out = g.copy()

    # Clear old frame boxes.
    for frame in frames:
        r0, r1, c0, c1 = frame["bbox"]
        out[r0:r1 + 1, c0:c1 + 1] = bg

    # Clear old 9-boxes that will be moved/eroded/erased.
    for comp in nine_comps:
        act = nine_actions.get(comp["id"], {}).get("action", "none")
        if act in ("shift", "erode", "erase"):
            r0, r1, c0, c1 = comp["bbox"]
            out[r0:r1 + 1, c0:c1 + 1] = bg

    # Draw final 9 components.
    for comp in nine_comps:
        act_info = nine_actions.get(comp["id"], {})
        act = act_info.get("action", "none")

        if act == "erase":
            continue

        if act in ("shift", "erode"):
            r0, r1, c0, c1 = act_info["new_bbox"]
        else:
            r0, r1, c0, c1 = comp["bbox"]

        out[r0:r1 + 1, c0:c1 + 1] = 9

    # Draw final frames.
    for op in frame_ops:
        r0, r1, c0, c1 = op["new_bbox"]
        out[r0:r1 + 1, c0:c1 + 1] = 0

        interior = op["interior"]
        ih, iw = interior.shape

        if ih > 0 and iw > 0:
            out[r0 + 1:r1, c0 + 1:c1] = interior

    return out