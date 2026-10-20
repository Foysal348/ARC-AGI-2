import numpy as np
from collections import deque, Counter

# -----------------------------------------------------------------------------
# Helpers for connected components, masks, orientations, and placement scoring
# -----------------------------------------------------------------------------

def _components(mask):
    """4-connected components; returns list of (pixels, bbox)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []

    for r in range(h):
        for c in range(w):
            if not (mask[r, c] and not seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pix = []
            while stack:
                cr, cc = stack.pop()
                pix.append((cr, cc))
                for nr, nc in ((cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            r0 = min(p[0] for p in pix)
            r1 = max(p[0] for p in pix)
            c0 = min(p[1] for p in pix)
            c1 = max(p[1] for p in pix)
            comps.append((pix, (r0, r1, c0, c1)))
    return comps


def _mask_from_pixels(pixels):
    if not pixels:
        return np.zeros((0, 0), dtype=bool)
    r0 = min(p[0] for p in pixels)
    r1 = max(p[0] for p in pixels)
    c0 = min(p[1] for p in pixels)
    c1 = max(p[1] for p in pixels)
    m = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pixels:
        m[r - r0, c - c0] = True
    return m


def _unique_orientations(mask):
    """8 unique orientations of a binary mask."""
    if mask.size == 0:
        return [mask]
    out = []
    cur = mask
    for _ in range(4):
        for m in (cur, np.fliplr(cur)):
            m = np.asarray(m)
            if m.sum() == 0:
                continue
            if not any(np.array_equal(m, x) for x in out):
                out.append(m)
        cur = np.rot90(cur)
    return out


def _can_place(canvas, mask, r, c):
    h, w = canvas.shape
    mh, mw = mask.shape
    if r < 0 or c < 0 or r + mh > h or c + mw > w:
        return False
    sub = canvas[r:r + mh, c:c + mw]
    return not np.any(sub & mask)


def _place(canvas, mask, r, c):
    canvas[r:r + mask.shape[0], c:c + mask.shape[1]] |= mask


def _bbox_of_mask(mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return ys.min(), ys.max(), xs.min(), xs.max()


def _union_bbox(boxes):
    r0 = min(b[0] for b in boxes)
    r1 = max(b[1] for b in boxes)
    c0 = min(b[2] for b in boxes)
    c1 = max(b[3] for b in boxes)
    return r0, r1, c0, c1


def _bbox_area(box):
    r0, r1, c0, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _component_key(comp):
    pix, bbox = comp
    r0, r1, c0, c1 = bbox
    return (r0, c0, c1 - c0 + 1, r1 - r0 + 1, -len(pix))


# -----------------------------------------------------------------------------
# Main transformation
# -----------------------------------------------------------------------------

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform scattered colored pieces into a compact assembled pattern.

    Rule implemented:
      1. The most frequent color is background.
      2. Non-background 4-connected components are pieces.
      3. The dominant "frame" color is the non-background color with the largest
         total pixel count.  It is treated as a fixed scaffold.
      4. All other pieces are packed around this scaffold:
           - first into empty holes inside the frame's bounding box;
           - then adjacent to the frame on the side that best preserves their
             original relative side;
           - orientations are chosen to minimize the final bounding-box area.
      5. If the assembled pattern can be translated so that the frame's bounding
         box becomes flush with its nearest original grid edge, do so.
      6. Colors are kept for the frame and for pieces where possible; in the
         small number of cases where the two movable pieces are clearly
         interchanged by the canonical packing, the color of each final region
         is taken from the piece whose shape was placed there.

    The solver is intentionally deterministic and conservative: it tries a small
    set of canonical placements and chooses the one with the smallest resulting
    bounding box, with tie-breakers based on contact with the frame and distance
    to the original position.
    """
    if grid.size == 0:
        return grid.copy()

    g = np.array(grid, dtype=int, copy=True)
    h, w = g.shape

    # Background is the most frequent color.
    bg = Counter(g.flatten()).most_common(1)[0][0]
    nonbg = g != bg

    if not nonbg.any():
        return g

    # Collect all non-background components.
    raw_comps = _components(nonbg)
    comps = []
    for pix, bbox in raw_comps:
        col = g[pix[0][0], pix[0][1]]
        if col == bg:
            continue
        comps.append({
            "color": col,
            "pixels": pix,
            "bbox": bbox,
            "mask": _mask_from_pixels(pix),
            "center": (
                sum(p[0] for p in pix) / len(pix),
                sum(p[1] for p in pix) / len(pix),
            ),
            "size": len(pix),
        })

    if not comps:
        return g

    # If there is only one component, nothing meaningful to pack.
    if len(comps) == 1:
        return g

    # Choose frame color: largest total non-background pixel count.
    color_total = Counter()
    for comp in comps:
        color_total[comp["color"]] += comp["size"]
    frame_color = max(color_total, key=lambda c: (color_total[c], -c))

    frame_comps = [c for c in comps if c["color"] == frame_color]
    other_comps = [c for c in comps if c["color"] != frame_color]

    # Frame bounding box and hole mask.
    frame_boxes = [c["bbox"] for c in frame_comps]
    fr0, fr1, fc0, fc1 = _union_bbox(frame_boxes)
    frame_h = fr1 - fr0 + 1
    frame_w = fc1 - fc0 + 1

    frame_mask = np.zeros((h, w), dtype=bool)
    for comp in frame_comps:
        for r, c in comp["pixels"]:
            frame_mask[r, c] = True

    hole_mask = np.zeros((h, w), dtype=bool)
    hole_mask[fr0:fr1 + 1, fc0:fc1 + 1] = True
    hole_mask &= ~frame_mask
    hole_mask &= (g == bg)

    # Determine the primary axis of the frame (vertical or horizontal separation).
    # We prefer an axis where frame components are disjoint and the other axis
    # overlaps.  If there is only one frame component, infer from the aspect ratio
    # of the frame bbox.
    def _axis_info():
        if len(frame_comps) <= 1:
            if frame_h >= frame_w:
                return "V"
            return "H"

        x_sep = 0
        y_sep = 0
        boxes = [c["bbox"] for c in frame_comps]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                r0i, r1i, c0i, c1i = boxes[i]
                r0j, r1j, c0j, c1j = boxes[j]
                # horizontal disjointness
                if c1i < c0j:
                    x_sep += c0j - c1i - 1
                elif c1j < c0i:
                    x_sep += c0i - c1j - 1
                # vertical disjointness
                if r1i < r0j:
                    y_sep += r0j - r1i - 1
                elif r1j < r0i:
                    y_sep += r0i - r1j - 1
        if x_sep > y_sep:
            return "H"
        if y_sep > x_sep:
            return "V"
        # tie: use bbox aspect
        return "V" if frame_h >= frame_w else "H"

    axis = _axis_info()

    # Smaller side of the frame: used for bias/tie-breaking.
    # Compare the centroid of the smallest frame component with the frame centroid.
    frame_ys, frame_xs = np.where(frame_mask)
    if len(frame_ys):
        frame_cy = frame_ys.mean()
        frame_cx = frame_xs.mean()
    else:
        frame_cy = (fr0 + fr1) / 2.0
        frame_cx = (fc0 + fc1) / 2.0

    if frame_comps:
        small_frame = min(frame_comps, key=lambda c: c["size"])
        scy, scx = small_frame["center"]
    else:
        scy, scx = frame_cy, frame_cx

    if axis == "H":
        small_side = "left" if scx <= frame_cx else "right"
    else:
        small_side = "top" if scy <= frame_cy else "bottom"

    # -----------------------------------------------------------------
    # Prepare a working canvas with the frame fixed in its original place.
    # -----------------------------------------------------------------
    placed = np.zeros((h, w), dtype=bool)
    color_canvas = np.full((h, w), bg, dtype=int)
    for comp in frame_comps:
        for r, c in comp["pixels"]:
            placed[r, c] = True
            color_canvas[r, c] = frame_color

    placed_boxes = [c["bbox"] for c in frame_comps]

    def _current_bbox():
        if not placed_boxes:
            return (0, 0, 0, 0)
        return _union_bbox(placed_boxes)

    def _score_configuration(canvas, boxes, original_other_comps):
        """
        Lower is better.
        Primary: bounding-box area.
        Secondary: bounding-box perimeter.
        Tertiary: keep pieces near their original locations.
        """
        if not boxes:
            return (0, 0, 0)
        r0, r1, c0, c1 = _union_bbox(boxes)
        area = _bbox_area((r0, r1, c0, c1))
        perim = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1))

        # Distance penalty: compare each placed non-frame mask to its original
        # centroid.  This is only a tie-breaker.
        dist = 0.0
        nonframe_mask = canvas.copy()
        for comp in frame_comps:
            for r, c in comp["pixels"]:
                nonframe_mask[r, c] = False
        ys, xs = np.where(nonframe_mask)
        if len(ys) == 0:
            return (area, perim, 0)

        # Pair non-frame pixels to the closest original non-frame centroid
        # approximately by averaging per component.  For compactness, just use
        # total mass centroid.
        if original_other_comps:
            tot = sum(c["size"] for c in original_other_comps)
            if tot:
                orig_cy = sum(c["center"][0] * c["size"] for c in original_other_comps) / tot
                orig_cx = sum(c["center"][1] * c["size"] for c in original_other_comps) / tot
                new_cy = ys.mean()
                new_cx = xs.mean()
                dist = abs(new_cy - orig_cy) + abs(new_cx - orig_cx)
        return (area, perim, int(dist * 4))

    # -----------------------------------------------------------------
    # 1) Try to place each non-frame component inside the frame hole.
    # -----------------------------------------------------------------
    remaining = list(other_comps)
    inside_placed = []
    outside_placed = []

    # Smaller pieces first for holes.
    for comp in sorted(remaining, key=lambda x: (x["size"], _component_key((x["pixels"], x["bbox"])))):
        best = None
        best_key = None

        for orient in _unique_orientations(comp["mask"]):
            oh, ow = orient.shape
            if oh > frame_h or ow > frame_w:
                continue
            # Only try translations whose bbox lies inside the frame bbox.
            rr = range(fr0, fr1 - oh + 2)
            cc = range(fc0, fc1 - ow + 2)
            for r in rr:
                for c in cc:
                    if not _can_place(placed, orient, r, c):
                        continue
                    # It must cover at least one hole cell.
                    if not np.any(orient & hole_mask[r:r + oh, c:c + ow]):
                        continue
                    # Penalize covering non-hole background outside frame (shouldn't happen)
                    cov = np.sum(orient & hole_mask[r:r + oh, c:c + ow])
                    area_after = _bbox_area(_current_bbox())  # unchanged if inside
                    # Contact with frame is a nice tie-breaker.
                    contact = 0
                    sub = placed[r:r + oh, c:c + ow]
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            if dr == 0 and dc == 0:
                                continue
                            if -1 <= r + dr < h and -1 <= c + dc < w:
                                pass
                    # Simpler: dilate orient by one pixel and intersect frame.
                    dil = np.zeros_like(orient)
                    dil[1:, :] |= orient[:-1, :]
                    dil[:-1, :] |= orient[1:, :]
                    dil[:, 1:] |= orient[:, :-1]
                    dil[:, :-1] |= orient[:, 1:]
                    contact = np.sum(dil & placed[r:r + oh, c:c + ow])

                    key = (
                        -cov,
                        -contact,
                        area_after,
                        r, c,
                    )
                    if best_key is None or key < best_key:
                        best_key = key
                        best = (orient, r, c)

        if best is not None:
            orient, r, c = best
            _place(placed, orient, r, c)
            r0, r1, c0, c1 = r, r + orient.shape[0] - 1, c, c + orient.shape[1] - 1
            placed_boxes.append((r0, r1, c0, c1))
            inside_placed.append((comp, (r, c, orient)))
            remaining.remove(comp)

    # -----------------------------------------------------------------
    # 2) Place remaining pieces outside the frame.
    # -----------------------------------------------------------------
    # For vertical frame axis: put outside pieces left/right of frame.
    # For horizontal frame axis: put above/below frame.
    # Choose the side closest to the piece's original center.
    # Choose orientation/offset minimizing final bbox area.

    def _bias_row(piece_h):
        """Vertical offset preference for V-axis outside placement."""
        # Bias toward the smaller side.
        if small_side == "top":
            return 0
        elif small_side == "bottom":
            return 1
        return 0

    def _bias_col(piece_w):
        """Horizontal offset preference for H-axis outside placement."""
        if small_side == "left":
            return 0
        elif small_side == "right":
            return 1
        return 0

    # Sort outside pieces: larger pieces first often gives a more compact core.
    remaining.sort(key=lambda c: (-c["size"], _component_key((c["pixels"], c["bbox"]))))

    for comp in remaining:
        best = None
        best_key = None

        for orient in _unique_orientations(comp["mask"]):
            oh, ow = orient.shape

            if axis == "V":
                # Left or right of frame.
                sides = []
                orig_cy, orig_cx = comp["center"]
                if orig_cx <= fc0:
                    sides.append(("left", fc0 - ow))
                    sides.append(("right", fc1 + 1))
                else:
                    sides.append(("right", fc1 + 1))
                    sides.append(("left", fc0 - ow))

                for side, c in sides:
                    # Vertical offsets: keep bbox compact, but allow small shift.
                    # Try offsets from -oh to frame_h.
                    for dr in range(-oh, frame_h + 1):
                        r = fr0 + dr
                        if not _can_place(placed, orient, r, c):
                            continue
                        boxes = placed_boxes + [(r, r + oh - 1, c, c + ow - 1)]
                        bbox = _union_bbox(boxes)
                        area = _bbox_area(bbox)

                        # Bias: if smaller side is top, prefer smaller r; if bottom, larger r.
                        bias_pen = 0
                        if small_side == "top":
                            bias_pen = r
                        elif small_side == "bottom":
                            bias_pen = -(r + oh)
                        else:
                            bias_pen = abs((r + oh / 2.0) - frame_cy)

                        key = (
                            area,
                            abs((r + oh / 2.0) - comp["center"][0]),
                            bias_pen,
                            r, c,
                        )
                        if best_key is None or key < best_key:
                            best_key = key
                            best = (orient, r, c)

            else:
                # Above or below frame.
                sides = []
                orig_cy, orig_cx = comp["center"]
                if orig_cy <= fr0:
                    sides.append(("above", fr0 - oh))
                    sides.append(("below", fr1 + 1))
                else:
                    sides.append(("below", fr1 + 1))
                    sides.append(("above", fr0 - oh))

                for side, r in sides:
                    for dc in range(-ow, frame_w + 1):
                        c = fc0 + dc
                        if not _can_place(placed, orient, r, c):
                            continue
                        boxes = placed_boxes + [(r, r + oh - 1, c, c + ow - 1)]
                        bbox = _union_bbox(boxes)
                        area = _bbox_area(bbox)

                        bias_pen = 0
                        if small_side == "left":
                            bias_pen = c
                        elif small_side == "right":
                            bias_pen = -(c + ow)
                        else:
                            bias_pen = abs((c + ow / 2.0) - frame_cx)

                        key = (
                            area,
                            abs((c + ow / 2.0) - comp["center"][1]),
                            bias_pen,
                            r, c,
                        )
                        if best_key is None or key < best_key:
                            best_key = key
                            best = (orient, r, c)

        if best is not None:
            orient, r, c = best
            _place(placed, orient, r, c)
            r0, r1, c0, c1 = r, r + orient.shape[0] - 1, c, c + orient.shape[1] - 1
            placed_boxes.append((r0, r1, c0, c1))
            outside_placed.append((comp, (r, c, orient)))

    # -----------------------------------------------------------------
    # 3) Assign final colors.
    #    - Frame pixels keep frame_color.
    #    - Each moved piece keeps its original color if it was placed.
    #    - If two pieces of different colors were both moved and the canonical
    #      packing clearly swaps their roles (as in Example 2), the above
    #      shape-based placement already determines which shape goes where;
    #      color is attached to the placed shape, which matches all examples.
    # -----------------------------------------------------------------
    out = np.full((h, w), bg, dtype=int)

    # Place frame first.
    for comp in frame_comps:
        for r, c in comp["pixels"]:
            out[r, c] = frame_color

    # Place inside pieces.
    for comp, (r, c, orient) in inside_placed:
        sub = out[r:r + orient.shape[0], c:c + orient.shape[1]]
        sub[orient] = comp["color"]

    # Place outside pieces.
    for comp, (r, c, orient) in outside_placed:
        sub = out[r:r + orient.shape[0], c:c + orient.shape[1]]
        sub[orient] = comp["color"]

    # -----------------------------------------------------------------
    # 4) Optional global translation: flush the frame bbox to its nearest
    #    original grid edge, if that does not cause out-of-bounds.
    # -----------------------------------------------------------------
    # Use the bbox of all placed non-background pixels.
    ys, xs = np.where(out != bg)
    if len(ys):
        r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
        # Distances of the frame bbox to the four edges in the original input.
        d_top = fr0
        d_bottom = (h - 1) - fr1
        d_left = fc0
        d_right = (w - 1) - fc1
        best_shift = (0, 0)
        best_pen = (min(d_top, d_bottom, d_left, d_right), 0)

        # Try shifting so the frame touches the nearest edge.
        candidates = []
        if d_top == min(d_top, d_bottom, d_left, d_right):
            candidates.append((0 - r0 + fr0, 0))
        if d_bottom == min(d_top, d_bottom, d_left, d_right):
            candidates.append(((h - 1) - r1, 0))
        if d_left == min(d_top, d_bottom, d_left, d_right):
            candidates.append((0, 0 - c0 + fc0))
        if d_right == min(d_top, d_bottom, d_left, d_right):
            candidates.append((0, (w - 1) - c1))

        # Also keep zero shift as a safe candidate.
        candidates.append((0, 0))

        for dr, dc in candidates:
            nr0 = r0 + dr
            nr1 = r1 + dr
            nc0 = c0 + dc
            nc1 = c1 + dc
            if nr0 < 0 or nc0 < 0 or nr1 >= h or nc1 >= w:
                continue
            # Prefer smaller distance of frame after shift to the chosen edge.
            frame_nr0 = fr0 + dr
            frame_nr1 = fr1 + dr
            frame_nc0 = fc0 + dc
            frame_nc1 = fc1 + dc
            pen = min(frame_nr0, (h - 1) - frame_nr1, frame_nc0, (w - 1) - frame_nc1)
            key = (pen, abs(dr) + abs(dc))
            if best_pen is None or key < best_pen:
                best_pen = key
                best_shift = (dr, dc)

        if best_shift != (0, 0):
            dr, dc = best_shift
            shifted = np.full((h, w), bg, dtype=int)
            r0s = max(0, r0 + dr)
            r1s = min(h - 1, r1 + dr)
            c0s = max(0, c0 + dc)
            c1s = min(w - 1, c1 + dc)
            r0o = r0s - dr
            c0o = c0s - dc
            shifted[r0s:r1s + 1, c0s:c1s + 1] = out[r0o:r0o + (r1s - r0s + 1), c0o:c0o + (c1s - c0s + 1)]
            out = shifted

    return out