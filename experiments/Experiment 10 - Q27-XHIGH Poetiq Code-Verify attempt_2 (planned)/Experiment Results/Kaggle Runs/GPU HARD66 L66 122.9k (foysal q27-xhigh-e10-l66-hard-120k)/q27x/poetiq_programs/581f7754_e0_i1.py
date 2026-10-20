import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Move colored objects so that their center/special pixel aligns with a
    matching border marker pixel.

    Rule summary:
      - The background color is the most frequent color.
      - A single border pixel of a non-background color is a marker/reference.
      - For every object containing that color, move the object so the special
        pixel aligns with the marker's row/column.
      - Markers on left/right borders imply vertical alignment (same row).
      - Markers on top/bottom borders imply horizontal alignment (same column).
      - If the object's frame is a straight 1-pixel-wide line ending at the
        special pixel, the far end of the line stays fixed and the line is
        extended/contracted to the new special-pixel position.
      - Otherwise the object moves rigidly.
    """
    if grid is None:
        return grid

    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2:
        return grid.copy()

    H, W = grid.shape
    if H == 0 or W == 0:
        return grid.copy()

    out = grid.copy()

    # Background is the most common color.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    nonbg = grid != bg
    if not nonbg.any():
        return out

    dirs8 = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    # ------------------------------------------------------------
    # 1. Find 8-connected components of non-background pixels.
    # ------------------------------------------------------------
    comp_id = np.full((H, W), -1, dtype=int)
    comps = []

    for r in range(H):
        for c in range(W):
            if nonbg[r, c] and comp_id[r, c] == -1:
                cid = len(comps)
                q = deque([(r, c)])
                comp_id[r, c] = cid
                pixels = []

                while q:
                    rr, cc = q.popleft()
                    pixels.append((rr, cc))

                    for dr, dc in dirs8:
                        nr, nc = rr + dr, cc + dc
                        if (
                            0 <= nr < H and 0 <= nc < W and
                            nonbg[nr, nc] and comp_id[nr, nc] == -1
                        ):
                            comp_id[nr, nc] = cid
                            q.append((nr, nc))

                color_counts = {}
                for rr, cc in pixels:
                    v = int(grid[rr, cc])
                    color_counts[v] = color_counts.get(v, 0) + 1

                touches_border = any(
                    rr == 0 or rr == H - 1 or cc == 0 or cc == W - 1
                    for rr, cc in pixels
                )

                comps.append({
                    "id": cid,
                    "pixels": pixels,
                    "counts": color_counts,
                    "border": touches_border
                })

    # ------------------------------------------------------------
    # 2. Find reference/marker pixels.
    #    We use single-pixel border components as markers.
    # ------------------------------------------------------------
    ref_candidates = {}
    for comp in comps:
        if len(comp["counts"]) == 1 and len(comp["pixels"]) == 1 and comp["border"]:
            color = next(iter(comp["counts"]))
            ref_candidates.setdefault(color, []).append((comp["id"], comp["pixels"][0]))

    def axis_for(r, c):
        horizontal_border = (r == 0 or r == H - 1)
        vertical_border = (c == 0 or c == W - 1)

        if horizontal_border and not vertical_border:
            return "h"
        if vertical_border and not horizontal_border:
            return "v"

        # Corner: prefer horizontal if possible.
        if horizontal_border:
            return "h"
        return "v"

    ref_info = {}
    ref_comp_ids = set()

    for color, items in ref_candidates.items():
        items.sort(key=lambda x: (x[1][0], x[1][1]))
        comp_id_ref, (r, c) = items[0]
        ref_info[color] = (r, c, axis_for(r, c))
        ref_comp_ids.add(comp_id_ref)

    if not ref_info:
        return out

    # ------------------------------------------------------------
    # Helper: collect the frame-color pixels connected to an anchor.
    # ------------------------------------------------------------
    def get_frame_pixels(anchor, frame_color):
        ar, ac = anchor
        starts = []
        for dr, dc in dirs8:
            nr, nc = ar + dr, ac + dc
            if 0 <= nr < H and 0 <= nc < W and int(grid[nr, nc]) == frame_color:
                starts.append((nr, nc))

        if not starts:
            return []

        seen = set(starts)
        q = deque(starts)
        pixels = []

        while q:
            rr, cc = q.popleft()
            pixels.append((rr, cc))

            for dr, dc in dirs8:
                nr, nc = rr + dr, cc + dc
                if (
                    0 <= nr < H and 0 <= nc < W and
                    (nr, nc) not in seen and
                    int(grid[nr, nc]) == frame_color
                ):
                    seen.add((nr, nc))
                    q.append((nr, nc))

        return pixels

    # ------------------------------------------------------------
    # 3. Identify movable objects.
    # ------------------------------------------------------------
    objects = []

    for comp in comps:
        if comp["id"] in ref_comp_ids:
            continue

        # The special color is the rarest color in this component
        # that also has a border reference.
        candidate_colors = [col for col in comp["counts"] if col in ref_info]
        if not candidate_colors:
            continue

        candidate_colors.sort(key=lambda col: (comp["counts"][col], col))
        special = candidate_colors[0]

        sp_pixels = [p for p in comp["pixels"] if int(grid[p[0], p[1]]) == special]
        if not sp_pixels:
            continue

        # Choose anchor special pixel.
        if len(sp_pixels) == 1:
            anchor = sp_pixels[0]
        else:
            cr = sum(p[0] for p in comp["pixels"]) / len(comp["pixels"])
            cc = sum(p[1] for p in comp["pixels"]) / len(comp["pixels"])
            anchor = min(
                sp_pixels,
                key=lambda p: (p[0] - cr) ** 2 + (p[1] - cc) ** 2
            )

        # Determine the associated frame color, if any.
        frame_color = None
        frame_pixels = []

        if len(comp["counts"]) > 1:
            adjacent_colors = set()
            ar, ac = anchor
            for dr, dc in dirs8:
                nr, nc = ar + dr, ac + dc
                if 0 <= nr < H and 0 <= nc < W:
                    v = int(grid[nr, nc])
                    if v != bg and v != special:
                        adjacent_colors.add(v)

            if adjacent_colors:
                frame_color = max(
                    adjacent_colors,
                    key=lambda col: (comp["counts"].get(col, 0), col)
                )
            else:
                other = [col for col in comp["counts"] if col != special]
                if other:
                    frame_color = max(
                        other,
                        key=lambda col: (comp["counts"][col], col)
                    )

            if frame_color is not None:
                frame_pixels = get_frame_pixels(anchor, frame_color)

        # Decide whether this is a straight-line object.
        mode = "rigid"
        line_axis = None
        far_end = None

        if frame_pixels:
            same_col = all(p[1] == frame_pixels[0][1] for p in frame_pixels)
            same_row = all(p[0] == frame_pixels[0][0] for p in frame_pixels)
            move_axis = ref_info[special][2]

            if same_col:
                col = frame_pixels[0][1]
                if anchor[1] == col:
                    rows = [p[0] for p in frame_pixels]
                    min_r, max_r = min(rows), max(rows)

                    if move_axis == "v":
                        if anchor[0] == max_r + 1:
                            mode = "line"
                            line_axis = "v"
                            far_end = (min_r, col)
                        elif anchor[0] == min_r - 1:
                            mode = "line"
                            line_axis = "v"
                            far_end = (max_r, col)

            elif same_row:
                row = frame_pixels[0][0]
                if anchor[0] == row:
                    cols = [p[1] for p in frame_pixels]
                    min_c, max_c = min(cols), max(cols)

                    if move_axis == "h":
                        if anchor[1] == max_c + 1:
                            mode = "line"
                            line_axis = "h"
                            far_end = (row, min_c)
                        elif anchor[1] == min_c - 1:
                            mode = "line"
                            line_axis = "h"
                            far_end = (row, max_c)

        # Compute movement delta for the anchor.
        r_ref, c_ref, axis = ref_info[special]

        if axis == "v":
            target_r, target_c = r_ref, anchor[1]
        else:
            target_r, target_c = anchor[0], c_ref

        dr = target_r - anchor[0]
        dc = target_c - anchor[1]

        # Pixels that belong to this movable object.
        if mode == "line":
            obj_pixels = [anchor] + frame_pixels
        else:
            obj_pixels = list(sp_pixels) + frame_pixels

        # Remove duplicate pixels, preserving order.
        obj_pixels = list(dict.fromkeys(obj_pixels))

        if not obj_pixels:
            continue

        objects.append({
            "mode": mode,
            "obj_pixels": obj_pixels,
            "anchor": anchor,
            "dr": dr,
            "dc": dc,
            "special": special,
            "frame_color": frame_color,
            "line_axis": line_axis,
            "far_end": far_end
        })

    # ------------------------------------------------------------
    # 4. Apply transformations.
    # ------------------------------------------------------------
    for obj in objects:
        dr, dc = obj["dr"], obj["dc"]

        if obj["mode"] == "line":
            # Erase old line object.
            for r, c in obj["obj_pixels"]:
                out[r, c] = bg

            if obj["line_axis"] == "v":
                far_r, col = obj["far_end"]
                new_r = obj["anchor"][0] + dr
                new_c = obj["anchor"][1] + dc

                if 0 <= new_r < H and 0 <= new_c < W:
                    start = min(far_r, new_r)
                    end = max(far_r, new_r)

                    for rr in range(start, end + 1):
                        if rr != new_r and 0 <= rr < H and 0 <= col < W:
                            out[rr, col] = obj["frame_color"]

                    out[new_r, new_c] = obj["special"]

            elif obj["line_axis"] == "h":
                row, far_c = obj["far_end"]
                new_r = obj["anchor"][0] + dr
                new_c = obj["anchor"][1] + dc

                if 0 <= new_r < H and 0 <= new_c < W:
                    start = min(far_c, new_c)
                    end = max(far_c, new_c)

                    for cc in range(start, end + 1):
                        if cc != new_c and 0 <= row < H and 0 <= cc < W:
                            out[row, cc] = obj["frame_color"]

                    out[new_r, new_c] = obj["special"]

        else:
            # Erase old object.
            for r, c in obj["obj_pixels"]:
                out[r, c] = bg

            # Place moved object.
            for r, c in obj["obj_pixels"]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    out[nr, nc] = int(grid[r, c])

    # Restore marker pixels in case an object moved over them.
    for color, (r, c, _) in ref_info.items():
        out[r, c] = color

    return out