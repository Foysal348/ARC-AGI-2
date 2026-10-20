import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by finding the 2-cell-thick rectangular frames and
    the isolated single-pixel seed colors.

    Rule implemented:
    1. Background is 8.
    2. Frames are 4-connected non-background components whose bounding box is
       a full rectangle and whose smaller side has length 2.
    3. Seeds are isolated single-pixel non-frame components.
    4. Let L be the maximum long-side length among all frames.
       Coordinates along a frame side are 1-indexed from the start of that
       side.
    5. For every seed aligned with a frame side, record its 1-indexed
       coordinate on that side.
       The same coordinate is reflected to the perpendicular orientation:
           reflected_coord = L + 1 - coord
    6. Each frame emits horizontal or vertical lines outward from its active
       side. The active side is the side with the unique common "active color"
       inferred from frames that are aligned with at least one seed.
    7. Lines are drawn only through background cells and stop at the first
       non-background cell.
    """
    H, W = grid.shape
    out = grid.astype(int, copy=True)

    background = 8

    # Find connected components of non-background pixels.
    comp_id = -np.ones((H, W), dtype=int)
    components = []

    for r in range(H):
        for c in range(W):
            if grid[r, c] == background or comp_id[r, c] != -1:
                continue

            cid = len(components)
            q = deque([(r, c)])
            comp_id[r, c] = cid
            cells = [(r, c)]

            while q:
                cr, cc = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if grid[nr, nc] != background and comp_id[nr, nc] == -1:
                            comp_id[nr, nc] = cid
                            q.append((nr, nc))
                            cells.append((nr, nc))

            components.append(cells)

    # Classify components into frames and seeds.
    frames = []
    seeds = []

    for cells in components:
        if len(cells) == 1:
            seeds.append(cells[0])
            continue

        rmin = min(r for r, c in cells)
        rmax = max(r for r, c in cells)
        cmin = min(c for r, c in cells)
        cmax = max(c for r, c in cells)
        h = rmax - rmin + 1
        w = cmax - cmin + 1

        # Full rectangle and 2-cell thick.
        if h * w == len(cells) and min(h, w) == 2:
            frames.append((rmin, cmin, rmax, cmax))

    if not frames or not seeds:
        return out

    # Determine the maximum long side length used for reflection.
    L = 0
    for rmin, cmin, rmax, cmax in frames:
        L = max(L, max(rmax - rmin + 1, cmax - cmin + 1))

    def side_cells(frame, side):
        rmin, cmin, rmax, cmax = frame
        if side == "top":
            return [(rmin, c) for c in range(cmin, cmax + 1)]
        if side == "bottom":
            return [(rmax, c) for c in range(cmin, cmax + 1)]
        if side == "left":
            return [(r, cmin) for r in range(rmin, rmax + 1)]
        if side == "right":
            return [(r, cmax) for r in range(rmin, rmax + 1)]
        return []

    # Identify the side color that frames use as their active side.
    # A side is active for a frame if a seed is aligned with that side and
    # the gap between the frame and the seed is all background.
    frame_active = {}
    active_color_votes = {}

    for frame in frames:
        rmin, cmin, rmax, cmax = frame
        h = rmax - rmin + 1
        w = cmax - cmin + 1

        for side in ("top", "bottom", "left", "right"):
            scells = side_cells(frame, side)
            if not scells:
                continue

            # Use the most common color on this side as the candidate color.
            color_counts = {}
            for rr, cc in scells:
                color_counts[grid[rr, cc]] = color_counts.get(grid[rr, cc], 0) + 1
            if not color_counts:
                continue
            side_color = max(color_counts.items(), key=lambda x: x[1])[0]

            # Check whether any seed is aligned outside this side.
            aligned = False
            if side in ("left", "right"):
                for sr, sc in seeds:
                    if rmin <= sr <= rmax:
                        if (side == "left" and sc < cmin) or (side == "right" and sc > cmax):
                            # Check background gap.
                            ok = True
                            if side == "left":
                                for cc in range(sc + 1, cmin):
                                    if grid[sr, cc] != background:
                                        ok = False
                                        break
                            else:
                                for cc in range(cmax + 1, sc):
                                    if grid[sr, cc] != background:
                                        ok = False
                                        break
                            if ok:
                                aligned = True
                                break
            else:
                for sr, sc in seeds:
                    if cmin <= sc <= cmax:
                        if (side == "top" and sr < rmin) or (side == "bottom" and sr > rmax):
                            ok = True
                            if side == "top":
                                for rr in range(sr + 1, rmin):
                                    if grid[rr, sc] != background:
                                        ok = False
                                        break
                            else:
                                for rr in range(rmax + 1, sr):
                                    if grid[rr, sc] != background:
                                        ok = False
                                        break
                            if ok:
                                aligned = True
                                break

            if aligned:
                frame_active[frame] = (side, side_color)
                active_color_votes[side_color] = active_color_votes.get(side_color, 0) + 1

    # Choose the global active color. If none can be inferred, default to 4.
    if active_color_votes:
        active_color = max(active_color_votes.items(), key=lambda x: x[1])[0]
    else:
        active_color = 4

    # Assign active sides to frames without a directly aligned seed.
    for frame in frames:
        if frame in frame_active:
            continue

        rmin, cmin, rmax, cmax = frame
        h = rmax - rmin + 1
        w = cmax - cmin + 1

        best_side = None
        if h >= w:  # vertical frame: active side is left or right
            for side in ("left", "right"):
                scells = side_cells(frame, side)
                if not scells:
                    continue
                if all(grid[rr, cc] == active_color for rr, cc in scells):
                    best_side = side
                    break
            if best_side is None:
                # Fallback: side with more active-color cells.
                best_side = "left"
                cnt_left = sum(1 for rr, cc in side_cells(frame, "left") if grid[rr, cc] == active_color)
                cnt_right = sum(1 for rr, cc in side_cells(frame, "right") if grid[rr, cc] == active_color)
                if cnt_right > cnt_left:
                    best_side = "right"
        else:       # horizontal frame: active side is top or bottom
            for side in ("top", "bottom"):
                scells = side_cells(frame, side)
                if not scells:
                    continue
                if all(grid[rr, cc] == active_color for rr, cc in scells):
                    best_side = side
                    break
            if best_side is None:
                best_side = "top"
                cnt_top = sum(1 for rr, cc in side_cells(frame, "top") if grid[rr, cc] == active_color)
                cnt_bottom = sum(1 for rr, cc in side_cells(frame, "bottom") if grid[rr, cc] == active_color)
                if cnt_bottom > cnt_top:
                    best_side = "bottom"

        frame_active[frame] = (best_side, active_color)

    # Collect coordinates for each seed color.
    # coords[color] = {"h": position or None, "v": position or None}
    coords = {}

    for sr, sc in seeds:
        color = int(grid[sr, sc])
        coords.setdefault(color, {"h": None, "v": None})

        for frame in frames:
            rmin, cmin, rmax, cmax = frame
            h = rmax - rmin + 1
            w = cmax - cmin + 1

            # Vertical frame: seed left/right gives a vertical coordinate.
            if h >= w:
                if rmin <= sr <= rmax:
                    if sc < cmin or sc > cmax:
                        # Ensure the gap is background.
                        ok = True
                        if sc < cmin:
                            for cc in range(sc + 1, cmin):
                                if grid[sr, cc] != background:
                                    ok = False
                                    break
                        else:
                            for cc in range(cmax + 1, sc):
                                if grid[sr, cc] != background:
                                    ok = False
                                    break
                        if ok:
                            p = sr - rmin + 1
                            if coords[color]["v"] is None:
                                coords[color]["v"] = p
                            break

            # Horizontal frame: seed above/below gives a horizontal coordinate.
            else:
                if cmin <= sc <= cmax:
                    if sr < rmin or sr > rmax:
                        ok = True
                        if sr < rmin:
                            for rr in range(sr + 1, rmin):
                                if grid[rr, sc] != background:
                                    ok = False
                                    break
                        else:
                            for rr in range(rmax + 1, sr):
                                if grid[rr, sc] != background:
                                    ok = False
                                    break
                        if ok:
                            p = sc - cmin + 1
                            if coords[color]["h"] is None:
                                coords[color]["h"] = p
                            break

    # Reflect known coordinates to the missing orientation.
    for color, cd in coords.items():
        if cd["h"] is not None and cd["v"] is None:
            cd["v"] = L + 1 - cd["h"]
        if cd["v"] is not None and cd["h"] is None:
            cd["h"] = L + 1 - cd["v"]

    # Draw lines from each frame's active side.
    for frame, (side, _) in frame_active.items():
        rmin, cmin, rmax, cmax = frame

        for color, cd in coords.items():
            if side in ("left", "right"):
                p = cd["v"]
                if p is None or p < 1 or p > rmax - rmin + 1:
                    continue
                r = rmin + p - 1

                if side == "left":
                    c = cmin - 1
                    direction = -1
                else:
                    c = cmax + 1
                    direction = 1

                while 0 <= c < W:
                    if out[r, c] != background:
                        break
                    out[r, c] = color
                    c += direction

            else:
                p = cd["h"]
                if p is None or p < 1 or p > cmax - cmin + 1:
                    continue
                c = cmin + p - 1

                if side == "top":
                    r = rmin - 1
                    direction = -1
                else:
                    r = rmax + 1
                    direction = 1

                while 0 <= r < H:
                    if out[r, c] != background:
                        break
                    out[r, c] = color
                    r += direction

    return out