import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Pull thin rope-like objects away from straight-line anchors.

    - Background is the most frequent color.
    - Connected components are found per color using 8-neighborhood.
    - A component is an anchor if it is a straight horizontal/vertical line.
    - A thin non-line component 4-adjacent to an anchor endpoint is a rope.
    - The rope is re-drawn as a taut path:
        * starts at its attachment cell,
        * moves outward from the anchor endpoint,
        * wraps around the nearest fixed component in that direction,
        * keeps the original rope length when possible,
        * stops at the grid border if length remains.
    """
    grid = np.asarray(grid)
    H, W = grid.shape
    if H == 0 or W == 0:
        return grid.copy()

    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    # ---------- component labeling (8-connected, per color) ----------
    comp_id = np.full((H, W), -1, dtype=int)
    comps = []
    n8 = [(-1, -1), (-1, 0), (-1, 1),
          (0, -1),           (0, 1),
          (1, -1),  (1, 0),  (1, 1)]

    for color in vals:
        if color == bg:
            continue
        mask = (grid == color)
        for r in range(H):
            for c in range(W):
                if not mask[r, c] or comp_id[r, c] != -1:
                    continue

                cid = len(comps)
                stack = [(r, c)]
                comp_id[r, c] = cid
                cells = []

                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))
                    for dr, dc in n8:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if mask[nr, nc] and comp_id[nr, nc] == -1:
                                comp_id[nr, nc] = cid
                                stack.append((nr, nc))

                comps.append({"color": int(color), "cells": cells})

    # ---------- component properties ----------
    for comp in comps:
        cells = comp["cells"]
        s = set(cells)
        comp["cell_set"] = s

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]

        is_line = False
        line_info = None
        if len(cells) >= 2:
            # horizontal line
            if min(rows) == max(rows):
                if max(cols) - min(cols) + 1 == len(cells):
                    is_line = True
                    # ('H', row, min_col, max_col)
                    line_info = ("H", min(rows), min(cols), max(cols))
            # vertical line
            if min(cols) == max(cols):
                if max(rows) - min(rows) + 1 == len(cells):
                    is_line = True
                    # ('V', col, min_row, max_row)
                    line_info = ("V", min(cols), min(rows), max(rows))

        comp["is_line"] = is_line
        comp["line_info"] = line_info

        # Thin = no 2x2 solid block. This distinguishes rope-like paths from blocks.
        thin = True
        for r, c in cells:
            if (r + 1, c) in s and (r, c + 1) in s and (r + 1, c + 1) in s:
                thin = False
                break
        comp["is_thin"] = thin

        # 8-neighbor degree, used to prefer true rope endpoints.
        degree = {}
        for r, c in cells:
            cnt = 0
            for dr, dc in n8:
                if (r + dr, c + dc) in s:
                    cnt += 1
            degree[(r, c)] = cnt
        comp["degree"] = degree

    # ---------- identify ropes attached to line endpoints ----------
    tasks = []
    rope_ids = []
    n4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i, comp in enumerate(comps):
        if comp["is_line"] or not comp["is_thin"] or len(comp["cells"]) < 2:
            continue

        candidates = []
        for r, c in comp["cells"]:
            for dr, dc in n4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue

                aid = comp_id[nr, nc]
                if aid == -1 or aid == i:
                    continue
                anchor = comps[aid]
                if not anchor["is_line"]:
                    continue

                info = anchor["line_info"]
                if info[0] == "V":
                    col, min_r, max_r = info[1], info[2], info[3]
                    d_top = abs(nr - min_r)
                    d_bot = abs(nr - max_r)
                    dist = min(d_top, d_bot)

                    # Only use true endpoint attachments.
                    if dist != 0:
                        continue

                    if d_top <= d_bot:
                        endpoint = (min_r, col)
                        D = (-1, 0)
                    else:
                        endpoint = (max_r, col)
                        D = (1, 0)
                    exit_coord = col

                else:
                    row, min_c, max_c = info[1], info[2], info[3]
                    d_left = abs(nc - min_c)
                    d_right = abs(nc - max_c)
                    dist = min(d_left, d_right)

                    if dist != 0:
                        continue

                    if d_left <= d_right:
                        endpoint = (row, min_c)
                        D = (0, -1)
                    else:
                        endpoint = (row, max_c)
                        D = (0, 1)
                    exit_coord = row

                cheb = max(abs(r - endpoint[0]), abs(c - endpoint[1]))
                deg = comp["degree"][(r, c)]
                candidates.append((dist, deg, cheb, aid, (r, c), (nr, nc), endpoint, D, exit_coord))

        if not candidates:
            continue

        candidates.sort()
        _, _, _, anchor_id, start, _, _, D, exit_coord = candidates[0]

        # Determine which side of the anchor line the rope is on.
        if D[1] == 0:  # vertical anchor
            anchor_col = exit_coord
            left = sum(1 for rr, cc in comp["cells"] if cc < anchor_col)
            right = sum(1 for rr, cc in comp["cells"] if cc > anchor_col)
            if left > right:
                side = -1
            elif right > left:
                side = 1
            else:
                side = 1 if start[1] > anchor_col else -1
        else:          # horizontal anchor
            anchor_row = exit_coord
            above = sum(1 for rr, cc in comp["cells"] if rr < anchor_row)
            below = sum(1 for rr, cc in comp["cells"] if rr > anchor_row)
            if above > below:
                side = -1
            elif below > above:
                side = 1
            else:
                side = 1 if start[0] > anchor_row else -1

        tasks.append({
            "comp": i,
            "color": comp["color"],
            "start": start,
            "anchor": anchor_id,
            "D": D,
            "exit": exit_coord,
            "side": side,
            "length": len(comp["cells"]),
        })
        rope_ids.append(i)

    if not tasks:
        return grid.copy()

    rope_id_set = set(rope_ids)

    # All moving rope cells are cleared; only fixed components block paths.
    moving_mask = np.zeros((H, W), dtype=bool)
    for i in rope_ids:
        for r, c in comps[i]["cells"]:
            moving_mask[r, c] = True
    fixed_mask = (grid != bg) & (~moving_mask)

    # ---------- target selection ----------
    def choose_target(task):
        own = task["comp"]
        anchor = task["anchor"]
        dr, dc = task["D"]
        sr, sc = task["start"]

        best_cells = None
        best_score = None

        for j, comp in enumerate(comps):
            if j == own or j == anchor:
                continue
            if j in rope_id_set:
                continue

            local_best = None
            for r, c in comp["cells"]:
                if dr == -1:
                    dot = sr - r
                    if dot <= 0:
                        continue
                    perp = abs(c - sc)
                elif dr == 1:
                    dot = r - sr
                    if dot <= 0:
                        continue
                    perp = abs(c - sc)
                elif dc == -1:
                    dot = sc - c
                    if dot <= 0:
                        continue
                    perp = abs(r - sr)
                else:
                    dot = c - sc
                    if dot <= 0:
                        continue
                    perp = abs(r - sr)

                score = (max(dot, perp), dot, perp)
                if local_best is None or score < local_best:
                    local_best = score

            if local_best is not None:
                if best_score is None or local_best < best_score:
                    best_score = local_best
                    best_cells = comp["cells"]

        return best_cells

    # ---------- guide generator ----------
    def make_get_guide(target_cells, dr, dc, side, exit_coord):
        if target_cells is None:
            return lambda _: exit_coord

        if dr != 0:
            by_row = {}
            for r, c in target_cells:
                by_row.setdefault(r, []).append(c)

            boundary = [None] * H
            for r, cs in by_row.items():
                if side == -1:
                    b = min(cs) - 1
                else:
                    b = max(cs) + 1
                b = max(0, min(W - 1, b))
                boundary[r] = b

            next_obs = [None] * H
            if dr == -1:
                last = None
                for r in range(H):
                    if boundary[r] is not None:
                        last = r
                    next_obs[r] = last
            else:
                last = None
                for r in range(H - 1, -1, -1):
                    if boundary[r] is not None:
                        last = r
                    next_obs[r] = last

            def get(row):
                obs = next_obs[row]
                if obs is not None:
                    return boundary[obs]
                return exit_coord

            return get

        else:
            by_col = {}
            for r, c in target_cells:
                by_col.setdefault(c, []).append(r)

            boundary = [None] * W
            for c, rs in by_col.items():
                if side == -1:
                    b = min(rs) - 1
                else:
                    b = max(rs) + 1
                b = max(0, min(H - 1, b))
                boundary[c] = b

            next_obs = [None] * W
            if dc == -1:
                last = None
                for c in range(W):
                    if boundary[c] is not None:
                        last = c
                    next_obs[c] = last
            else:
                last = None
                for c in range(W - 1, -1, -1):
                    if boundary[c] is not None:
                        last = c
                    next_obs[c] = last

            def get(col):
                obs = next_obs[col]
                if obs is not None:
                    return boundary[obs]
                return exit_coord

            return get

    # ---------- path generation ----------
    def generate_path(task):
        r, c = task["start"]
        steps = task["length"] - 1
        dr, dc = task["D"]

        target_cells = choose_target(task)
        get_guide = make_get_guide(target_cells, dr, dc, task["side"], task["exit"])

        path = [(r, c)]
        path_set = {(r, c)}

        for _ in range(steps):
            best = None

            if dr != 0:
                nr_prog = r + dr
                if nr_prog < 0 or nr_prog >= H:
                    break

                for nr in (nr_prog, r):
                    for nc in (c - 1, c, c + 1):
                        if nr == r and nc == c:
                            continue
                        if not (0 <= nr < H and 0 <= nc < W):
                            continue
                        if fixed_mask[nr, nc] or (nr, nc) in path_set:
                            continue

                        progress = 0 if nr == nr_prog else 1
                        guide = get_guide(nr)
                        side_pref = 0
                        if task["side"] != 0:
                            side_pref = 1 if (nc - c) * task["side"] > 0 else 2

                        score = (progress, abs(nc - guide), abs(nc - c), side_pref)
                        if best is None or score < best[0]:
                            best = (score, (nr, nc))

            else:
                nc_prog = c + dc
                if nc_prog < 0 or nc_prog >= W:
                    break

                for nc in (nc_prog, c):
                    for nr in (r - 1, r, r + 1):
                        if nr == r and nc == c:
                            continue
                        if not (0 <= nr < H and 0 <= nc < W):
                            continue
                        if fixed_mask[nr, nc] or (nr, nc) in path_set:
                            continue

                        progress = 0 if nc == nc_prog else 1
                        guide = get_guide(nc)
                        side_pref = 0
                        if task["side"] != 0:
                            side_pref = 1 if (nr - r) * task["side"] > 0 else 2

                        score = (progress, abs(nr - guide), abs(nr - r), side_pref)
                        if best is None or score < best[0]:
                            best = (score, (nr, nc))

            if best is None:
                break

            r, c = best[1]
            path.append((r, c))
            path_set.add((r, c))

        return task["color"], path

    # ---------- build output ----------
    out = grid.copy()

    # Clear all rope components.
    for i in rope_ids:
        for r, c in comps[i]["cells"]:
            out[r, c] = bg

    # Draw new rope paths.
    for task in tasks:
        color, path = generate_path(task)
        for r, c in path:
            if not fixed_mask[r, c]:
                out[r, c] = color

    return out