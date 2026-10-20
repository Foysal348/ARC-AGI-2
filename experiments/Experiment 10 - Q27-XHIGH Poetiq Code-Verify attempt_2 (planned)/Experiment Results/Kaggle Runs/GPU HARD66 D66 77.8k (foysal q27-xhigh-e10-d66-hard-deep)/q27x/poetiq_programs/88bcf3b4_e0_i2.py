import numpy as np
from collections import deque, Counter


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform "snake" objects attached to straight anchor lines.

    Rule inferred from the examples:
    1. Find the background color as the most frequent cell value.
    2. Split all non-background cells into 8-connected components.
    3. A moving component is a non-straight component that has an endpoint
       orthogonally adjacent to an endpoint of a straight anchor line.
    4. For that attachment:
         - x-coordinate grows away from the anchor endpoint along the anchor line.
         - y-coordinate grows to the side of the anchor where the snake endpoint sits.
       The snake endpoint is at local (0, 1).
    5. The target is the nearest static component ahead in the positive x half-plane
       and not behind the anchor side (y >= 0).
    6. If the target's farthest side from the anchor is y = Y, the new snake path is:
         - diagonal approach to y = Y + 1,
         - horizontal stretch along y = Y + 1 over the target's far-side x-range,
         - diagonal exit away from the target.
       The path is stopped by grid bounds or by static non-moving cells.
    7. The drawn path length is min(original snake length, valid canonical path length).
       The original snake cells are erased, then the new path is drawn.
    """
    grid = np.array(grid)
    if grid.ndim != 2:
        return grid.copy()

    H, W = grid.shape
    if H == 0 or W == 0:
        return grid.copy()

    bg = Counter(grid.flatten()).most_common(1)[0][0]

    # ------------------------------------------------------------------
    # Connected components (8-connected, same color)
    # ------------------------------------------------------------------
    visited = np.zeros((H, W), dtype=bool)
    comps = []
    cell_to_comp = {}

    for r in range(H):
        for c in range(W):
            if visited[r, c] or grid[r, c] == bg:
                continue

            color = int(grid[r, c])
            q = deque([(r, c)])
            visited[r, c] = True
            cells = []

            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))

                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if not visited[nr, nc] and grid[nr, nc] == color:
                                visited[nr, nc] = True
                                q.append((nr, nc))

            cid = len(comps)
            cell_set = set(cells)

            rows = {r0 for r0, _ in cells}
            cols = {c0 for _, c0 in cells}

            straight = (len(rows) == 1 or len(cols) == 1)
            horizontal = (len(rows) == 1 and len(cols) > 1)
            vertical = (len(cols) == 1 and len(rows) > 1)

            min_row = min(rows)
            max_row = max(rows)
            min_col = min(cols)
            max_col = max(cols)

            if straight:
                if horizontal:
                    endpoints = [(min_row, min_col), (min_row, max_col)]
                elif vertical:
                    endpoints = [(min_row, min_col), (max_row, max_col)]
                else:
                    endpoints = [(min_row, min_col)]
            else:
                endpoints = []

            comp = {
                "id": cid,
                "color": color,
                "cells": cells,
                "set": cell_set,
                "len": len(cells),
                "straight": straight,
                "horizontal": horizontal,
                "vertical": vertical,
                "min_row": min_row,
                "max_row": max_row,
                "min_col": min_col,
                "max_col": max_col,
                "endpoints": endpoints,
            }
            comps.append(comp)

            for cell in cells:
                cell_to_comp[cell] = cid

    # ------------------------------------------------------------------
    # Helper: 8-connected endpoints of a component
    # ------------------------------------------------------------------
    def endpoint_cells(comp):
        s = comp["set"]
        eps = []
        for r, c in comp["cells"]:
            deg = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if (r + dr, c + dc) in s:
                        deg += 1
            if deg == 1:
                eps.append((r, c))
        return eps

    # ------------------------------------------------------------------
    # Local coordinate helpers
    # ------------------------------------------------------------------
    def to_local(orient, r, c):
        pr, pc = orient["pr"], orient["pc"]

        if orient["x_axis"] == "row":
            x = orient["x_sign"] * (r - pr)
        else:
            x = orient["x_sign"] * (c - pc)

        if orient["y_axis"] == "row":
            y = orient["y_sign"] * (r - pr)
        else:
            y = orient["y_sign"] * (c - pc)

        return x, y

    def to_global(orient, x, y):
        pr, pc = orient["pr"], orient["pc"]
        r, c = pr, pc

        if orient["x_axis"] == "row":
            r += orient["x_sign"] * x
        else:
            c += orient["x_sign"] * x

        if orient["y_axis"] == "row":
            r += orient["y_sign"] * y
        else:
            c += orient["y_sign"] * y

        return r, c

    # ------------------------------------------------------------------
    # Find possible moving components and their attachments to anchors
    # ------------------------------------------------------------------
    candidates = []
    potential_moving_ids = set()

    for C in comps:
        # Moving objects are non-straight and have at least one snake endpoint.
        if C["len"] <= 1 or C["straight"]:
            continue

        for h in endpoint_cells(C):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = (h[0] + dr, h[1] + dc)
                if p not in cell_to_comp:
                    continue

                A_id = cell_to_comp[p]
                if A_id == C["id"]:
                    continue

                A = comps[A_id]
                if A["len"] < 2 or not A["straight"]:
                    continue
                if p not in A["endpoints"]:
                    continue

                # The snake endpoint must be perpendicular to the anchor line.
                if A["horizontal"]:
                    # Anchor is horizontal. Attachment must be directly above/below the endpoint.
                    if h[1] != p[1] or p[0] != A["min_row"]:
                        continue

                    if p[1] == A["min_col"]:
                        x_sign = -1  # away direction is left
                    elif p[1] == A["max_col"]:
                        x_sign = 1   # away direction is right
                    else:
                        continue

                    if h[0] < p[0]:
                        y_sign = -1  # snake is above the anchor
                    elif h[0] > p[0]:
                        y_sign = 1   # snake is below the anchor
                    else:
                        continue

                    x_axis = "col"
                    y_axis = "row"

                else:
                    # Anchor is vertical. Attachment must be directly left/right of the endpoint.
                    if h[0] != p[0] or p[1] != A["min_col"]:
                        continue

                    if p[0] == A["min_row"]:
                        x_sign = -1  # away direction is up
                    elif p[0] == A["max_row"]:
                        x_sign = 1   # away direction is down
                    else:
                        continue

                    if h[1] < p[1]:
                        y_sign = -1  # snake is left of the anchor
                    elif h[1] > p[1]:
                        y_sign = 1   # snake is right of the anchor
                    else:
                        continue

                    x_axis = "row"
                    y_axis = "col"

                orient = {
                    "pr": p[0],
                    "pc": p[1],
                    "x_axis": x_axis,
                    "x_sign": x_sign,
                    "y_axis": y_axis,
                    "y_sign": y_sign,
                }

                candidates.append({
                    "moving": C,
                    "anchor": A,
                    "orient": orient,
                    "H": h,
                    "P": p,
                })
                potential_moving_ids.add(C["id"])

    if not candidates:
        return grid.copy()

    # ------------------------------------------------------------------
    # Target evaluation
    # ------------------------------------------------------------------
    def evaluate_target(orient, D):
        eligible = []
        for r, c in D["cells"]:
            x, y = to_local(orient, r, c)
            if x > 0 and y >= 0:
                eligible.append((x, y))

        if not eligible:
            return None

        Y = max(y for _, y in eligible)
        side_cells = [(x, y) for x, y in eligible if y == Y]
        xmin = min(x for x, _ in side_cells)
        xmax = max(x for x, _ in side_cells)
        min_x = min(x for x, _ in eligible)

        return {
            "component": D,
            "Y": Y,
            "xmin": xmin,
            "xmax": xmax,
            "min_x": min_x,
        }

    # ------------------------------------------------------------------
    # Generate canonical path in local coordinates
    # ------------------------------------------------------------------
    def generate_path(orient, target_info, forbidden, max_steps=None):
        Yp = target_info["Y"] + 1

        # Need at least Yp-1 steps to climb from y=1 to y=Yp.
        top_start = max(target_info["xmin"], Yp - 1, 0)
        top_end = max(target_info["xmax"], top_start)

        points = []
        x = 0
        limit = max_steps if max_steps is not None else 4 * (H + W) + 100

        while len(points) < limit:
            if x <= top_start:
                y = min(x + 1, Yp)
            elif x <= top_end:
                y = Yp
            else:
                y = Yp - (x - top_end)

            if y < 0:
                break

            r, c = to_global(orient, x, y)
            if r < 0 or r >= H or c < 0 or c >= W:
                break
            if (r, c) in forbidden:
                break

            points.append((r, c))
            x += 1

        return points

    # ------------------------------------------------------------------
    # For each moving component choose the best anchor/target pair
    # ------------------------------------------------------------------
    best_for_moving = {}

    for cand in candidates:
        C = cand["moving"]
        orient = cand["orient"]

        best_target = None
        best_score = None

        for D in comps:
            # Exclude the moving component itself, its anchor, and other moving snakes.
            if D["id"] == C["id"]:
                continue
            if D["id"] == cand["anchor"]["id"]:
                continue
            if D["id"] in potential_moving_ids:
                continue

            info = evaluate_target(orient, D)
            if info is None:
                continue

            # Prefer the nearest eligible static object ahead.
            score = (info["min_x"], info["xmin"], info["xmax"], -info["Y"])
            if best_score is None or score < best_score:
                best_score = score
                best_target = info

        if best_target is None:
            continue

        if C["id"] not in best_for_moving or best_score < best_for_moving[C["id"]]["score"]:
            best_for_moving[C["id"]] = {
                "candidate": cand,
                "target": best_target,
                "score": best_score,
            }

    if not best_for_moving:
        return grid.copy()

    # ------------------------------------------------------------------
    # Build forbidden static cells: everything that will remain in output.
    # Selected moving components are erased, so their original cells are allowed.
    # ------------------------------------------------------------------
    selected_moving_ids = set(best_for_moving.keys())
    forbidden_static = set()
    for D in comps:
        if D["id"] not in selected_moving_ids:
            forbidden_static.update(D["cells"])

    # ------------------------------------------------------------------
    # Final transformations
    # ------------------------------------------------------------------
    transformations = []

    for cid, data in best_for_moving.items():
        cand = data["candidate"]
        C = cand["moving"]
        orient = cand["orient"]
        target_info = data["target"]

        points = generate_path(orient, target_info, forbidden_static)
        if not points:
            continue

        draw_len = min(C["len"], len(points))
        if draw_len <= 0:
            continue

        transformations.append({
            "color": C["color"],
            "old_cells": C["cells"],
            "new_cells": points[:draw_len],
            "score": data["score"],
        })

    if not transformations:
        return grid.copy()

    # Deterministic drawing order.
    transformations.sort(key=lambda t: (t["score"][0], t["score"][1], t["score"][2], len(t["new_cells"])))

    out = grid.copy()

    # Erase all selected moving components first.
    for t in transformations:
        for r, c in t["old_cells"]:
            out[r, c] = bg

    # Draw the new snake paths.
    for t in transformations:
        for r, c in t["new_cells"]:
            out[r, c] = t["color"]

    return out