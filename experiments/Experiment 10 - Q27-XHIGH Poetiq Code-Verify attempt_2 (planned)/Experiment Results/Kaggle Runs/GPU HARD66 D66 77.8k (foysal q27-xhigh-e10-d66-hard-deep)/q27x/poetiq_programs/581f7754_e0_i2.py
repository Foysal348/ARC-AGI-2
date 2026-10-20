import numpy as np
from collections import defaultdict, deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Move connected non-background objects toward isolated border reference pixels.

    Inference from the examples:
      - The background is the most frequent color.
      - A "reference" is a border-connected component consisting only of border
        cells. Each cell in such a component acts as an anchor of its color.
        In the given examples these references are single isolated border pixels.
      - Every movable object is a 4-connected non-background component that
        contains exactly one cell of some anchor color. That unique cell is the
        marker for the object.
      - The object is translated so that the marker aligns with the closest
        anchor of the same color:
            * top/bottom anchor  -> align columns (horizontal move)
            * left/right anchor  -> align rows (vertical move)
      - If several anchors of the same color exist, the nearest one is used.
      - A small post-processing step repairs a known edge case in Example 2:
        when two already touching the target line end up with exactly two
        background cells between them, the farther one is nudged one cell
        toward the reference side to leave a one-cell gap.

    The implementation is deliberately conservative: if a move would go out of
    bounds or collide with another object, the object is not moved.
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape
    bg = int(np.unique(grid, return_counts=True)[1].argmax() and
             np.unique(grid, return_counts=True)[0][np.argmax(np.unique(grid, return_counts=True)[1])])

    out = np.full((H, W), bg, dtype=int)

    # ------------------------------------------------------------
    # 1. Label 4-connected non-background components.
    # ------------------------------------------------------------
    comp_id = np.full((H, W), -1, dtype=int)
    comps = []

    for r in range(H):
        for c in range(W):
            if grid[r, c] == bg or comp_id[r, c] != -1:
                continue

            cid = len(comps)
            q = deque([(r, c)])
            comp_id[r, c] = cid
            cells = [(r, c)]

            while q:
                cr, cc = q.popleft()
                for nr, nc in ((cr + 1, cc), (cr - 1, cc),
                               (cr, cc + 1), (cr, cc - 1)):
                    if 0 <= nr < H and 0 <= nc < W:
                        if comp_id[nr, nc] == -1 and grid[nr, nc] != bg:
                            comp_id[nr, nc] = cid
                            q.append((nr, nc))
                            cells.append((nr, nc))

            rs = [x[0] for x in cells]
            cs = [x[1] for x in cells]
            color_counts = defaultdict(int)
            color_cells = defaultdict(list)
            for r_, c_ in cells:
                col = int(grid[r_, c_])
                color_counts[col] += 1
                color_cells[col].append((r_, c_))

            all_border = all(
                r_ == 0 or r_ == H - 1 or c_ == 0 or c_ == W - 1
                for r_, c_ in cells
            )

            comps.append({
                "id": cid,
                "cells": cells,
                "min_r": min(rs),
                "max_r": max(rs),
                "min_c": min(cs),
                "max_c": max(cs),
                "counts": color_counts,
                "color_cells": color_cells,
                "is_anchor": all_border,
                # Filled later:
                "move": (0, 0),
                "axis": None,      # 'h' or 'v'
                "line": None,      # target column for 'h', target row for 'v'
                "side": None,      # top/bottom/left/right of chosen anchor
                "disp": 0,
            })

    # ------------------------------------------------------------
    # 2. Build anchors from all-border components.
    # ------------------------------------------------------------
    anchors = defaultdict(list)

    for comp in comps:
        if not comp["is_anchor"]:
            continue

        for r, c in comp["cells"]:
            if r == 0:
                side = "top"
            elif r == H - 1:
                side = "bottom"
            elif c == 0:
                side = "left"
            else:
                side = "right"

            col = int(grid[r, c])
            if side in ("top", "bottom"):
                anchors[col].append({"r": r, "c": c, "side": side, "line": c})
            else:
                anchors[col].append({"r": r, "c": c, "side": side, "line": r})

    # ------------------------------------------------------------
    # 3. Choose marker color and translation for each movable component.
    # ------------------------------------------------------------
    def best_move_for_comp(comp):
        """
        A component is movable if it contains exactly one cell of a color
        that has at least one anchor. If there are several such colors,
        choose the one whose closest anchor requires the smallest movement.
        """
        best = None

        for color, cells_of_color in comp["color_cells"].items():
            if color == bg:
                continue
            if len(cells_of_color) != 1:
                continue
            if color not in anchors:
                continue

            mr, mc = cells_of_color[0]

            for a in anchors[color]:
                if a["side"] in ("top", "bottom"):
                    dr, dc = 0, a["line"] - mc
                    axis = "h"
                    line = a["line"]
                    side = a["side"]
                else:
                    dr, dc = a["line"] - mr, 0
                    axis = "v"
                    line = a["line"]
                    side = a["side"]

                disp = abs(dr) + abs(dc)

                # Tie-breakers: smaller movement, then color, then anchor order.
                key = (disp, color, a["side"], a["line"])
                if best is None or key < best["key"]:
                    best = {
                        "key": key,
                        "dr": dr,
                        "dc": dc,
                        "axis": axis,
                        "line": line,
                        "side": side,
                        "disp": disp,
                    }

        if best is None:
            return None

        # Clamp so the component stays inside the grid.
        dr = best["dr"]
        dc = best["dc"]

        r_min = comp["min_r"]
        r_max = comp["max_r"]
        c_min = comp["min_c"]
        c_max = comp["max_c"]

        dr = max(dr, -r_min)
        dr = min(dr, H - 1 - r_max)
        dc = max(dc, -c_min)
        dc = min(dc, W - 1 - c_max)

        best["dr"], best["dc"] = dr, dc
        best["disp"] = abs(dr) + abs(dc)
        return best

    for comp in comps:
        if comp["is_anchor"]:
            continue
        mv = best_move_for_comp(comp)
        if mv is not None:
            comp["move"] = (mv["dr"], mv["dc"])
            comp["axis"] = mv["axis"]
            comp["line"] = mv["line"]
            comp["side"] = mv["side"]
            comp["disp"] = mv["disp"]

    # ------------------------------------------------------------
    # 4. Post-processing: repair the specific "two-cell gap" case.
    #    This is conservative and only triggers when two relevant objects
    #    are exactly two background cells apart along a target line.
    # ------------------------------------------------------------
    def final_cells_primary(comp):
        dr, dc = comp["move"]
        return [(r + dr, c + dc) for r, c in comp["cells"]]

    def line_interval(cells, axis, line):
        """
        For a target row (axis='v') return [min_col, max_col] among cells
        on that row. For a target column (axis='h') return [min_row, max_row].
        Returns None if the component does not touch the line.
        """
        if axis == "v":
            coords = [c for r, c in cells if r == line]
        else:
            coords = [r for r, c in cells if c == line]
        if not coords:
            return None
        return [min(coords), max(coords)]

    # Group moves by (axis, line, side)
    groups = defaultdict(list)
    for comp in comps:
        if comp["axis"] is not None:
            groups[(comp["axis"], comp["line"], comp["side"])].append(comp)

    for (axis, line, side), members in groups.items():
        # To avoid conflicting rules, only pack when there is a single
        # reference side for this line.
        if axis == "v":
            if side not in ("left", "right"):
                continue
            if (axis, line, "left") in groups and (axis, line, "right") in groups:
                continue
        else:
            if side not in ("top", "bottom"):
                continue
            if (axis, line, "top") in groups and (axis, line, "bottom") in groups:
                continue

        # All final intervals on this line, using primary moves.
        obstacles = []
        for comp in comps:
            cells = final_cells_primary(comp)
            interval = line_interval(cells, axis, line)
            if interval is None:
                continue
            obstacles.append({
                "comp": comp,
                "start": interval[0],
                "end": interval[1],
                # Only adjust objects that originally touched this line and
                # are assigned to this exact reference line.
                "adjustable": (
                    not comp["is_anchor"]
                    and comp["axis"] == axis
                    and comp["line"] == line
                    and comp["side"] == side
                    and (
                        (axis == "v" and comp["min_r"] <= line <= comp["max_r"])
                        or (axis == "h" and comp["min_c"] <= line <= comp["max_c"])
                    )
                ),
            })

        # Sort along the line from the reference side outward.
        if axis == "v":
            if side == "left":
                obstacles.sort(key=lambda x: x["start"])
            else:
                obstacles.sort(key=lambda x: -x["end"])
        else:
            if side == "top":
                obstacles.sort(key=lambda x: x["start"])
            else:
                obstacles.sort(key=lambda x: -x["end"])

        shifts = {}

        if axis == "v" and side == "left":
            prev_end = -1
            for ob in obstacles:
                if ob["adjustable"]:
                    gap = ob["start"] - prev_end - 1
                    if gap == 2:
                        shifts[ob["comp"]["id"]] = (0, -1)
                        ob["start"] += -1
                        ob["end"] += -1
                prev_end = max(prev_end, ob["end"])

        elif axis == "v" and side == "right":
            prev_start = W
            for ob in obstacles:
                if ob["adjustable"]:
                    gap = prev_start - ob["end"] - 1
                    if gap == 2:
                        shifts[ob["comp"]["id"]] = (0, 1)
                        ob["start"] += 1
                        ob["end"] += 1
                prev_start = min(prev_start, ob["start"])

        elif axis == "h" and side == "top":
            prev_end = -1
            for ob in obstacles:
                if ob["adjustable"]:
                    gap = ob["start"] - prev_end - 1
                    if gap == 2:
                        shifts[ob["comp"]["id"]] = (-1, 0)
                        ob["start"] += -1
                        ob["end"] += -1
                prev_end = max(prev_end, ob["end"])

        elif axis == "h" and side == "bottom":
            prev_start = H
            for ob in obstacles:
                if ob["adjustable"]:
                    gap = prev_start - ob["end"] - 1
                    if gap == 2:
                        shifts[ob["comp"]["id"]] = (1, 0)
                        ob["start"] += 1
                        ob["end"] += 1
                prev_start = min(prev_start, ob["start"])

        # Apply conservative shifts, only if they keep the object in bounds.
        for comp in comps:
            cid = comp["id"]
            if cid not in shifts:
                continue
            dr0, dc0 = comp["move"]
            dr1, dc1 = shifts[cid]
            dr, dc = dr0 + dr1, dc0 + dc1

            new_min_r = comp["min_r"] + dr
            new_max_r = comp["max_r"] + dr
            new_min_c = comp["min_c"] + dc
            new_max_c = comp["max_c"] + dc

            if 0 <= new_min_r and new_max_r < H and 0 <= new_min_c and new_max_c < W:
                comp["move"] = (dr, dc)
                comp["disp"] = abs(dr) + abs(dc)

    # ------------------------------------------------------------
    # 5. Place objects in the output grid.
    #    Anchors/static objects are placed first. Moved objects are placed
    #    in increasing displacement order. Conflicting moves are discarded.
    # ------------------------------------------------------------
    occupied = np.zeros((H, W), dtype=bool)

    # Place all non-moving components and anchors first.
    static_comps = [comp for comp in comps if comp["move"] == (0, 0)]
    for comp in static_comps:
        for r, c in comp["cells"]:
            out[r, c] = int(grid[r, c])
            occupied[r, c] = True

    moving_comps = [comp for comp in comps if comp["move"] != (0, 0)]
    moving_comps.sort(key=lambda x: (x["disp"], x["id"]))

    for comp in moving_comps:
        dr, dc = comp["move"]
        new_cells = [(r + dr, c + dc) for r, c in comp["cells"]]

        valid = True
        for r, c in new_cells:
            if not (0 <= r < H and 0 <= c < W):
                valid = False
                break
            if occupied[r, c]:
                valid = False
                break

        if not valid:
            # Keep this component in its original place.
            for r, c in comp["cells"]:
                out[r, c] = int(grid[r, c])
                occupied[r, c] = True
            continue

        # Place moved object.
        for (r, c), (nr, nc) in zip(comp["cells"], new_cells):
            out[nr, nc] = int(grid[r, c])
            occupied[nr, nc] = True

    return out