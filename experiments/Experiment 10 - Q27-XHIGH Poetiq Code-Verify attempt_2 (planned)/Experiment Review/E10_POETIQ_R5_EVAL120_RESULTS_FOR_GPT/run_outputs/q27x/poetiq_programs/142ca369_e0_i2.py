import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by treating 3-cell L-shapes as initial diagonal emitters,
    straight 3-cell lines as diagonal mirrors, and single pixels as same-color
    targets that reflect a beam.

    Main rules inferred from the examples:
    - Original pixels are always preserved.
    - Each L-shaped 3-cell component emits from its elbow in the diagonal
      direction away from its missing corner.
    - A diagonal beam that hits a straight 3-cell line causes that line to emit
      its own color by reflecting the incoming diagonal:
          vertical line: flip column direction
          horizontal line: flip row direction
      The emitted beam starts from the appropriate end of the line.
    - A diagonal beam that can aim at a same-color single pixel stops just
      before that pixel and reflects once, flipping the axis along which it
      made its final straight approach.
    - Unhit straight lines are given a fallback beam:
          1) if they have a non-parallel same-color line partner, point away
             from that partner;
          2) otherwise point at the nearest first component found on one of
             their diagonal rays;
          3) otherwise point to the border direction with maximum clear path.
    - All emitted cells are resolved by minimum distance from the beam origin;
      original pixels have distance 0 and always win.  A beam is drawn only
      until its first non-owned cell, so it visually stops at conflicts.
    """
    if grid.size == 0:
        return grid

    H, W = grid.shape
    out = grid.copy()

    # ------------------------------------------------------------------
    # 1. Extract 4-connected components
    # ------------------------------------------------------------------
    comp_id = -np.ones((H, W), dtype=int)
    comps = []

    for r in range(H):
        for c in range(W):
            if grid[r, c] != 0 and comp_id[r, c] == -1:
                color = int(grid[r, c])
                q = deque([(r, c)])
                cells = []
                cid = len(comps)
                comp_id[r, c] = cid

                while q:
                    x, y = q.popleft()
                    cells.append((x, y))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if comp_id[nx, ny] == -1 and grid[nx, ny] == color:
                                comp_id[nx, ny] = cid
                                q.append((nx, ny))

                rs = [p[0] for p in cells]
                cs = [p[1] for p in cells]
                r0, r1 = min(rs), max(rs)
                c0, c1 = min(cs), max(cs)

                comp = {
                    "id": cid,
                    "color": color,
                    "cells": cells,
                    "r0": r0, "r1": r1,
                    "c0": c0, "c1": c1,
                    "h": r1 - r0 + 1,
                    "w": c1 - c0 + 1,
                    "size": len(cells),
                    "center_r": (r0 + r1) / 2.0,
                    "center_c": (c0 + c1) / 2.0,
                    "type": None,
                    "start": None,
                    "dir": None,
                }

                if comp["size"] == 1:
                    comp["type"] = "S"
                    comp["cell"] = cells[0]

                elif comp["size"] == 3 and comp["h"] == 2 and comp["w"] == 2:
                    comp["type"] = "L"
                    cellset = set(cells)
                    mr = mc = None
                    for rr in (r0, r1):
                        for cc in (c0, c1):
                            if (rr, cc) not in cellset:
                                mr, mc = rr, cc

                    er = r1 if mr == r0 else r0
                    ec = c1 if mc == c0 else c0
                    dr = 1 if mr < er else -1
                    dc = 1 if mc < ec else -1

                    comp["start"] = (er, ec)
                    comp["dir"] = (dr, dc)

                elif comp["size"] == 3 and (
                    (comp["h"] == 1 and comp["w"] == 3) or
                    (comp["h"] == 3 and comp["w"] == 1)
                ):
                    comp["type"] = "Iv" if comp["h"] == 3 else "Ih"

                else:
                    comp["type"] = "O"

                comps.append(comp)

    if not comps:
        return out

    # Singles by color, used for same-color aiming/reflection.
    singles_by_color = {}
    for comp in comps:
        if comp["type"] == "S":
            singles_by_color.setdefault(comp["color"], []).append(comp)

    # ------------------------------------------------------------------
    # 2. Helpers
    # ------------------------------------------------------------------
    def in_bounds(r, c):
        return 0 <= r < H and 0 <= c < W

    def is_original(r, c):
        return grid[r, c] != 0

    def dir_priority(d):
        # Clockwise order that matched the observed tie-breaks.
        return {
            (1, 1): 0,
            (1, -1): 1,
            (-1, -1): 2,
            (-1, 1): 3,
        }.get(d, 4)

    def line_reflect(comp, din):
        """Reflect an incoming diagonal direction off a straight 3-cell line."""
        dr, dc = din
        if comp["type"] == "Iv":
            dout = (dr, -dc)
            r = comp["r0"] if dout[0] > 0 else comp["r1"]
            c = comp["c0"]
        else:  # horizontal
            dout = (-dr, dc)
            r = comp["r0"]
            c = comp["c0"] if dout[1] > 0 else comp["c1"]
        return (r, c), dout

    def fallback_line_beam(comp):
        """Choose a beam for an unhit straight line component."""
        # 1) Non-parallel same-color line partner: point away from it.
        partner = None
        for other in comps:
            if (other["id"] != comp["id"] and
                other["type"] == comp["type"] and
                other["color"] == comp["color"]):
                partner = other
                break

        if partner is not None:
            dr = comp["center_r"] - partner["center_r"]
            dc = comp["center_c"] - partner["center_c"]
            if dr != 0 and dc != 0:
                d = (1 if dr > 0 else -1, 1 if dc > 0 else -1)
                if comp["type"] == "Iv":
                    r = comp["r0"] if d[0] > 0 else comp["r1"]
                    c = comp["c0"]
                else:
                    r = comp["r0"]
                    c = comp["c0"] if d[1] > 0 else comp["c1"]
                return (r, c), d

        # 2) Aim at the nearest first component found on a diagonal ray.
        best = None
        best_score = None
        for sr, sc in comp["cells"]:
            for d in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
                dr, dc = d
                step = 0
                first_cid = None
                while True:
                    step += 1
                    nr, nc = sr + dr * step, sc + dc * step
                    if not in_bounds(nr, nc):
                        break
                    cid = int(comp_id[nr, nc])
                    if cid != -1 and cid != comp["id"]:
                        first_cid = cid
                        break
                if first_cid is None:
                    continue
                target = comps[first_cid]
                center_dist = abs((sr + sc) / 2.0 - (comp["center_r"] + comp["center_c"]) / 2.0)
                score = (step, dir_priority(d), center_dist, target["color"], target["id"])
                if best_score is None or score < best_score:
                    best_score = score
                    best = ((sr, sc), d)

        if best is not None:
            return best

        # 3) Point to the border with the longest clear diagonal path.
        best = None
        best_score = None
        for d in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            dr, dc = d
            if comp["type"] == "Iv":
                r = comp["r0"] if dr > 0 else comp["r1"]
                c = comp["c0"]
            else:
                r = comp["r0"]
                c = comp["c0"] if dc > 0 else comp["c1"]

            row_steps = (H - 1 - r) if dr > 0 else r
            col_steps = (W - 1 - c) if dc > 0 else c
            length = min(row_steps, col_steps)
            score = (-length, dir_priority(d))
            if best_score is None or score < best_score:
                best_score = score
                best = ((r, c), d)

        # Absolute fallback, should rarely be needed.
        if best is None:
            if comp["type"] == "Iv":
                return (comp["r0"], comp["c0"]), (1, 1)
            return (comp["r0"], comp["c0"]), (1, 1)

    def choose_single_target(origin, d, color):
        """Choose the nearest same-color single in the beam's strict quadrant."""
        orr, occ = origin
        dr, dc = d
        best = None
        best_score = None
        for t in singles_by_color.get(color, []):
            tr, tc = t["cell"]
            adr = tr - orr
            adc = tc - occ
            if adr * dr <= 0 or adc * dc <= 0:
                continue
            score = (max(abs(adr), abs(adc)), abs(adr) + abs(adc), t["id"])
            if best_score is None or score < best_score:
                best_score = score
                best = t
        return best

    def simulate_beam(origin, d, color, source_id, exclude_ids=frozenset()):
        """
        Simulate one beam.

        Returns:
            path: list of cells to potentially paint
            events: list of possible generated beams/hits
        """
        orr, occ = origin
        dr, dc = d
        path = []
        events = []

        # Try to aim at a same-color single first, if one is in the quadrant.
        target = choose_single_target(origin, d, color)
        if target is not None:
            tr, tc = target["cell"]
            adr = tr - orr
            adc = tc - occ
            abs_dr = abs(adr)
            abs_dc = abs(adc)
            m = min(abs_dr, abs_dc)

            # Diagonal part.
            for step in range(1, m + 1):
                nr = orr + dr * step
                nc = occ + dc * step
                if not in_bounds(nr, nc):
                    break
                cid = int(comp_id[nr, nc])
                if cid != -1:
                    if cid == target["id"]:
                        # Target exactly diagonal: stop just before it.
                        events.append(("single_no_reflect", target["id"]))
                        return path, events
                    # Another original object blocks the approach.
                    if cid not in exclude_ids:
                        events.append(("blocked", cid))
                    return path, events
                path.append((nr, nc))
            else:
                # Reached diagonal endpoint without block.
                if abs_dr == abs_dc:
                    # Target was exactly diagonal and was skipped above; fallback.
                    events.append(("single_no_reflect", target["id"]))
                    return path, events

                if abs_dr > abs_dc:
                    straight = (dr, 0)
                    L = abs_dr - abs_dc
                    refl_d = (-dr, dc)
                    sr, sc = orr + dr * m, occ + dc * m
                else:
                    straight = (0, dc)
                    L = abs_dc - abs_dr
                    refl_d = (dr, -dc)
                    sr, sc = orr + dr * m, occ + dc * m

                last = (sr, sc)
                reached = False
                for k in range(1, L + 1):
                    nr = sr + straight[0] * k
                    nc = sc + straight[1] * k
                    if not in_bounds(nr, nc):
                        break
                    cid = int(comp_id[nr, nc])
                    if cid != -1:
                        if cid == target["id"] and k == L:
                            reached = True
                        else:
                            if cid not in exclude_ids:
                                events.append(("blocked", cid))
                        return path, events

                    if k < L:
                        path.append((nr, nc))
                        last = (nr, nc)

                if reached:
                    events.append(("single_reflect", target["id"], last, refl_d))
                return path, events

        # Ordinary pure diagonal beam.
        step = 0
        while True:
            step += 1
            nr = orr + dr * step
            nc = occ + dc * step
            if not in_bounds(nr, nc):
                break

            cid = int(comp_id[nr, nc])
            if cid != -1 and cid != source_id:
                if cid not in exclude_ids:
                    ctype = comps[cid]["type"]
                    if ctype == "Iv" or ctype == "Ih":
                        events.append(("line_hit", cid, (dr, dc)))
                    elif ctype == "S":
                        # Generic single reflection: flip the row component.
                        events.append(("single_reflect", cid, (orr + dr * (step - 1), occ + dc * (step - 1)), (-dr, dc)))
                    else:
                        events.append(("blocked", cid))
                return path, events

            path.append((nr, nc))

        return path, events

    # ------------------------------------------------------------------
    # 3. Beam generation and propagation
    # ------------------------------------------------------------------
    assigned = [False] * len(comps)
    beams = []
    event_queue = deque()

    def add_beam(origin, d, color, source_id, exclude_ids=frozenset()):
        path, events = simulate_beam(origin, d, color, source_id, exclude_ids)
        bid = len(beams)
        beams.append({
            "id": bid,
            "origin": origin,
            "dir": d,
            "color": color,
            "source_id": source_id,
            "path": path,
        })
        event_queue.append((bid, events))

    # L-shapes are initial emitters.
    for comp in comps:
        if comp["type"] == "L":
            assigned[comp["id"]] = True
            add_beam(comp["start"], comp["dir"], comp["color"], comp["id"],
                     exclude_ids=frozenset([comp["id"]]))

    # Propagate hits from all current beams.
    while event_queue:
        bid, events = event_queue.popleft()
        for ev in events:
            kind = ev[0]

            if kind == "line_hit":
                _, cid, din = ev
                comp = comps[cid]
                if not assigned[cid]:
                    assigned[cid] = True
                    start, dout = line_reflect(comp, din)
                    add_beam(start, dout, comp["color"], cid,
                             exclude_ids=frozenset([cid]))

            elif kind == "single_reflect":
                _, cid, start, dout = ev
                comp = comps[cid]
                if not assigned[cid]:
                    assigned[cid] = True
                    add_beam(start, dout, comp["color"], cid,
                             exclude_ids=frozenset([cid]))

            # "single_no_reflect" and "blocked" create no new beam.

    # Fallback for straight lines that were never hit.
    changed = True
    while changed:
        changed = False
        for comp in comps:
            if comp["type"] not in ("Iv", "Ih"):
                continue
            if assigned[comp["id"]]:
                continue

            assigned[comp["id"]] = True
            start, d = fallback_line_beam(comp)
            add_beam(start, d, comp["color"], comp["id"],
                     exclude_ids=frozenset([comp["id"]]))
            changed = True

            # Process any hits caused by this fallback beam before choosing
            # the next fallback line.
            while event_queue:
                bid, events = event_queue.popleft()
                for ev in events:
                    kind = ev[0]
                    if kind == "line_hit":
                        _, cid, din = ev
                        other = comps[cid]
                        if not assigned[cid]:
                            assigned[cid] = True
                            st, dout = line_reflect(other, din)
                            add_beam(st, dout, other["color"], cid,
                                     exclude_ids=frozenset([cid]))
                    elif kind == "single_reflect":
                        _, cid, st, dout = ev
                        other = comps[cid]
                        if not assigned[cid]:
                            assigned[cid] = True
                            add_beam(st, dout, other["color"], cid,
                                     exclude_ids=frozenset([cid]))

    # ------------------------------------------------------------------
    # 4. Resolve beam conflicts by minimum distance
    # ------------------------------------------------------------------
    INF = 10**9
    best_dist = np.full((H, W), INF, dtype=int)
    best_beam = np.full((H, W), -1, dtype=int)
    best_color = np.zeros((H, W), dtype=int)

    # Original pixels win.
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 0:
                best_dist[r, c] = 0
                best_beam[r, c] = -1
                best_color[r, c] = int(grid[r, c])

    # Consider all beam cells.
    for beam in beams:
        bid = beam["id"]
        col = int(beam["color"])
        for step, (r, c) in enumerate(beam["path"], start=1):
            if step < best_dist[r, c]:
                best_dist[r, c] = step
                best_beam[r, c] = bid
                best_color[r, c] = col
            elif step == best_dist[r, c]:
                # Deterministic tie-break: lower color, then lower beam id.
                if col < best_color[r, c] or (col == best_color[r, c] and bid < best_beam[r, c]):
                    best_dist[r, c] = step
                    best_beam[r, c] = bid
                    best_color[r, c] = col

    # Draw each beam only up to its first cell that it does not own.
    for beam in beams:
        bid = beam["id"]
        col = int(beam["color"])
        for r, c in beam["path"]:
            if best_beam[r, c] == bid:
                out[r, c] = col
            else:
                break

    return out