import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid by emitting diagonal beams from small components.

    Component types handled:
    - 2x2 L-shaped 3-cell components:
        They emit diagonally away from the missing corner of their 2x2 box.
    - Straight 3-cell components (horizontal/vertical):
        If another straight component of the same color and orientation exists,
        they emit diagonally away from their same-color partner.  The perpendicular
        component points toward the partner when the two objects are separated more
        than they are aligned, otherwise it points away.
        If no same-color straight partner exists, a fallback chain heuristic is used.
    - Single-pixel components:
        They act as targets for same-color emitters when the emitter's diagonal
        points into the target's quadrant.  The emitter beam then goes diagonally,
        turns straight to the target, and reflects diagonally from the cell before
        the target.

    All beams are resolved by minimum distance from their source; original pixels
    have priority 0.  A beam stops at the first cell claimed by another beam or
    by an original pixel.
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
                comp_id[r, c] = len(comps)

                while q:
                    x, y = q.popleft()
                    cells.append((x, y))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if comp_id[nx, ny] == -1 and grid[nx, ny] == color:
                                comp_id[nx, ny] = len(comps)
                                q.append((nx, ny))

                rs = [p[0] for p in cells]
                cs = [p[1] for p in cells]
                r0, r1 = min(rs), max(rs)
                c0, c1 = min(cs), max(cs)
                comps.append({
                    "id": len(comps),
                    "color": color,
                    "cells": cells,
                    "r0": r0, "r1": r1,
                    "c0": c0, "c1": c1,
                    "h": r1 - r0 + 1,
                    "w": c1 - c0 + 1,
                    "size": len(cells),
                    "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0)
                })

    if not comps:
        return out

    # ------------------------------------------------------------------
    # 2. Classify components and compute intrinsic directions
    # ------------------------------------------------------------------
    for comp in comps:
        comp["type"] = None
        comp["dir"] = None
        comp["start"] = None

        if comp["size"] == 1:
            comp["type"] = "single"
            comp["center_cell"] = comp["cells"][0]

        elif comp["size"] == 3 and comp["h"] == 2 and comp["w"] == 2:
            comp["type"] = "L"
            cellset = set(comp["cells"])
            # missing corner
            mr, mc = None, None
            for rr in (comp["r0"], comp["r1"]):
                for cc in (comp["c0"], comp["c1"]):
                    if (rr, cc) not in cellset:
                        mr, mc = rr, cc
            # elbow is diagonally opposite missing corner
            er = comp["r1"] if mr == comp["r0"] else comp["r0"]
            ec = comp["c1"] if mc == comp["c0"] else comp["c0"]
            comp["start"] = (er, ec)
            dr = 1 if mr < er else -1
            dc = 1 if mc < ec else -1
            comp["dir"] = (dr, dc)

        elif comp["size"] == 3 and (
            (comp["h"] == 1 and comp["w"] == 3) or
            (comp["h"] == 3 and comp["w"] == 1)
        ):
            if comp["h"] == 3:
                comp["type"] = "Iv"
            else:
                comp["type"] = "Ih"
            comp["start"] = None
            comp["dir"] = None

        else:
            # Fallback for unexpected small components: treat as single/no emit
            comp["type"] = "other"

    # ------------------------------------------------------------------
    # Helper: endpoint for an I component given a diagonal direction
    # ------------------------------------------------------------------
    def i_start_for_dir(comp, d):
        dr, dc = d
        if comp["type"] == "Iv":
            c = comp["c0"]
            r = comp["r0"] if dr > 0 else comp["r1"]
            return (r, c)
        else:
            r = comp["r0"]
            c = comp["c0"] if dc > 0 else comp["c1"]
            return (r, c)

    # ------------------------------------------------------------------
    # 3. Pair same-color, same-orientation straight components
    # ------------------------------------------------------------------
    straight_groups = {}
    for comp in comps:
        if comp["type"] in ("Iv", "Ih"):
            key = (comp["color"], comp["type"])
            straight_groups.setdefault(key, []).append(comp)

    single_i_comps = []
    for comp in comps:
        if comp["type"] in ("Iv", "Ih"):
            group = straight_groups[(comp["color"], comp["type"])]
            if len(group) >= 2:
                # nearest same-color same-orientation partner
                best = None
                best_d = None
                for other in group:
                    if other["id"] == comp["id"]:
                        continue
                    d = ((comp["center"][0] - other["center"][0]) ** 2 +
                         (comp["center"][1] - other["center"][1]) ** 2)
                    if best is None or d < best_d:
                        best = other
                        best_d = d
                comp["partner"] = best

                # Direction by pair geometry
                cr, cc = comp["center"]
                pr, pc = best["center"]
                dr = cr - pr
                dc = cc - pc

                if comp["type"] == "Iv":
                    parallel, perp = dr, dc
                    s_par = 1 if parallel > 0 else (-1 if parallel < 0 else (1 if comp["id"] < best["id"] else -1))
                    if abs(parallel) >= abs(perp):
                        s_perp = 1 if perp > 0 else (-1 if perp < 0 else (1 if comp["id"] < best["id"] else -1))
                    else:
                        s_perp = -1 if perp > 0 else (1 if perp < 0 else (-1 if comp["id"] < best["id"] else 1))
                    comp["dir"] = (s_par, s_perp)
                else:  # Ih
                    parallel, perp = dc, dr
                    s_par = 1 if parallel > 0 else (-1 if parallel < 0 else (1 if comp["id"] < best["id"] else -1))
                    if abs(parallel) >= abs(perp):
                        s_perp = 1 if perp > 0 else (-1 if perp < 0 else (1 if comp["id"] < best["id"] else -1))
                    else:
                        s_perp = -1 if perp > 0 else (1 if perp < 0 else (-1 if comp["id"] < best["id"] else 1))
                    comp["dir"] = (s_perp, s_par)
            else:
                comp["partner"] = None
                single_i_comps.append(comp)

    # ------------------------------------------------------------------
    # 4. Fallback direction assignment for unpaired straight components
    #    (chain heuristic, used to match Example #2 style cases)
    # ------------------------------------------------------------------
    def first_component_on_ray(start, d, exclude_id=None):
        r, c = start
        dr, dc = d
        step = 0
        while True:
            r += dr
            c += dc
            step += 1
            if r < 0 or r >= H or c < 0 or c >= W:
                return None, step - 1
            cid = int(comp_id[r, c])
            if cid != -1 and cid != exclude_id:
                return cid, step
            # allow background; stop only at border or component

    def direction_priority(d):
        dr, dc = d
        return (1 if dr > 0 else 0, 1 if dc > 0 else 0)

    # Use L components as chain seeds.
    visited = set()
    queue = deque()

    # Process L components first to mark exact targets for fallback chain
    for comp in sorted([x for x in comps if x["type"] == "L"], key=lambda x: (x["r0"], x["c0"], x["id"])):
        if comp["dir"] is None:
            continue
        start = comp["start"]
        target_id, _ = first_component_on_ray(start, comp["dir"], exclude_id=comp["id"])
        if target_id is not None and target_id not in visited:
            visited.add(target_id)
            tcomp = comps[target_id]
            if tcomp["type"] in ("Iv", "Ih") and tcomp.get("partner") is None:
                queue.append(target_id)

    # BFS through targeted unpaired straight components
    processed = set()
    while queue:
        cid = queue.popleft()
        if cid in processed:
            continue
        comp = comps[cid]
        if comp["type"] not in ("Iv", "Ih") or comp.get("partner") is not None:
            continue
        processed.add(cid)
        visited.add(cid)

        best_dir = None
        best_score = None
        best_target = None

        for d in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            start = i_start_for_dir(comp, d)
            target_id, dist = first_component_on_ray(start, d, exclude_id=comp["id"])

            # If target is unvisited, this is a candidate target direction
            if target_id is not None and target_id not in visited:
                score = (dist, direction_priority(d))
                if best_score is None or score > best_score:
                    best_score = score
                    best_dir = d
                    best_target = target_id
            else:
                # No usable target: remember length to first obstacle/border
                score_edge = (0, dist, direction_priority(d))
                if best_dir is None:
                    # we will store separately below
                    pass

        # If no unvisited target, choose longest effective ray
        if best_dir is None:
            best_len = -1
            for d in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                start = i_start_for_dir(comp, d)
                _, dist = first_component_on_ray(start, d, exclude_id=comp["id"])
                score = (dist, direction_priority(d))
                if best_len < 0 or dist > best_len or (dist == best_len and score > best_score if best_score is not None else dist > best_len):
                    best_dir = d
                    best_len = dist
                    best_score = score

        if best_dir is not None:
            comp["dir"] = best_dir
            if best_target is not None:
                tcomp = comps[best_target]
                if tcomp["type"] in ("Iv", "Ih") and tcomp.get("partner") is None:
                    queue.append(best_target)

    # Any remaining unpaired straight components: assign in row-major order
    for comp in sorted([x for x in single_i_comps], key=lambda x: (x["r0"], x["c0"], x["id"])):
        if comp["id"] in processed or comp["dir"] is not None:
            continue
        processed.add(comp["id"])
        visited.add(comp["id"])

        best_dir = None
        best_score = None
        best_target = None

        for d in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            start = i_start_for_dir(comp, d)
            target_id, dist = first_component_on_ray(start, d, exclude_id=comp["id"])
            if target_id is not None and target_id not in visited:
                score = (dist, direction_priority(d))
                if best_score is None or score > best_score:
                    best_score = score
                    best_dir = d
                    best_target = target_id

        if best_dir is None:
            best_len = -1
            for d in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                start = i_start_for_dir(comp, d)
                _, dist = first_component_on_ray(start, d, exclude_id=comp["id"])
                if best_len < 0 or dist > best_len:
                    best_dir = d
                    best_len = dist
                    best_score = (dist, direction_priority(d))

        if best_dir is not None:
            comp["dir"] = best_dir
            if best_target is not None:
                tcomp = comps[best_target]
                if tcomp["type"] in ("Iv", "Ih") and tcomp.get("partner") is None:
                    queue.append(best_target)
                    # process immediately if possible
                    while queue:
                        cid = queue.popleft()
                        if cid in processed:
                            continue
                        c2 = comps[cid]
                        if c2["type"] not in ("Iv", "Ih") or c2.get("partner") is not None:
                            continue
                        processed.add(cid)
                        visited.add(cid)
                        # simple assignment for chained remaining singles
                        bd = None
                        bs = None
                        bt = None
                        for d2 in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                            st2 = i_start_for_dir(c2, d2)
                            tid2, dist2 = first_component_on_ray(st2, d2, exclude_id=c2["id"])
                            if tid2 is not None and tid2 not in visited:
                                sc2 = (dist2, direction_priority(d2))
                                if bs is None or sc2 > bs:
                                    bs = sc2
                                    bd = d2
                                    bt = tid2
                        if bd is None:
                            bl = -1
                            for d2 in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                                st2 = i_start_for_dir(c2, d2)
                                _, dist2 = first_component_on_ray(st2, d2, exclude_id=c2["id"])
                                if dist2 > bl:
                                    bd = d2
                                    bl = dist2
                                    bs = (dist2, direction_priority(d2))
                        if bd is not None:
                            c2["dir"] = bd
                            if bt is not None:
                                t2 = comps[bt]
                                if t2["type"] in ("Iv", "Ih") and t2.get("partner") is None:
                                    queue.append(bt)

    # ------------------------------------------------------------------
    # 5. Build beam paths
    # ------------------------------------------------------------------
    beams = []

    def in_bounds(r, c):
        return 0 <= r < H and 0 <= c < W

    def is_input(r, c):
        return grid[r, c] != 0

    def add_simple_beam(comp, path):
        r, c = comp["start"]
        dr, dc = comp["dir"]
        dist = 0
        while True:
            r += dr
            c += dc
            dist += 1
            if not in_bounds(r, c) or is_input(r, c):
                break
            path.append(((r, c), dist))

    def add_aimed_beam(comp, target, path):
        sr, sc = comp["start"]
        dr, dc = comp["dir"]
        tr, tc = target

        adr = tr - sr
        adc = tc - sc

        # Only aim if target is strictly in the diagonal quadrant
        if (adr * dr <= 0) or (adc * dc <= 0):
            add_simple_beam(comp, path)
            return

        abs_dr = abs(adr)
        abs_dc = abs(adc)
        m = min(abs_dr, abs_dc)

        dist = 0

        # Diagonal segment, stopping before the target if it is exactly diagonal
        for step in range(1, m + 1):
            nr = sr + dr * step
            nc = sc + dc * step
            if not in_bounds(nr, nc) or is_input(nr, nc):
                break
            path.append(((nr, nc), step))
            dist = step
        else:
            # reached the intended diagonal length without hitting input
            dist = m

        if abs_dr == abs_dc:
            # target is directly diagonal; no reflection in observed cases
            return

        # Which axis was reached first?
        if abs_dr > abs_dc:
            straight = (dr, 0)
            L = abs_dr - abs_dc
            jr, jc = sr + dr * m, sc + dc * m
            jdist = m
        else:
            straight = (0, dc)
            L = abs_dc - abs_dr
            jr, jc = sr + dr * m, sc + dc * m
            jdist = m

        # Walk straight until the cell just before the target
        for k in range(1, L):
            jr += straight[0]
            jc += straight[1]
            jdist += 1
            if not in_bounds(jr, jc) or is_input(jr, jc):
                return
            path.append(((jr, jc), jdist))

        # Reflection direction
        if straight[0] != 0:
            rdir, cdir = -dr, dc
        else:
            rdir, cdir = dr, -dc

        # Continue from the cell before target in reflected diagonal direction
        rr, cc = jr, jc
        d = jdist
        while True:
            rr += rdir
            cc += cdir
            d += 1
            if not in_bounds(rr, cc) or is_input(rr, cc):
                break
            path.append(((rr, cc), d))

    # Choose same-color single target if present and in the beam's quadrant
    singles_by_color = {}
    for comp in comps:
        if comp["type"] == "single":
            singles_by_color.setdefault(comp["color"], []).append(comp["center_cell"])

    for comp in comps:
        if comp["type"] not in ("L", "Iv", "Ih"):
            continue
        if comp["dir"] is None:
            continue
        if comp["type"] in ("Iv", "Ih"):
            comp["start"] = i_start_for_dir(comp, comp["dir"])

        path = []
        target = None
        for t in singles_by_color.get(comp["color"], []):
            # Avoid using the component's own cell (not relevant for non-singles)
            if (t[0], t[1]) == comp["start"]:
                continue
            target = t
            break

        if target is not None:
            add_aimed_beam(comp, target, path)
        else:
            add_simple_beam(comp, path)

        if path:
            beams.append({
                "src": comp["id"],
                "color": comp["color"],
                "path": path
            })

    # ------------------------------------------------------------------
    # 6. Resolve conflicts by minimum distance
    # ------------------------------------------------------------------
    best_dist = np.full((H, W), np.inf, dtype=int)
    best_src = -np.ones((H, W), dtype=int)
    best_color = np.zeros((H, W), dtype=int)

    for r in range(H):
        for c in range(W):
            if grid[r, c] != 0:
                best_dist[r, c] = 0
                best_src[r, c] = -1
                best_color[r, c] = grid[r, c]

    for beam in beams:
        src = beam["src"]
        col = beam["color"]
        for (r, c), d in beam["path"]:
            if d < best_dist[r, c]:
                best_dist[r, c] = d
                best_src[r, c] = src
                best_color[r, c] = col
            elif d == best_dist[r, c] and best_src[r, c] != -1:
                # Tie-break: lower color, then lower source id
                if (col, src) < (best_color[r, c], best_src[r, c]):
                    best_dist[r, c] = d
                    best_src[r, c] = src
                    best_color[r, c] = col

    # ------------------------------------------------------------------
    # 7. Materialize each beam only until first conflicting cell
    # ------------------------------------------------------------------
    for beam in beams:
        src = beam["src"]
        col = beam["color"]
        for (r, c), d in beam["path"]:
            if best_src[r, c] == src:
                out[r, c] = col
            else:
                break

    return out