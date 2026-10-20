import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform ARC-style grids containing 3x3 box objects and a path of 5s.

    Rule implemented:
    - Detect 3x3 box objects: center + border color.
    - Trace the path of 5s (allowing straight/diagonal one-cell gaps).
    - Keep the boxes at the two ends of the 5-path.
    - For each kept box, set its center to the center color of the box
      whose border color equals this box's original center color.
    - Erase all other boxes, preserving 5s.
    """
    if grid is None:
        return np.zeros((0, 0), dtype=int)

    g = np.array(grid, dtype=int)
    if g.ndim != 2:
        return g
    h, w = g.shape
    if h == 0 or w == 0:
        return g

    # ---------- small helpers ----------
    def cheb(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def norm_vec(a, b):
        dr = a[0] - b[0]
        dc = a[1] - b[1]
        dr = 0 if dr == 0 else (1 if dr > 0 else -1)
        dc = 0 if dc == 0 else (1 if dc > 0 else -1)
        return dr, dc

    DIRS = {
        (0, 1): 0,
        (1, 1): 1,
        (1, 0): 2,
        (1, -1): 3,
        (0, -1): 4,
        (-1, -1): 5,
        (-1, 0): 6,
        (-1, 1): 7,
    }

    def angle_penalty(d1, d2):
        if d1 == (0, 0) or d2 == (0, 0):
            return 0
        i1 = DIRS[d1]
        i2 = DIRS[d2]
        d = abs(i1 - i2)
        return min(d, 8 - d)

    def region_distance(point, region):
        r, c = point
        r1, r2, c1, c2 = region
        dr = max(r1 - r, 0, r - r2)
        dc = max(c1 - c, 0, c - c2)
        return max(dr, dc)

    # ---------- detect box objects ----------
    objects = []
    seen_centers = set()

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center_color = g[r, c]
            if center_color == 0 or center_color == 5:
                continue

            border = g[r - 1, c]
            if border == 0 or border == 5 or border == center_color:
                continue
            if g[r + 1, c] != border:
                continue
            if g[r, c - 1] != border:
                continue
            if g[r, c + 1] != border:
                continue

            ok = True
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if rr == r and cc == c:
                        continue
                    val = g[rr, cc]
                    # corners may be empty, border color, or 5 (path overlap)
                    if val != border and val != 0 and val != 5:
                        ok = False
                        break
                if not ok:
                    break

            if not ok or (r, c) in seen_centers:
                continue

            seen_centers.add((r, c))
            objects.append({
                "center": (r, c),
                "frame": border,
                "center_color": center_color,
                "region": (r - 1, r + 1, c - 1, c + 1),
            })

    for i, o in enumerate(objects):
        o["id"] = i

    if not objects:
        return g.copy()

    frame_to_objs = {}
    frame_as_center = {}
    for o in objects:
        frame_to_objs.setdefault(o["frame"], []).append(o)
        frame_as_center.setdefault(o["center_color"], []).append(o["id"])

    def indegree(o):
        return len([x for x in frame_as_center.get(o["frame"], []) if x != o["id"]])

    # ---------- endpoint / path helpers ----------
    def endpoint_candidates(ep, nb, objs, frame_as_center):
        """
        Score boxes that can be considered attached to an endpoint 5-cell.
        """
        dirv = norm_vec(ep, nb) if nb is not None else (0, 0)

        candidates = []
        seen = set()

        # First, boxes near the endpoint cell itself.
        for dist in (1, 2, 3):
            for o in objs:
                if o["id"] in seen:
                    continue
                if region_distance(ep, o["region"]) <= dist:
                    candidates.append(o)
                    seen.add(o["id"])
            if candidates:
                break

        # If none, try the one-cell forward direction.
        if not candidates:
            forward = (ep[0] + dirv[0], ep[1] + dirv[1]) if dirv != (0, 0) else ep
            for dist in (1, 2, 3):
                for o in objs:
                    if o["id"] in seen:
                        continue
                    if region_distance(forward, o["region"]) <= dist:
                        candidates.append(o)
                        seen.add(o["id"])
                if candidates:
                    break

        if not candidates:
            return []

        scored = []
        for o in candidates:
            cr, cc = o["center"]
            dot = 0
            if dirv != (0, 0):
                dot = (cr - ep[0]) * dirv[0] + (cc - ep[1]) * dirv[1]

            forward = (ep[0] + dirv[0], ep[1] + dirv[1]) if dirv != (0, 0) else ep
            dist_f = region_distance(forward, o["region"])
            dist_e = region_distance(ep, o["region"])
            source = 1 if indegree(o) == 0 else 0

            score = dot * 100 - dist_f * 10 - dist_e * 5 + source * 2
            scored.append((score, o))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:4]

    def best_endpoint_pair(path):
        if not path:
            return (None, None), 0

        if len(path) == 1:
            c0 = endpoint_candidates(path[0], None, objects, frame_as_center)
            c1 = []
        else:
            c0 = endpoint_candidates(path[0], path[1], objects, frame_as_center)
            c1 = endpoint_candidates(path[-1], path[-2], objects, frame_as_center)

        best_pair = (None, None)
        best_pair_score = 0

        if c0 and c1:
            for s0, b0 in c0:
                for s1, b1 in c1:
                    sc = s0 + s1
                    if b0["id"] != b1["id"]:
                        sc += 50
                    if sc > best_pair_score:
                        best_pair_score = sc
                        best_pair = (b0, b1)

        return best_pair, best_pair_score

    def score_path(path, N):
        if not path:
            return -10**18, (None, None)

        L = len(path)
        pair, pair_score = best_endpoint_pair(path)

        primary = 0
        if L == N:
            primary += 10000
        primary += L * 10

        b0, b1 = pair
        if b0 is not None:
            primary += 100
        if b1 is not None:
            primary += 100
        if b0 is not None and b1 is not None and b0["id"] != b1["id"]:
            primary += 50

        return primary * 10000 + pair_score, pair

    def trace_greedy(start, first, adj, N):
        if first is None:
            return [start]

        path = [start, first]
        visited = {start, first}
        prev = start
        curr = first

        while len(path) < N:
            cands = [n for n in adj[curr] if n not in visited]
            if not cands:
                break

            if len(cands) == 1:
                nxt = cands[0]
            else:
                dirv = norm_vec(curr, prev)

                def cand_score(n):
                    nd = norm_vec(n, curr)
                    turn = angle_penalty(dirv, nd)
                    onward = sum(1 for m in adj[n] if m not in visited)
                    dead = 1 if (onward == 0 and len(path) < N - 1) else 0
                    return (dead, turn, onward)

                nxt = min(cands, key=cand_score)

            path.append(nxt)
            visited.add(nxt)
            prev, curr = curr, nxt

        return path

    def find_best_path(comp, adj):
        cells = list(comp)
        N = len(cells)
        if N == 1:
            path = cells[:]
            _, pair = score_path(path, N)
            return path, pair

        degree = {v: len(adj[v]) for v in cells}

        box_near = [
            v for v in cells
            if any(region_distance(v, o["region"]) <= 1 for o in objects)
        ]

        starts = []
        seen = set()

        for v in sorted(cells, key=lambda x: (degree[x], x[0], x[1])):
            if degree[v] <= 1 and v not in seen:
                starts.append(v)
                seen.add(v)

        for v in sorted(box_near, key=lambda x: (degree[x], x[0], x[1])):
            if v not in seen:
                starts.append(v)
                seen.add(v)

        for v in sorted(cells, key=lambda x: (degree[x], x[0], x[1])):
            if v not in seen:
                starts.append(v)
                seen.add(v)

        best_path = None
        best_pair = (None, None)
        best_score = -10**18

        max_attempts = max(200, N * 15)
        attempts = 0

        for start in starts:
            if attempts >= max_attempts:
                break

            neigh = adj[start]
            if not neigh:
                path = [start]
                sc, pair = score_path(path, N)
                if sc > best_score:
                    best_score = sc
                    best_path = path
                    best_pair = pair
                continue

            firsts = sorted(neigh, key=lambda n: (degree[n], n[0], n[1]))
            for first in firsts:
                attempts += 1
                if attempts >= max_attempts:
                    break

                path = trace_greedy(start, first, adj, N)
                if path is None:
                    continue

                sc, pair = score_path(path, N)
                if sc > best_score:
                    best_score = sc
                    best_path = path
                    best_pair = pair

        return best_path, best_pair

    # ---------- build 5-path graph ----------
    fives = [(r, c) for r in range(h) for c in range(w) if g[r, c] == 5]
    kept = set()
    best_comp = None

    if fives:
        cell_set = set(fives)

        # Normal 8-connected components.
        normal_adj = {v: [] for v in fives}
        for r, c in fives:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    n = (r + dr, c + dc)
                    if n in cell_set:
                        normal_adj[(r, c)].append(n)

        normal_comp_id = {}
        normal_comps = []
        for v in fives:
            if v in normal_comp_id:
                continue
            cid = len(normal_comps)
            stack = [v]
            normal_comp_id[v] = cid
            comp = []
            while stack:
                x = stack.pop()
                comp.append(x)
                for y in normal_adj[x]:
                    if y not in normal_comp_id:
                        normal_comp_id[y] = cid
                        stack.append(y)
            normal_comps.append(comp)

        # Add dashed gaps: straight/diagonal distance-2 gaps with an empty middle.
        gap_adj = {v: [] for v in fives}
        gap_offsets = [
            (0, 2), (0, -2), (2, 0), (-2, 0),
            (2, 2), (2, -2), (-2, 2), (-2, -2)
        ]

        for r, c in fives:
            for dr, dc in gap_offsets:
                n = (r + dr, c + dc)
                if n not in cell_set:
                    continue
                if normal_comp_id[n] == normal_comp_id[(r, c)]:
                    continue
                mr, mc = r + dr // 2, c + dc // 2
                if 0 <= mr < h and 0 <= mc < w and g[mr, mc] == 0:
                    gap_adj[(r, c)].append(n)
                    gap_adj[n].append((r, c))

        aug_adj = {
            v: sorted(set(normal_adj[v] + gap_adj[v]))
            for v in fives
        }

        # Augmented components (dashed gaps can merge components).
        aug_comp_id = {}
        aug_comps = []
        for v in fives:
            if v in aug_comp_id:
                continue
            cid = len(aug_comps)
            stack = [v]
            aug_comp_id[v] = cid
            comp = []
            while stack:
                x = stack.pop()
                comp.append(x)
                for y in aug_adj[x]:
                    if y not in aug_comp_id:
                        aug_comp_id[y] = cid
                        stack.append(y)
            aug_comps.append(comp)

        best_score = -10**18
        best_pair = (None, None)

        for comp in aug_comps:
            path, pair = find_best_path(comp, aug_adj)
            if path is None:
                continue
            sc, _ = score_path(path, len(comp))
            key = (sc, len(comp))
            if key > (best_score, 0):
                best_score = sc
                best_pair = pair
                best_comp = comp

        if best_pair:
            for b in best_pair:
                if b is not None:
                    kept.add(b["id"])

    # ---------- fallback kept selection ----------
    if len(kept) < 2:
        pool = [o for o in objects if o["id"] not in kept]
        if pool:
            kept_centers = [o["center"] for o in objects if o["id"] in kept]

            def fallback_key(o):
                if best_comp:
                    dcomp = min(region_distance(v, o["region"]) for v in best_comp)
                else:
                    dcomp = 9
                if kept_centers:
                    dkept = min(cheb(o["center"], kc) for kc in kept_centers)
                else:
                    dkept = 0
                source = 1 if indegree(o) == 0 else 0
                return (dcomp, -source, dkept, o["center"][0], o["center"][1])

            for o in sorted(pool, key=fallback_key):
                kept.add(o["id"])
                if len(kept) >= 2:
                    break

    # ---------- choose new centers for kept boxes ----------
    new_centers = {}
    for o in objects:
        if o["id"] not in kept:
            continue

        cands = [
            c for c in frame_to_objs.get(o["center_color"], [])
            if c["id"] != o["id"]
        ]
        if cands:
            cands.sort(key=lambda x: (cheb(o["center"], x["center"]), x["id"]))
            new_centers[o["id"]] = cands[0]["center_color"]
        else:
            new_centers[o["id"]] = o["center_color"]

    # ---------- build output ----------
    out = g.copy()

    # Erase non-kept boxes, but keep 5s.
    for o in objects:
        if o["id"] in kept:
            continue
        r1, r2, c1, c2 = o["region"]
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if out[r, c] != 5:
                    out[r, c] = 0

    # Restore kept boxes from original grid, then set new centers.
    for o in objects:
        if o["id"] not in kept:
            continue
        r1, r2, c1, c2 = o["region"]
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                out[r, c] = g[r, c]
        cr, cc = o["center"]
        out[cr, cc] = new_centers[o["id"]]

    return out