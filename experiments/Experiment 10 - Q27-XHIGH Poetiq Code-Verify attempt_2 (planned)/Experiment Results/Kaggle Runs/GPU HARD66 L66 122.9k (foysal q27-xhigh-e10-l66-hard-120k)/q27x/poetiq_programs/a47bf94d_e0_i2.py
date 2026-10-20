def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform colored 3x3 ports connected by 8/5/9 wire networks.

    Rules:
    - Ports are 3x3 solid blocks, edge-crosses (E), corner-crosses (X), or empty.
    - Wire cells are 8, 5, and 9. 9 blocks are used for routing only and are
      left unchanged in the output.
    - A port has a wire endpoint at center +/- 2 in one cardinal direction.
    - Connected ports are paired by non-conflicting wire paths.
    - The color comes from the non-empty port in a pair.
    - Pattern mapping:
        solid/E -> E
        X       -> X
        empty paired with E-source -> X
        empty paired with X-source -> E
    """
    import numpy as np

    g = np.array(grid, dtype=int, copy=True)
    if g.ndim != 2:
        return g

    H, W = g.shape
    if H < 3 or W < 3:
        return g

    WIRE = (8, 5, 9)
    wire = np.isin(g, WIRE)

    SIDES = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
    OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
    BIT = {"N": 1, "S": 2, "E": 4, "W": 8}

    def inb(r, c):
        return 0 <= r < H and 0 <= c < W

    def is_isolated(r, c, val):
        """Reject patterns that are part of a larger same-color blob."""
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr, cc = r + dr, c + dc
                if g[rr, cc] != val:
                    continue
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if r - 1 <= nr <= r + 1 and c - 1 <= nc <= c + 1:
                        continue
                    if inb(nr, nc) and g[nr, nc] == val:
                        return False
        return True

    def analyze_3x3(r, c):
        """
        Return ('empty', None), ('solid', val), ('e', val), or ('x', val),
        or None if this 3x3 is not a valid isolated port.
        """
        sub = g[r - 1:r + 2, c - 1:c + 2]

        # Ports do not contain wire cells.
        if np.any(np.isin(sub, WIRE)):
            return None

        uniq = np.unique(sub)

        if len(uniq) == 1:
            val = int(uniq[0])
            if val == 0:
                return ("empty", None)
            if val in WIRE:
                return None
            if not is_isolated(r, c, val):
                return None
            return ("solid", val)

        nonzero = uniq[uniq != 0]
        if len(nonzero) == 1:
            val = int(nonzero[0])
            if val in WIRE:
                return None

            # E pattern:
            # 0 c 0
            # c 0 c
            # 0 c 0
            if (
                sub[0, 1] == val and
                sub[1, 0] == val and
                sub[1, 2] == val and
                sub[2, 1] == val and
                sub[0, 0] == 0 and sub[0, 2] == 0 and
                sub[1, 1] == 0 and
                sub[2, 0] == 0 and sub[2, 2] == 0
            ):
                if is_isolated(r, c, val):
                    return ("e", val)

            # X pattern:
            # c 0 c
            # 0 c 0
            # c 0 c
            if (
                sub[0, 0] == val and sub[0, 2] == val and
                sub[1, 1] == val and
                sub[2, 0] == val and sub[2, 2] == val and
                sub[0, 1] == 0 and sub[1, 0] == 0 and
                sub[1, 2] == 0 and sub[2, 1] == 0
            ):
                if is_isolated(r, c, val):
                    return ("x", val)

        return None

    ports = []
    endpoints = []
    endpoint_cells = {}
    port_centers = []

    def add_endpoint(cell, port_id, side):
        ep_id = len(endpoints)
        endpoints.append({"cell": cell, "port": port_id, "side": side})
        endpoint_cells.setdefault(cell, []).append(ep_id)
        ports[port_id]["endpoints"].append(ep_id)

    # Detect ports.
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            # Avoid overlapping port centers, especially for empty 3x3 zones.
            if any(abs(r - pr) <= 1 and abs(c - pc) <= 1 for pr, pc in port_centers):
                continue

            analysis = analyze_3x3(r, c)
            if analysis is None:
                continue

            state, color = analysis

            # Empty ports are only relevant if they actually have a wire endpoint.
            if state == "empty":
                has_ep = False
                for side, (dr, dc) in SIDES.items():
                    er, ec = r + 2 * dr, c + 2 * dc
                    if inb(er, ec) and wire[er, ec]:
                        has_ep = True
                        break
                if not has_ep:
                    continue

            port_id = len(ports)
            ports.append({
                "id": port_id,
                "center": (r, c),
                "state": state,
                "color": color,
                "endpoints": [],
            })
            port_centers.append((r, c))

            for side, (dr, dc) in SIDES.items():
                er, ec = r + 2 * dr, c + 2 * dc
                if inb(er, ec) and wire[er, ec]:
                    add_endpoint((er, ec), port_id, side)

    if not ports:
        return g.copy()

    # Build wire adjacency.
    adj = {}
    for r in range(H):
        for c in range(W):
            if not wire[r, c]:
                continue
            neigh = []
            for side, (dr, dc) in SIDES.items():
                nr, nc = r + dr, c + dc
                if inb(nr, nc) and wire[nr, nc]:
                    neigh.append((side, (nr, nc)))
            adj[(r, c)] = neigh

    used = [[0] * W for _ in range(H)]

    def dir_from_delta(dr, dc):
        if dr == -1:
            return "N"
        if dr == 1:
            return "S"
        if dc == 1:
            return "E"
        if dc == -1:
            return "W"
        return None

    def find_paths(start_ep_id, target_port_ids, used_masks, max_paths=6, max_steps=200):
        """
        Find several feasible wire paths from one endpoint to any endpoint
        of a target colored port, respecting already-used wire directions.
        """
        target_port_ids = set(target_port_ids)
        if not target_port_ids:
            return []

        start_ep = endpoints[start_ep_id]
        start_cell = start_ep["cell"]
        if start_cell not in adj:
            return []

        target_ep_ids = set()
        target_cells = []
        for ep_id, ep in enumerate(endpoints):
            if ep_id == start_ep_id:
                continue
            if ep["port"] in target_port_ids:
                target_ep_ids.add(ep_id)
                target_cells.append(ep["cell"])

        if not target_ep_ids:
            return []

        max_steps = min(max_steps, H * W)
        results = []
        states = 0
        STATE_LIMIT = 8000

        path = [start_cell]
        visited = {start_cell}
        start_side = start_ep["side"]

        def min_target_dist(cell):
            r, c = cell
            best = 10**9
            for tr, tc in target_cells:
                d = abs(r - tr) + abs(c - tc)
                if d < best:
                    best = d
            return best

        def candidates(cell, back_dir):
            neigh = adj.get(cell, [])
            r, c = cell
            deg = len(neigh)
            cand = []

            for d, nxt in neigh:
                if back_dir is not None and d == back_dir:
                    continue

                # Do not use a direction already occupied by another path.
                if used_masks[r][c] & BIT[d]:
                    continue
                nr, nc = nxt
                if used_masks[nr][nc] & BIT[OPP[d]]:
                    continue

                # A single path should not revisit its own cells.
                if nxt in visited:
                    continue

                dist = min_target_dist(nxt)

                if back_dir is None:
                    # Starting move: prefer the natural outward direction.
                    pref = 0 if d == start_side else 1
                    cand.append((pref, dist, d, nxt, 0))
                else:
                    straight = OPP[back_dir]

                    # At a true 4-way crossing, go straight.
                    if deg == 4 and d != straight:
                        continue

                    tc = 0 if d == straight else 1
                    cand.append((tc, dist, d, nxt, tc))

            cand.sort(key=lambda x: (x[0], x[1], x[2]))
            return [(x[2], x[3], x[4]) for x in cand]

        def dfs(cell, back_dir, turns, steps):
            nonlocal states

            if len(results) >= max_paths:
                return
            if steps > max_steps:
                return
            if turns > 25:
                return

            dist = min_target_dist(cell)
            if steps + dist > max_steps:
                return

            # Ports are terminals: stop when entering any port endpoint.
            if steps > 0 and cell in endpoint_cells:
                for ep_id in endpoint_cells[cell]:
                    if ep_id in target_ep_ids:
                        results.append((turns, steps, tuple(path), ep_id, start_ep_id))
                        return
                return

            states += 1
            if states > STATE_LIMIT:
                return

            for d, nxt, tc in candidates(cell, back_dir):
                path.append(nxt)
                visited.add(nxt)

                dfs(nxt, OPP[d], turns + tc, steps + 1)

                path.pop()
                visited.remove(nxt)

                if len(results) >= max_paths:
                    return

        dfs(start_cell, None, 0, 0)
        results.sort(key=lambda x: (x[0], x[1]))
        return results[:max_paths]

    colored_ports = [p["id"] for p in ports if p["color"] is not None]
    empty_ports = [p["id"] for p in ports if p["state"] == "empty"]

    # Order empty ports by how many immediate colored paths they have.
    empty_order = []
    for p in empty_ports:
        cnt = 0
        for ep in ports[p]["endpoints"]:
            res = find_paths(ep, set(colored_ports), used, max_paths=3)
            cnt += len(res)
            if cnt >= 3:
                break
        empty_order.append((cnt, p))
    empty_order.sort(key=lambda x: (x[0], x[1]))
    empty_order = [p for _, p in empty_order]

    best_score = -1
    best_cost = 10**9
    best_pairs = []
    search_states = 0
    SEARCH_LIMIT = 20000

    def update_best(score, cost, pairs):
        nonlocal best_score, best_cost, best_pairs
        if score > best_score or (score == best_score and cost < best_cost):
            best_score = score
            best_cost = cost
            best_pairs = pairs[:]

    def apply_path(path, sign):
        """Mark or unmark wire directions used by a chosen path."""
        for i in range(len(path) - 1):
            r, c = path[i]
            nr, nc = path[i + 1]
            d = dir_from_delta(nr - r, nc - c)
            if d is None:
                continue
            b = BIT[d]
            ob = BIT[OPP[d]]
            if sign == 1:
                used[r][c] |= b
                used[nr][nc] |= ob
            else:
                used[r][c] &= ~b
                used[nr][nc] &= ~ob

    def search(remaining_empty, assigned, pairs, score, cost):
        nonlocal search_states

        if not remaining_empty:
            update_best(score, cost, pairs)
            return

        # Branch and bound.
        if score + len(remaining_empty) < best_score:
            return
        if score + len(remaining_empty) == best_score and cost >= best_cost:
            return

        p = None
        for ep_port in empty_order:
            if ep_port in remaining_empty and ep_port not in assigned:
                p = ep_port
                break

        if p is None:
            update_best(score, cost, pairs)
            return

        search_states += 1
        if search_states > SEARCH_LIMIT:
            return

        available_colored = {q for q in colored_ports if q not in assigned}
        all_paths = []
        for ep in ports[p]["endpoints"]:
            all_paths.extend(find_paths(ep, available_colored, used, max_paths=6))

        # If no path exists for this empty port, skip it.
        if not all_paths:
            search(
                remaining_empty - {p},
                assigned | {p},
                pairs,
                score,
                cost,
            )
            return

        all_paths.sort(key=lambda x: (x[0], x[1]))

        # Try pairing this empty port with each feasible colored port.
        for turns, length, path, target_ep, start_ep in all_paths:
            target_port = endpoints[target_ep]["port"]
            if target_port in assigned:
                continue

            apply_path(path, 1)
            search(
                remaining_empty - {p},
                assigned | {p, target_port},
                pairs + [(start_ep, target_ep)],
                score + 1,
                cost + turns + length,
            )
            apply_path(path, -1)

        # Or leave this empty port empty.
        search(
            remaining_empty - {p},
            assigned | {p},
            pairs,
            score,
            cost,
        )

    search(set(empty_ports), set(), [], 0, 0)

    # Determine final port colors and patterns.
    out_color = {}
    out_pattern = {}
    opposite = {"e": "x", "x": "e"}

    for p in ports:
        if p["color"] is not None:
            out_color[p["id"]] = p["color"]
            out_pattern[p["id"]] = "x" if p["state"] == "x" else "e"

    for start_ep, target_ep in best_pairs:
        p1_id = endpoints[start_ep]["port"]
        p2_id = endpoints[target_ep]["port"]
        p1 = ports[p1_id]
        p2 = ports[p2_id]

        if p1["color"] is not None and p2["color"] is not None:
            # Both already colored: keep them unchanged.
            continue
        if p1["color"] is not None:
            src = p1
        elif p2["color"] is not None:
            src = p2
        else:
            continue

        src_pat = "x" if src["state"] == "x" else "e"

        for p in (p1, p2):
            if p["state"] == "empty":
                pat = opposite[src_pat]
                col = src["color"]
            else:
                pat = "x" if p["state"] == "x" else "e"
                col = p["color"] if p["color"] is not None else src["color"]

            out_color[p["id"]] = col
            out_pattern[p["id"]] = pat

    out = g.copy()

    for p in ports:
        col = out_color.get(p["id"])
        pat = out_pattern.get(p["id"])
        if col is None or pat is None:
            continue

        r, c = p["center"]

        # Clear the old 3x3 port region.
        out[r - 1:r + 2, c - 1:c + 2] = 0

        if pat == "e":
            out[r - 1, c] = col
            out[r + 1, c] = col
            out[r, c - 1] = col
            out[r, c + 1] = col
        elif pat == "x":
            out[r - 1, c - 1] = col
            out[r - 1, c + 1] = col
            out[r, c] = col
            out[r + 1, c - 1] = col
            out[r + 1, c + 1] = col

    return out