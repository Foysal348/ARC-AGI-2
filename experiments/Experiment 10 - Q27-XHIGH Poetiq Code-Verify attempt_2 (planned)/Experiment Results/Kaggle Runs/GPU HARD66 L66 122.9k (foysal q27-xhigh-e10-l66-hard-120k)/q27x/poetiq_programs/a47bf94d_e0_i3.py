import numpy as np
import heapq
from collections import defaultdict, deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform 3x3 port symbols connected by wire networks.

    The input contains:
      - small 3x3 symbols in colors other than 0, 5, 8, 9
      - wire cells 8 and 5
      - solid 9 blocks that act as straight-through connector boxes

    For each wire endpoint, a 3x3 port is inferred.  The port color is taken
    from a non-empty connected port.  The output symbol shape is determined by
    the side on which the wire endpoint lies:
        - wire endpoint on the south or east side of the port center: E-symbol
        - wire endpoint on the north or west side of the port center: X-symbol

    X-symbols are normalized to even column centers to match the examples.
    """

    g = np.array(grid, dtype=int, copy=True)
    if g.ndim != 2:
        return g

    H, W = g.shape
    if H < 3 or W < 3:
        return g

    N, E, S, Wd = 0, 1, 2, 3
    DIRS = {
        N: (-1, 0),
        E: (0, 1),
        S: (1, 0),
        Wd: (0, -1),
    }
    OPP = {N: S, S: N, E: Wd, Wd: E}
    RIGHT = {N: E, E: S, S: Wd, Wd: N}
    LEFT = {N: Wd, Wd: N, N: E, E: S}

    wire85 = (g == 8) | (g == 5)
    wire9 = (g == 9)

    # ------------------------------------------------------------------
    # 9-block handling
    # ------------------------------------------------------------------
    nine_id = -1 * np.ones((H, W), dtype=int)
    blocks = []

    if np.any(wire9):
        seen = np.zeros((H, W), dtype=bool)
        for r in range(H):
            for c in range(W):
                if not wire9[r, c] or seen[r, c]:
                    continue
                q = deque([(r, c)])
                seen[r, c] = True
                cells = []
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for d in DIRS.values():
                        nr, nc = cr + d[0], cc + d[1]
                        if 0 <= nr < H and 0 <= nc < W and wire9[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                bid = len(blocks)
                rs = [x[0] for x in cells]
                cs = [x[1] for x in cells]
                blocks.append((min(rs), max(rs), min(cs), max(cs)))
                for rr, cc in cells:
                    nine_id[rr, cc] = bid

    def block_exit_primary(r, c, d):
        """
        If moving from 8/5 cell (r,c) in direction d immediately enters a 9 block,
        jump straight through the block to the next 8/5 cell, if present.
        Returns (exit_r, exit_c, cost) or None.
        """
        dr, dc = DIRS[d]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < H and 0 <= nc < W):
            return None
        if not wire9[nr, nc]:
            return None
        bid = nine_id[nr, nc]
        if bid < 0:
            return None
        r0, r1, c0, c1 = blocks[bid]

        if d == N:
            if nr != r1:
                return None
            er, ec = r0 - 1, nc
        elif d == S:
            if nr != r0:
                return None
            er, ec = r1 + 1, nc
        elif d == Wd:
            if nc != c1:
                return None
            er, ec = nr, c0 - 1
        elif d == E:
            if nc != c0:
                return None
            er, ec = nr, c1 + 1
        else:
            return None

        if 0 <= er < H and 0 <= ec < W and wire85[er, ec]:
            cost = abs(er - r) + abs(ec - c)
            return (er, ec, cost)
        return None

    # ------------------------------------------------------------------
    # Port detection
    # ------------------------------------------------------------------
    def inb(r, c):
        return 0 <= r < H and 0 <= c < W

    def port_color_at(center):
        """
        Return color of a 3x3 port centered at center, or None if no port.
        Reject regions containing wire/9 cells.
        """
        r, c = center
        if not inb(r - 1, c - 1) or not inb(r + 1, c + 1):
            return None
        sub = g[r - 1:r + 2, c - 1:c + 2]

        # Ports do not contain wire cells or 9 blocks.
        if np.any(np.isin(sub, [5, 8, 9])):
            return None

        nonzero = sub[sub != 0]
        if len(nonzero) == 0:
            return 0  # empty port
        if len(np.unique(nonzero)) != 1:
            return None
        val = int(nonzero[0])
        if val in (5, 8, 9):
            return None
        # Ignore very sparse accidental patterns.
        if len(nonzero) < 4:
            return None
        return val

    endpoint_to_port = {}
    ports = []
    seen_port_centers = set()

    # Find degree-1 8/5 cells as wire endpoints.
    for r in range(H):
        for c in range(W):
            if not wire85[r, c]:
                continue
            nbrs = []
            for d, (dr, dc) in DIRS.items():
                nr, nc = r + dr, c + dc
                if inb(nr, nc) and wire85[nr, nc]:
                    nbrs.append(d)
            if len(nbrs) != 1:
                continue

            in_dir = nbrs[0]
            dr, dc = DIRS[in_dir]
            cr, cc = r - 2 * dr, c - 2 * dc
            if not inb(cr, cc):
                continue

            color = port_color_at((cr, cc))
            if color is None:
                continue

            key = (cr, cc, in_dir)
            if key in seen_port_centers:
                continue
            seen_port_centers.add(key)

            # Output symbol shape from endpoint side:
            #   in_dir N (port below) or W (port to right) -> X
            #   in_dir S or E -> E
            is_x = (in_dir == N or in_dir == Wd)

            # For X, match the examples by putting the center on an even column.
            out_cr, out_cc = cr, cc
            if is_x and out_cc % 2 == 1:
                if inb(out_cr, out_cc + 1):
                    out_cc += 1

            pid = len(ports)
            ports.append({
                "id": pid,
                "endpoint": (r, c),
                "in_dir": in_dir,
                "center0": (cr, cc),
                "center_out": (out_cr, out_cc),
                "is_x": is_x,
                "color": None if color == 0 else int(color),
            })
            endpoint_to_port[(r, c)] = pid

    if not ports:
        return g.copy()

    empty_ports = [p for p in ports if p["color"] is None]
    colored_ports = [p for p in ports if p["color"] is not None]

    # ------------------------------------------------------------------
    # Deterministic straight-first trace through 8/5 and primary 9 jumps.
    # Used to add very strong candidate connections.
    # ------------------------------------------------------------------
    def trace_deterministic(start_pid):
        start = ports[start_pid]["endpoint"]
        if start not in endpoint_to_port:
            return None
        r, c = start
        nbrs = []
        for d, (dr, dc) in DIRS.items():
            nr, nc = r + dr, c + dc
            if inb(nr, nc) and wire85[nr, nc]:
                nbrs.append(d)
        if len(nbrs) != 1:
            return None

        in_dir = nbrs[0]
        dr, dc = DIRS[in_dir]
        cur = (r + dr, c + dc)
        prev = start
        visited = {prev, cur}
        max_steps = H * W * 4

        for _ in range(max_steps):
            if cur in endpoint_to_port and cur != start:
                return endpoint_to_port[cur]

            r, c = cur
            cand = []

            # Candidate moves from current cell, not backtracking.
            for d, (dr2, dc2) in DIRS.items():
                if d == OPP[in_dir]:
                    continue
                nr, nc = r + dr2, c + dc2
                ok = False
                nxt = None
                cost = 0
                if inb(nr, nc) and wire85[nr, nc]:
                    ok = True
                    nxt = (nr, nc)
                    cost = 1
                else:
                    jump = block_exit_primary(r, c, d)
                    if jump is not None:
                        ok = True
                        nxt = (jump[0], jump[1])
                        cost = jump[2]

                if not ok or nxt in visited:
                    continue

                # Prefer straight, then right, then left.
                if d == in_dir:
                    pref = 0
                elif d == RIGHT[in_dir]:
                    pref = 1
                elif d == LEFT[in_dir]:
                    pref = 2
                else:
                    pref = 3
                cand.append((pref, cost, d, nxt))

            if not cand:
                return None

            cand.sort(key=lambda x: (x[0], x[1]))
            _, _, d, nxt = cand[0]
            in_dir = d
            visited.add(nxt)
            prev, cur = cur, nxt

        return None

    # Build candidate edges only between empty and colored ports.
    # cost levels:
    #   0        : deterministic edge
    #   1000 + c : Dijkstra over primary network (turns allowed)
    #   20000 + c: Dijkstra over 9-as-passable network (fallback)
    edge_cost = defaultdict(lambda: float("inf"))

    for p in ports:
        t = trace_deterministic(p["id"])
        if t is None:
            continue
        a, b = p["id"], t
        pa, pb = ports[a], ports[b]
        if (pa["color"] is None) != (pb["color"] is None):
            edge_cost[(min(a, b), max(a, b))] = min(edge_cost[(min(a, b), max(a, b))], 0.0)

    # ------------------------------------------------------------------
    # Dijkstra on wire graph for fallback reachability/costs.
    # ------------------------------------------------------------------
    TURN_PENALTY = 20
    INF = float("inf")

    def dijkstra_costs(start_pid, use_9_passable=False):
        """
        Return dictionary endpoint_cell -> cost from this start endpoint.
        Primary mode: 9 blocks are straight-through jumps only.
        Fallback mode: 9 cells are ordinary passable cells.
        """
        start = ports[start_pid]["endpoint"]

        def passable(r, c):
            if use_9_passable:
                return wire85[r, c] or wire9[r, c]
            return wire85[r, c]

        dist = {}
        heap = []

        # Initial moves from endpoint.
        r0, c0 = start
        for d, (dr, dc) in DIRS.items():
            nr, nc = r0 + dr, c0 + dc
            nxt = None
            if inb(nr, nc) and passable(nr, nc):
                nxt = (nr, nc)
                cost = 1
            elif (not use_9_passable):
                jump = block_exit_primary(r0, c0, d)
                if jump is not None:
                    nxt = (jump[0], jump[1])
                    cost = jump[2]
            if nxt is not None:
                key = (nxt[0], nxt[1], d)
                if key not in dist or cost < dist[key]:
                    dist[key] = cost
                    heapq.heappush(heap, (cost, nxt[0], nxt[1], d))

        best_cell = {}

        while heap:
            cost, r, c, in_d = heapq.heappop(heap)
            if (r, c, in_d) not in dist or dist[(r, c, in_d)] < cost:
                continue

            if (r, c) in endpoint_to_port:
                if (r, c) not in best_cell or cost < best_cell[(r, c)]:
                    best_cell[(r, c)] = cost

            for nd, (dr, dc) in DIRS.items():
                if nd == OPP[in_d]:
                    continue
                nr, nc = r + dr, c + dc
                add = 1
                if nd != in_d:
                    add += TURN_PENALTY

                nxt = None
                if inb(nr, nc) and passable(nr, nc):
                    nxt = (nr, nc)
                    step_cost = add
                elif (not use_9_passable):
                    jump = block_exit_primary(r, c, nd)
                    if jump is not None:
                        nxt = (jump[0], jump[1])
                        step_cost = jump[2] + (TURN_PENALTY if nd != in_d else 0)

                if nxt is None:
                    continue
                key = (nxt[0], nxt[1], nd)
                nc2 = cost + step_cost
                if key not in dist or nc2 < dist[key]:
                    dist[key] = nc2
                    heapq.heappush(heap, (nc2, nxt[0], nxt[1], nd))

        return best_cell

    # Add primary Dijkstra edges.
    for e in empty_ports:
        if not colored_ports:
            break
        costs = dijkstra_costs(e["id"], use_9_passable=False)
        for cport in colored_ports:
            ep = cport["endpoint"]
            if ep in costs:
                idx = (min(e["id"], cport["id"]), max(e["id"], cport["id"]))
                edge_cost[idx] = min(edge_cost[idx], 1000.0 + costs[ep])

    # Fallback: treat 9 blocks as fully passable.
    for e in empty_ports:
        # If it already has a strong primary edge, still allow cheaper fallback?
        # Use fallback only if no primary edge exists yet.
        has_primary = any(
            (min(e["id"], c["id"]), max(e["id"], c["id"])) in edge_cost and
            edge_cost[(min(e["id"], c["id"]), max(e["id"], c["id"]))] < 1000.0
            for c in colored_ports
        )
        if has_primary:
            continue
        costs = dijkstra_costs(e["id"], use_9_passable=True)
        for cport in colored_ports:
            ep = cport["endpoint"]
            if ep in costs:
                idx = (min(e["id"], cport["id"]), max(e["id"], cport["id"]))
                edge_cost[idx] = min(edge_cost[idx], 20000.0 + costs[ep])

    # ------------------------------------------------------------------
    # Min-cost flow to assign colors to empty ports.
    # ------------------------------------------------------------------
    def min_cost_flow(n, s, t, edges, flow_amount):
        """
        Simple min-cost max-flow with potentials and SPFA initialization.
        edges: list of (u, v, cap, cost)
        """
        adj = [[] for _ in range(n)]

        def add(u, v, cap, cost):
            adj[u].append([v, cap, cost, len(adj[v])])
            adj[v].append([u, 0, -cost, len(adj[u]) - 1])

        for u, v, cap, cost in edges:
            add(u, v, cap, cost)

        # potentials via Bellman-Ford/SPFA
        pot = [0] * n
        for _ in range(n):
            updated = False
            for u in range(n):
                for e in adj[u]:
                    v, cap, cost, rev = e
                    if cap > 0 and pot[u] + cost < pot[v]:
                        pot[v] = pot[u] + cost
                        updated = True
            if not updated:
                break

        flow = 0
        cost_total = 0
        parent = [None] * n
        parent_edge = [None] * n

        while flow < flow_amount:
            dist = [INF] * n
            inq = [False] * n
            dist[s] = 0
            q = deque([s])
            inq[s] = True

            while q:
                u = q.popleft()
                inq[u] = False
                for i, e in enumerate(adj[u]):
                    v, cap, cost, rev = e
                    if cap > 0:
                        nd = dist[u] + cost + pot[u] - pot[v]
                        if nd < dist[v]:
                            dist[v] = nd
                            parent[v] = u
                            parent_edge[v] = i
                            if not inq[v]:
                                q.append(v)
                                inq[v] = True

            if dist[t] == INF:
                break

            for v in range(n):
                if dist[v] < INF:
                    pot[v] += dist[v]

            addf = flow_amount - flow
            v = t
            while v != s:
                u = parent[v]
                i = parent_edge[v]
                e = adj[u][i]
                addf = min(addf, e[1])
                v = u

            v = t
            while v != s:
                u = parent[v]
                i = parent_edge[v]
                e = adj[u][i]
                e[1] -= addf
                adj[v][e[3]][1] += addf
                v = u

            flow += addf
            cost_total += addf * (pot[t] - pot[s])

        return flow, cost_total

    num_empty = len(empty_ports)
    num_colored = len(colored_ports)

    if num_empty and num_colored:
        # Nodes:
        # 0 = source
        # 1..num_empty = empty ports
        # 1+num_empty .. 1+num_empty+num_colored = colored ports
        # last = sink
        S = 0
        E0 = 1
        C0 = E0 + num_empty
        T = C0 + num_colored
        n = T + 1

        mcf_edges = []
        for i, e in enumerate(empty_ports):
            mcf_edges.append((S, E0 + i, 1, 0))
            # unmatched option
            mcf_edges.append((E0 + i, T, 1, 1000000))

        for j, c in enumerate(colored_ports):
            mcf_edges.append((C0 + j, T, 1, 0))

        # Empty -> colored edges.
        for i, e in enumerate(empty_ports):
            for j, c in enumerate(colored_ports):
                idx = (min(e["id"], c["id"]), max(e["id"], c["id"]))
                if idx in edge_cost and edge_cost[idx] < float("inf"):
                    mcf_edges.append((E0 + i, C0 + j, 1, int(edge_cost[idx])))

        flow, _ = min_cost_flow(n, S, T, mcf_edges, num_empty)

        # Extract assignments by checking residual capacities.
        # Rebuild a simple view from mcf_edges? Simpler: rerun greedy on chosen?
        # Instead, during flow we can store selected edges; easier:
        # Since flow is small, use augmenting path tracking again to record assignment.
        # We'll perform a second min-cost max-flow with parent reconstruction.
        def mcf_assign(n, s, t, edges, flow_amount):
            adj = [[] for _ in range(n)]

            def add(u, v, cap, cost):
                adj[u].append([v, cap, cost, len(adj[v])])
                adj[v].append([u, 0, -cost, len(adj[u]) - 1])

            for u, v, cap, cost in edges:
                add(u, v, cap, cost)

            pot = [0] * n
            for _ in range(n):
                updated = False
                for u in range(n):
                    for e in adj[u]:
                        v, cap, cost, rev = e
                        if cap > 0 and pot[u] + cost < pot[v]:
                            pot[v] = pot[u] + cost
                            updated = True
                if not updated:
                    break

            flow = 0
            parent = [None] * n
            parent_edge = [None] * n
            assign = {}

            while flow < flow_amount:
                dist = [INF] * n
                inq = [False] * n
                dist[s] = 0
                q = deque([s])
                inq[s] = True

                while q:
                    u = q.popleft()
                    inq[u] = False
                    for i, e in enumerate(adj[u]):
                        v, cap, cost, rev = e
                        if cap > 0:
                            nd = dist[u] + cost + pot[u] - pot[v]
                            if nd < dist[v]:
                                dist[v] = nd
                                parent[v] = u
                                parent_edge[v] = i
                                if not inq[v]:
                                    q.append(v)
                                    inq[v] = True

                if dist[t] == INF:
                    break

                for v in range(n):
                    if dist[v] < INF:
                        pot[v] += dist[v]

                addf = flow_amount - flow
                v = t
                while v != s:
                    u = parent[v]
                    i = parent_edge[v]
                    e = adj[u][i]
                    addf = min(addf, e[1])
                    v = u

                v = t
                while v != s:
                    u = parent[v]
                    i = parent_edge[v]
                    e = adj[u][i]
                    e[1] -= addf
                    adj[v][e[3]][1] += addf
                    if E0 <= u < E0 + num_empty and C0 <= v < C0 + num_colored:
                        # u is empty node, v colored node
                        assign[u - E0] = v - C0
                    v = u

                flow += addf

            return assign

        assignment = mcf_assign(n, S, T, mcf_edges, num_empty)

        for i, j in assignment.items():
            e = empty_ports[i]
            c = colored_ports[j]
            if c["color"] is not None:
                e["color"] = c["color"]

    # ------------------------------------------------------------------
    # Draw output symbols.
    # ------------------------------------------------------------------
    out = g.copy()
    is_wire = (g == 5) | (g == 8) | (g == 9)

    # Collect all port regions to clear (original and shifted output).
    clear_regions = []
    for p in ports:
        clear_regions.append(p["center0"])
        if p["center_out"] != p["center0"]:
            clear_regions.append(p["center_out"])

    for r, c in clear_regions:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if inb(rr, cc) and not is_wire[rr, cc]:
                    out[rr, cc] = 0

    # Draw final colored symbols.
    for p in ports:
        if p["color"] is None:
            continue
        col = int(p["color"])
        r, c = p["center_out"]
        if not inb(r, c):
            continue

        if p["is_x"]:
            cells = [
                (r - 1, c - 1),
                (r - 1, c + 1),
                (r, c),
                (r + 1, c - 1),
                (r + 1, c + 1),
            ]
        else:
            cells = [
                (r - 1, c),
                (r + 1, c),
                (r, c - 1),
                (r, c + 1),
            ]

        for rr, cc in cells:
            if inb(rr, cc) and not is_wire[rr, cc]:
                out[rr, cc] = col

    return out