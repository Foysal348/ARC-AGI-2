import numpy as np


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the colored pieces around the color-4 anchor.

    Each color > 2 is treated as one rigid piece.  Cells of color 2 that are
    orthogonally adjacent to a piece are that piece's "ports".  Pieces are
    translated only.  The solver chooses translations that maximize the number
    of coincident port pairs between different pieces, while avoiding shape
    overlaps and shape/port overlaps.

    After a relative assembly is found, it is placed in the output grid with
    these preferences:
      * keep the anchor's original column if possible;
      * keep the anchor's original row if possible;
      * ensure at least 4 empty rows above the assembled pattern when possible.
    """
    grid = np.array(grid, dtype=int, copy=True)
    if grid.ndim != 2:
        return grid

    H, W = grid.shape
    if H == 0 or W == 0:
        return grid

    # ------------------------------------------------------------------
    # Parse pieces.
    # A piece is all cells of one color > 2.  Its ports are the color-2 cells
    # orthogonally adjacent to it.
    # ------------------------------------------------------------------
    pieces = []
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for color in sorted(int(c) for c in np.unique(grid)):
        if color <= 2:
            continue

        ys, xs = np.where(grid == color)
        if len(ys) == 0:
            continue

        min_r = int(ys.min())
        min_c = int(xs.min())

        shape = set()
        for y, x in zip(ys, xs):
            shape.add((int(y - min_r), int(x - min_c)))

        ports = set()
        for y, x in zip(ys, xs):
            for dy, dx in dirs:
                ny, nx = int(y) + dy, int(x) + dx
                if 0 <= ny < H and 0 <= nx < W and grid[ny, nx] == 2:
                    ports.add((int(ny - min_r), int(nx - min_c)))

        # A port should not also be a shape cell, but guard against odd input.
        ports = {p for p in ports if p not in shape}

        if not shape:
            continue

        shape_list = list(shape)
        port_list = list(ports)

        pieces.append({
            "color": color,
            "orig": (min_r, min_c),
            "shape_set": shape,
            "shape_list": shape_list,
            "port_set": ports,
            "port_list": port_list,
            "all_list": shape_list + port_list,
        })

    if not pieces:
        return np.zeros((H, W), dtype=int)

    N = len(pieces)

    # Anchor: color 4 if present, otherwise the largest piece.
    anchor = None
    for i, p in enumerate(pieces):
        if p["color"] == 4:
            anchor = i
            break
    if anchor is None:
        anchor = max(range(N), key=lambda i: len(pieces[i]["shape_list"]))

    anchor_orig = pieces[anchor]["orig"]
    rel_orig = [
        (pieces[i]["orig"][0] - anchor_orig[0],
         pieces[i]["orig"][1] - anchor_orig[1])
        for i in range(N)
    ]

    # ------------------------------------------------------------------
    # Pairwise maximum possible port coincidences, used only for pruning.
    # ------------------------------------------------------------------
    max_pair = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            cnt = {}
            for a in pieces[i]["port_list"]:
                for b in pieces[j]["port_list"]:
                    off = (a[0] - b[0], a[1] - b[1])
                    cnt[off] = cnt.get(off, 0) + 1
            if cnt:
                m = max(cnt.values())
                max_pair[i][j] = max_pair[j][i] = m

    # ------------------------------------------------------------------
    # Placement normalization.
    # ------------------------------------------------------------------
    default_top_margin = min(4, H)

    def get_placement(t, top_margin=default_top_margin):
        """
        Convert relative translations (anchor at (0,0)) to absolute row/col
        offsets.  Returns (row_offset, col_offset, penalty) or None.
        """
        min_r = min(t[i][0] + c[0] for i in range(N) for c in pieces[i]["all_list"])
        max_r = max(t[i][0] + c[0] for i in range(N) for c in pieces[i]["all_list"])
        min_c = min(t[i][1] + c[1] for i in range(N) for c in pieces[i]["all_list"])
        max_c = max(t[i][1] + c[1] for i in range(N) for c in pieces[i]["all_list"])

        # Rows.
        # Keep the anchor row when possible, but require the first non-zero
        # row to be at least top_margin when feasible.
        lo_r = max(-min_r, top_margin - min_r)
        hi_r = H - 1 - max_r
        if lo_r > hi_r:
            return None
        desired_r = anchor_orig[0]
        if desired_r < lo_r:
            gr = lo_r
        elif desired_r > hi_r:
            gr = hi_r
        else:
            gr = desired_r

        # Columns.
        lo_c = -min_c
        hi_c = W - 1 - max_c
        if lo_c > hi_c:
            return None
        desired_c = anchor_orig[1]
        if desired_c < lo_c:
            gc = lo_c
        elif desired_c > hi_c:
            gc = hi_c
        else:
            gc = desired_c

        penalty = abs(gr - desired_r) + abs(gc - desired_c)
        return gr, gc, penalty

    # ------------------------------------------------------------------
    # Search state / DFS.
    # ------------------------------------------------------------------
    best_shared = -1
    best_movement = None
    best_fit = None
    best_open = None
    best_t = None

    def upper_bound(placed_set, unplaced):
        ub = 0
        for i in placed_set:
            for j in unplaced:
                ub += max_pair[i][j]
        for a in range(len(unplaced)):
            for b in range(a + 1, len(unplaced)):
                ub += max_pair[unplaced[a]][unplaced[b]]
        return ub

    def get_candidates(p, t, placed_shape, port_count):
        """
        Candidate translations for piece p that make at least one of its ports
        coincide with a port of an already placed piece.
        """
        piece = pieces[p]
        if not piece["port_list"]:
            return []

        shape_list = piece["shape_list"]
        port_list = piece["port_list"]
        cands_dict = {}

        for cell in list(port_count.keys()):
            cr, cc = cell
            for pr, pc in port_list:
                tr = cr - pr
                tc = cc - pc
                key = (tr, tc)
                if key in cands_dict:
                    continue

                sa = set()
                valid = True
                for sr, sc in shape_list:
                    cell2 = (tr + sr, tc + sc)
                    if cell2 in placed_shape or cell2 in port_count:
                        valid = False
                        break
                    sa.add(cell2)
                if not valid:
                    continue

                pa = set()
                for rr, rc in port_list:
                    cell2 = (tr + rr, tc + rc)
                    if cell2 in placed_shape:
                        valid = False
                        break
                    pa.add(cell2)
                if not valid:
                    continue

                if sa & pa:
                    continue

                new_shared = 0
                for cell2 in pa:
                    if cell2 in port_count:
                        new_shared += len(port_count[cell2])

                move = abs(tr - rel_orig[p][0]) + abs(tc - rel_orig[p][1])
                cands_dict[key] = ((tr, tc), sa, pa, new_shared, move)

        cands = list(cands_dict.values())
        # Prefer high immediate shared-port count, then low movement.
        cands.sort(key=lambda x: (-x[3], x[4], x[0]))
        return cands

    def dfs(placed_set, t, placed_shape, port_count, shared, movement):
        nonlocal best_shared, best_movement, best_fit, best_open, best_t

        if len(placed_set) == N:
            # Prefer the default 4-row top margin, but fall back to no margin
            # if the relative solution would otherwise not fit.
            placement = get_placement(t, default_top_margin)
            margin_used = default_top_margin
            if placement is None:
                placement = get_placement(t, 0)
                margin_used = 0
            if placement is None:
                return

            gr, gc, fit_pen = placement
            if margin_used != default_top_margin:
                fit_pen += 1000

            open_count = sum(1 for v in port_count.values() if len(v) == 1)

            better = False
            if best_shared == -1:
                better = True
            elif shared != best_shared:
                better = shared > best_shared
            elif movement != best_movement:
                better = movement < best_movement
            elif fit_pen != best_fit:
                better = fit_pen < best_fit
            elif open_count != best_open:
                better = open_count < best_open

            if better:
                best_shared = shared
                best_movement = movement
                best_fit = fit_pen
                best_open = open_count
                best_t = t.copy()
            return

        unplaced = [i for i in range(N) if i not in placed_set]

        if best_shared != -1 and shared + upper_bound(placed_set, unplaced) < best_shared:
            return

        cand_map = {}
        for p in unplaced:
            cands = get_candidates(p, t, placed_shape, port_count)
            if cands:
                cand_map[p] = cands

        if not cand_map:
            return

        # MRV: place the piece with fewest candidates; tie-break by score.
        chosen = min(
            cand_map.items(),
            key=lambda kv: (len(kv[1]), -max(c[3] for c in kv[1]))
        )
        p = chosen[0]

        for cand in chosen[1]:
            tr, tc, sa, pa, ns, mv = cand

            # Apply placement.
            t[p] = (tr, tc)
            placed_set.add(p)

            for cell in sa:
                placed_shape.add(cell)

            for cell in pa:
                if cell in port_count:
                    port_count[cell].add(p)
                else:
                    port_count[cell] = {p}

            dfs(placed_set, t, placed_shape, port_count, shared + ns, movement + mv)

            # Undo placement.
            del t[p]
            placed_set.remove(p)

            for cell in pa:
                s = port_count[cell]
                s.remove(p)
                if not s:
                    del port_count[cell]

            for cell in sa:
                placed_shape.remove(cell)

    # Initial state: anchor at (0, 0).
    initial_t = {anchor: (0, 0)}
    initial_shape = set(pieces[anchor]["shape_set"])
    initial_port_count = {}
    for cell in pieces[anchor]["port_set"]:
        initial_port_count[cell] = {anchor}

    dfs({anchor}, initial_t, initial_shape, initial_port_count, 0, 0)

    # ------------------------------------------------------------------
    # Fallback: keep original relative layout if the connected search failed.
    # ------------------------------------------------------------------
    if best_t is None:
        best_t = {i: rel_orig[i] for i in range(N)}

    placement = get_placement(best_t, default_top_margin)
    if placement is None:
        placement = get_placement(best_t, 0)
    if placement is None:
        # Last resort: draw as much as possible without normalization.
        placement = (0, 0, 0)

    gr, gc, _ = placement
    out = np.zeros((H, W), dtype=int)

    # Draw shapes first.
    for i in range(N):
        tr, tc = best_t[i]
        ar = tr + gr
        ac = tc + gc
        for dr, dc in pieces[i]["shape_list"]:
            r = ar + dr
            c = ac + dc
            if 0 <= r < H and 0 <= c < W:
                out[r, c] = pieces[i]["color"]

    # Then draw ports as 2 where empty.
    for i in range(N):
        tr, tc = best_t[i]
        ar = tr + gr
        ac = tc + gc
        for dr, dc in pieces[i]["port_list"]:
            r = ar + dr
            c = ac + dc
            if 0 <= r < H and 0 <= c < W and out[r, c] == 0:
                out[r, c] = 2

    return out