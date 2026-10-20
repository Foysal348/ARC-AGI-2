import numpy as np


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Assemble rigid color pieces around the fixed color-4 piece.

    Each piece consists of all cells of one color > 2. Cells with value 2
    that are orthogonally adjacent to a piece are its connection ports.
    Pieces are translated only. In the output, ports from different pieces
    coincide. The color-4 piece remains fixed.
    """
    grid = np.array(grid, dtype=int, copy=True)
    if grid.ndim != 2:
        return grid

    H, W = grid.shape
    if H == 0 or W == 0:
        return grid

    def is_solid_piece(shape_set):
        if not shape_set:
            return False
        h = max(r for r, c in shape_set) + 1
        w = max(c for r, c in shape_set) + 1
        return len(shape_set) == h * w

    # ------------------------------------------------------------------
    # Parse pieces by color. A color may have several disconnected parts;
    # all cells of the same color move together.
    # ------------------------------------------------------------------
    pieces = []
    colors = [int(c) for c in np.unique(grid) if int(c) > 2]

    for color in colors:
        ys, xs = np.where(grid == color)
        if len(ys) == 0:
            continue

        top_left = (int(ys.min()), int(xs.min()))
        shape_set = {
            (int(y - top_left[0]), int(x - top_left[1]))
            for y, x in zip(ys, xs)
        }

        ports_set = set()
        for y, x in zip(ys, xs):
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = int(y) + dy, int(x) + dx
                if 0 <= ny < H and 0 <= nx < W and grid[ny, nx] == 2:
                    ports_set.add((ny - top_left[0], nx - top_left[1]))

        # A port can never be a shape cell, but guard against weird input.
        ports_set = {p for p in ports_set if p not in shape_set}

        all_rel = shape_set | ports_set
        min_dr = min(r for r, c in all_rel)
        max_dr = max(r for r, c in all_rel)
        min_dc = min(c for r, c in all_rel)
        max_dc = max(c for r, c in all_rel)

        pieces.append({
            "color": color,
            "orig": top_left,
            "shape": shape_set,
            "shape_list": list(shape_set),
            "ports": ports_set,
            "ports_list": list(ports_set),
            "min_dr": min_dr,
            "max_dr": max_dr,
            "min_dc": min_dc,
            "max_dc": max_dc,
            "n_cells": len(shape_set),
        })

    if not pieces:
        return np.zeros((H, W), dtype=int)

    # ------------------------------------------------------------------
    # Choose anchor. The intended anchor is color 4. If absent, fall back
    # to the largest solid rectangular piece, then largest piece.
    # ------------------------------------------------------------------
    anchor = None
    for i, p in enumerate(pieces):
        if p["color"] == 4:
            anchor = i
            break

    if anchor is None:
        solid = [i for i, p in enumerate(pieces) if is_solid_piece(p["shape"])]
        if solid:
            anchor = max(solid, key=lambda i: pieces[i]["n_cells"])
        else:
            anchor = max(range(len(pieces)), key=lambda i: pieces[i]["n_cells"])

    def build_output(solution):
        out = np.zeros((H, W), dtype=int)

        for pid, t in solution.items():
            piece = pieces[pid]
            for dr, dc in piece["shape_list"]:
                r, c = t[0] + dr, t[1] + dc
                if 0 <= r < H and 0 <= c < W:
                    out[r, c] = piece["color"]

        for pid, t in solution.items():
            piece = pieces[pid]
            for dr, dc in piece["ports_list"]:
                r, c = t[0] + dr, t[1] + dc
                if 0 <= r < H and 0 <= c < W and out[r, c] == 0:
                    out[r, c] = 2

        return out

    # If there is nothing to move, return the normalized anchor output.
    if len(pieces) == 1:
        return build_output({anchor: pieces[anchor]["orig"]})

    def in_bounds(pid, t):
        p = pieces[pid]
        return (
            0 <= t[0] + p["min_dr"]
            and t[0] + p["max_dr"] < H
            and 0 <= t[1] + p["min_dc"]
            and t[1] + p["max_dc"] < W
        )

    def shape_abs(pid, t):
        return {(t[0] + dr, t[1] + dc) for dr, dc in pieces[pid]["shape_list"]}

    def port_abs(pid, t):
        return {(t[0] + dr, t[1] + dc) for dr, dc in pieces[pid]["ports_list"]}

    def find_solution(require_exact2, require_all_ports, require_connected):
        n_pieces = len(pieces)

        trans = {anchor: pieces[anchor]["orig"]}
        placed_ids = [anchor]
        placed_set = {anchor}
        placed_shape = shape_abs(anchor, trans[anchor])
        port_counts = {}

        for cell in port_abs(anchor, trans[anchor]):
            port_counts[cell] = 1

        def get_candidates(pid, trans, placed_shape, port_counts):
            piece = pieces[pid]
            cand_dict = {}
            port_keys = set(port_counts.keys())

            # Fallback for pieces with no ports: try keeping them in place.
            if not piece["ports"]:
                t = piece["orig"]
                if in_bounds(pid, t):
                    sa = shape_abs(pid, t)
                    if not (sa & placed_shape) and not (sa & port_keys):
                        pa = port_abs(pid, t)
                        if not (pa & placed_shape):
                            cand_dict[t] = (sa, pa)

            # Generate translations by making one port coincide with one
            # already placed port.
            if piece["ports"]:
                for qid, qt in trans.items():
                    qpiece = pieces[qid]
                    if not qpiece["ports"]:
                        continue

                    for qr in qpiece["ports_list"]:
                        qcell = (qt[0] + qr[0], qt[1] + qr[1])
                        for ar in piece["ports_list"]:
                            t = (qcell[0] - ar[0], qcell[1] - ar[1])
                            if t in cand_dict:
                                continue
                            if not in_bounds(pid, t):
                                continue

                            sa = shape_abs(pid, t)

                            # Shape cannot overlap placed shapes or ports.
                            if sa & placed_shape:
                                continue
                            if sa & port_keys:
                                continue

                            pa = port_abs(pid, t)

                            # Port cannot overlap a placed shape.
                            if pa & placed_shape:
                                continue

                            total_match = 0
                            open_match = 0
                            valid = True

                            for cell in pa:
                                cnt = port_counts.get(cell, 0)
                                if cnt:
                                    total_match += 1
                                    if cnt == 1:
                                        open_match += 1
                                    elif cnt >= 2 and require_exact2:
                                        valid = False
                                        break

                            if not valid or total_match == 0:
                                continue

                            cand_dict[t] = (sa, pa)

            candidates = []
            for t, (sa, pa) in cand_dict.items():
                cost = piece["n_cells"] * (
                    abs(t[0] - piece["orig"][0]) + abs(t[1] - piece["orig"][1])
                )
                open_match = sum(1 for cell in pa if port_counts.get(cell, 0) == 1)
                total_match = sum(1 for cell in pa if port_counts.get(cell, 0) > 0)
                candidates.append((cost, -open_match, -total_match, t, sa, pa))

            candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
            return candidates

        def final_valid(trans):
            # Check shape overlaps and bounds.
            shape_owner = {}
            for pid, t in trans.items():
                for dr, dc in pieces[pid]["shape_list"]:
                    cell = (t[0] + dr, t[1] + dc)
                    if not (0 <= cell[0] < H and 0 <= cell[1] < W):
                        return False
                    if cell in shape_owner:
                        return False
                    shape_owner[cell] = pid

            # Collect port sharing.
            port_pids = {}
            for pid, t in trans.items():
                for dr, dc in pieces[pid]["ports_list"]:
                    cell = (t[0] + dr, t[1] + dc)
                    if not (0 <= cell[0] < H and 0 <= cell[1] < W):
                        return False
                    if cell in shape_owner:
                        return False
                    port_pids.setdefault(cell, set()).add(pid)

            if require_all_ports:
                for pids in port_pids.values():
                    if len(pids) < 2:
                        return False

            if require_exact2:
                for pids in port_pids.values():
                    if len(pids) != 2:
                        return False

            if require_connected:
                adj = {pid: set() for pid in trans}
                for pids in port_pids.values():
                    if len(pids) >= 2:
                        lst = list(pids)
                        for i in range(len(lst)):
                            for j in range(i + 1, len(lst)):
                                adj[lst[i]].add(lst[j])
                                adj[lst[j]].add(lst[i])

                seen = {anchor}
                stack = [anchor]
                while stack:
                    u = stack.pop()
                    for v in adj[u]:
                        if v not in seen:
                            seen.add(v)
                            stack.append(v)

                # Pieces without ports cannot be connected; exempt them.
                for pid in trans:
                    if pid not in seen and pieces[pid]["ports"]:
                        return False

            return True

        def dfs(placed_ids, placed_set, placed_shape, port_counts, trans):
            if len(placed_ids) == n_pieces:
                if final_valid(trans):
                    return trans.copy()
                return None

            best_pid = None
            best_cands = None

            for pid in range(n_pieces):
                if pid in placed_set:
                    continue

                cands = get_candidates(pid, trans, placed_shape, port_counts)
                if not cands:
                    continue

                if best_cands is None or len(cands) < len(best_cands):
                    best_pid = pid
                    best_cands = cands
                    if len(cands) == 1:
                        break

            if best_pid is None:
                return None

            for cand in best_cands:
                t, sa, pa = cand[3], cand[4], cand[5]

                # Apply placement.
                undo_ports = []
                for cell in pa:
                    port_counts[cell] = port_counts.get(cell, 0) + 1
                    undo_ports.append(cell)

                placed_ids.append(best_pid)
                placed_set.add(best_pid)
                trans[best_pid] = t

                result = dfs(
                    placed_ids,
                    placed_set,
                    placed_shape | sa,
                    port_counts,
                    trans,
                )

                if result is not None:
                    return result

                # Undo placement.
                placed_ids.pop()
                placed_set.remove(best_pid)
                del trans[best_pid]

                for cell in undo_ports:
                    if port_counts[cell] == 1:
                        del port_counts[cell]
                    else:
                        port_counts[cell] -= 1

            return None

        return dfs(placed_ids, placed_set, placed_shape, port_counts, trans)

    # Try increasingly relaxed constraints.
    solution = find_solution(True, True, True)
    if solution is None:
        solution = find_solution(False, True, True)
    if solution is None:
        solution = find_solution(False, False, True)
    if solution is None:
        solution = find_solution(False, False, False)

    # If the solver fails, fall back to the original piece positions.
    if solution is None:
        solution = {i: pieces[i]["orig"] for i in range(len(pieces))}

    return build_output(solution)