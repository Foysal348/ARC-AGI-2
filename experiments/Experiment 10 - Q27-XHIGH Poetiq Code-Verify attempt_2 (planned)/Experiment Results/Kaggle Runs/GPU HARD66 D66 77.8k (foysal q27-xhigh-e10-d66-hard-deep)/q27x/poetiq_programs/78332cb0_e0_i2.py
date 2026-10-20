import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reconnect separated tiles into a single straight line.

    Tiles are separated by full rows/columns of 6.  Each tile may have
    non-background pixels touching its border.  A vertical output order
    connects a tile's bottom-border pixels to the next tile's top-border
    pixels.  A horizontal output order connects right-border pixels to
    left-border pixels.

    The function finds a path through all tiles in one of those two
    directions, preferring a path whose ends have no unmatched stubs and
    whose internal connections have as little unmatched stubbing as
    possible.  The selected tiles are then placed in order, with a full
    row/column of 6 between consecutive tiles.
    """
    g = np.array(grid, dtype=int, copy=True)

    if g.ndim != 2:
        return g

    H, W = g.shape
    if H == 0 or W == 0:
        return g

    def split_intervals(seps, length):
        """Split [0, length) by separator indices into inclusive intervals."""
        seps = sorted(set(seps))
        intervals = []
        prev = -1
        for s in seps:
            start = prev + 1
            end = s - 1
            if start <= end:
                intervals.append((start, end))
            prev = s

        start = prev + 1
        end = length - 1
        if start <= end:
            intervals.append((start, end))

        return intervals

    def full_lines(color):
        """Return full row indices and full column indices of a color."""
        if H == 0:
            rows = np.array([], dtype=int)
        else:
            rows = np.where(np.all(g == color, axis=1))[0]

        if W == 0:
            cols = np.array([], dtype=int)
        else:
            cols = np.where(np.all(g == color, axis=0))[0]

        return rows, cols

    # In this task the separator color is normally 6.  Fall back to a
    # dynamic detector for robustness.
    sep_color = None
    if np.any(g == 6):
        rows6, cols6 = full_lines(6)
        if len(rows6) > 0 or len(cols6) > 0:
            sep_color = 6

    if sep_color is None:
        vals, counts = np.unique(g, return_counts=True)
        bg_tmp = int(vals[np.argmax(counts)]) if vals.size else 0
        best = -1

        for c in vals:
            c = int(c)
            if c == bg_tmp:
                continue
            rows_c, cols_c = full_lines(c)
            score = len(rows_c) * W + len(cols_c) * H
            if score > best:
                best = score
                sep_color = c

        if sep_color is None:
            return g

    # Background is usually 7 in this task.  If 7 is present, use it;
    # otherwise use the most frequent non-separator color.
    vals, counts = np.unique(g, return_counts=True)
    if 7 in vals and 7 != sep_color:
        bg = 7
    else:
        non_sep_mask = vals != sep_color
        if non_sep_mask.any():
            bg = int(vals[non_sep_mask][np.argmax(counts[non_sep_mask])])
        else:
            bg = 0

    sep_rows = [int(r) for r in np.where(np.all(g == sep_color, axis=1))[0]]
    sep_cols = [int(c) for c in np.where(np.all(g == sep_color, axis=0))[0]]

    row_intervals = split_intervals(sep_rows, H)
    col_intervals = split_intervals(sep_cols, W)

    if not row_intervals or not col_intervals:
        return g

    panels = []

    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            sub = g[r0:r1 + 1, c0:c1 + 1]
            if sub.size == 0:
                continue

            obj_mask = (sub != bg) & (sub != sep_color)
            if not obj_mask.any():
                continue

            top = set(np.where(obj_mask[0, :])[0].tolist())
            bottom = set(np.where(obj_mask[-1, :])[0].tolist())
            left = set(np.where(obj_mask[:, 0])[0].tolist())
            right = set(np.where(obj_mask[:, -1])[0].tolist())

            # A tile with no border stubs cannot be part of the chain.
            if not (top or bottom or left or right):
                continue

            panels.append({
                "grid": sub,
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
            })

    n = len(panels)
    if n == 0:
        return g
    if n == 1:
        return panels[0]["grid"].copy()

    def build_adj(out_key, in_key):
        """
        Build a directed graph.
        Edge i -> j means tile i's out_key stubs can touch tile j's in_key stubs.
        """
        adj = [[] for _ in range(n)]
        score = [[0] * n for _ in range(n)]

        for i in range(n):
            out_set = panels[i][out_key]
            if not out_set:
                continue

            for j in range(n):
                if i == j:
                    continue

                in_set = panels[j][in_key]
                if not in_set:
                    continue

                if out_set & in_set:
                    adj[i].append(j)
                    # Penalize unmatched endpoint coordinates.
                    score[i][j] = len(out_set.symmetric_difference(in_set))

        return adj, score

    def find_best_path(adj, start_candidates, end_candidates,
                       edge_score, start_in_lens, end_out_lens, limit=2000):
        """
        Find a Hamiltonian path.  If multiple paths exist, prefer the one
        with the smallest unmatched-stub penalty.
        """
        if n == 1:
            return [0]

        outdeg = [len(adj[i]) for i in range(n)]
        indeg = [0] * n
        for i in range(n):
            for j in adj[i]:
                indeg[j] += 1

        starts = [i for i in start_candidates if outdeg[i] > 0]
        ends = set(end_candidates)

        if not starts or not ends:
            return None

        full = (1 << n) - 1

        next_order = [
            sorted(adj[i], key=lambda j: (edge_score[i][j], outdeg[j], indeg[j], j))
            for i in range(n)
        ]
        starts.sort(key=lambda i: (indeg[i], outdeg[i], i))

        best_path = None
        best_score = None
        complete = 0
        stop = False
        fail = set()

        def has_out_to_unvisited(v, mask):
            for w in adj[v]:
                if not (mask & (1 << w)):
                    return True
            return False

        def dfs(u, mask, path, partial_score):
            nonlocal best_path, best_score, complete, stop

            if stop:
                return

            state = (u, mask)
            if state in fail:
                return

            if mask == full:
                if u in ends:
                    total_score = partial_score + end_out_lens[u]
                    complete += 1

                    if best_score is None or total_score < best_score:
                        best_score = total_score
                        best_path = path[:]
                        if best_score == 0:
                            stop = True

                    if complete >= limit:
                        stop = True
                    return

                fail.add(state)
                return

            # Prune if more than one unvisited node cannot move forward.
            dead_count = 0
            dead_node = -1
            for v in range(n):
                if mask & (1 << v):
                    continue
                if not has_out_to_unvisited(v, mask):
                    dead_count += 1
                    dead_node = v
                    if dead_count > 1:
                        break

            if dead_count > 1:
                fail.add(state)
                return

            # The only dead end must be an allowed final node.
            if dead_count == 1 and dead_node not in ends:
                fail.add(state)
                return

            candidates = []
            for v in next_order[u]:
                if mask & (1 << v):
                    continue

                # If v would not be the final node, it must be able to continue.
                if (mask | (1 << v)) != full:
                    if not has_out_to_unvisited(v, mask):
                        continue

                candidates.append(v)

            if not candidates:
                fail.add(state)
                return

            for v in candidates:
                new_score = partial_score + edge_score[u][v]

                # If we already have a no-worse path, no need to explore worse ones.
                if best_score is not None and new_score >= best_score:
                    continue

                path.append(v)
                dfs(v, mask | (1 << v), path, new_score)
                path.pop()

                if stop:
                    return

            fail.add(state)
            return

        for s in starts:
            if best_score is not None and start_in_lens[s] >= best_score:
                continue

            path = [s]
            dfs(s, 1 << s, path, start_in_lens[s])

            if stop:
                break

        return best_path

    def try_orientation(out_key, in_key, other_keys, strict):
        """
        Try to build a chain.
        strict=True requires the first tile to have no in_key stubs and the
        last tile to have no out_key stubs.
        """
        adj, edge_score = build_adj(out_key, in_key)

        in_lens = [len(panels[i][in_key]) for i in range(n)]
        out_lens = [len(panels[i][out_key]) for i in range(n)]

        if strict:
            starts = [i for i in range(n) if in_lens[i] == 0]
            ends = [i for i in range(n) if out_lens[i] == 0]
        else:
            starts = [i for i in range(n) if len(adj[i]) > 0]
            ends = list(range(n))

        path = find_best_path(adj, starts, ends, edge_score, in_lens, out_lens)
        if path is None:
            return None

        irrelevant = 0
        for i in range(n):
            for k in other_keys:
                irrelevant += len(panels[i][k])

        dangling = len(panels[path[0]][in_key]) + len(panels[path[-1]][out_key])
        for a, b in zip(path, path[1:]):
            dangling += edge_score[a][b]

        total_edges = sum(len(a) for a in adj)

        # If the input is wider than tall, prefer vertical output;
        # if taller than wide, prefer horizontal; square ties prefer vertical.
        if len(col_intervals) > len(row_intervals):
            preferred = "v"
        elif len(row_intervals) > len(col_intervals):
            preferred = "h"
        else:
            preferred = "v"

        orient_name = "v" if out_key == "bottom" else "h"
        tie = 0 if orient_name == preferred else 1

        return (irrelevant, dangling, total_edges, 0 if strict else 1, tie, path)

    strict_candidates = []
    relaxed_candidates = []

    # Vertical: bottom -> top.
    res = try_orientation("bottom", "top", ("left", "right"), True)
    if res is not None:
        strict_candidates.append(("v", res))
    else:
        res = try_orientation("bottom", "top", ("left", "right"), False)
        if res is not None:
            relaxed_candidates.append(("v", res))

    # Horizontal: right -> left.
    res = try_orientation("right", "left", ("top", "bottom"), True)
    if res is not None:
        strict_candidates.append(("h", res))
    else:
        res = try_orientation("right", "left", ("top", "bottom"), False)
        if res is not None:
            relaxed_candidates.append(("h", res))

    if strict_candidates:
        strict_candidates.sort(key=lambda item: item[1][:5])
        orientation = strict_candidates[0][0]
        path = strict_candidates[0][1][5]
    elif relaxed_candidates:
        relaxed_candidates.sort(key=lambda item: item[1][:5])
        orientation = relaxed_candidates[0][0]
        path = relaxed_candidates[0][1][5]
    else:
        # Fallback: preserve reading order and choose a line direction
        # based on the input layout.
        if len(row_intervals) == 1:
            orientation = "v"
        elif len(col_intervals) == 1:
            orientation = "h"
        else:
            orientation = "v"

        path = list(range(n))

    selected = [panels[i]["grid"] for i in path]

    if not selected:
        return g

    if orientation == "v":
        out_h = sum(sub.shape[0] for sub in selected) + (len(selected) - 1)
        out_w = max(sub.shape[1] for sub in selected)
        out = np.full((out_h, out_w), bg, dtype=int)

        r = 0
        for idx, sub in enumerate(selected):
            sh, sw = sub.shape
            out[r:r + sh, :sw] = sub
            r += sh

            if idx != len(selected) - 1:
                out[r, :] = sep_color
                r += 1

        return out

    else:
        out_h = max(sub.shape[0] for sub in selected)
        out_w = sum(sub.shape[1] for sub in selected) + (len(selected) - 1)
        out = np.full((out_h, out_w), bg, dtype=int)

        c = 0
        for idx, sub in enumerate(selected):
            sh, sw = sub.shape
            out[:sh, c:c + sw] = sub
            c += sw

            if idx != len(selected) - 1:
                out[:, c] = sep_color
                c += 1

        return out