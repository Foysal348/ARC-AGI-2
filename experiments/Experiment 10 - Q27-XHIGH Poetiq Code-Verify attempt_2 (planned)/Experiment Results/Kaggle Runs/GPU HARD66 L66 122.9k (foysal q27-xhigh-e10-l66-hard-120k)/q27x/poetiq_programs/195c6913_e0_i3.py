import numpy as np
from collections import deque


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Recover the one-pixel guide line hidden inside the large colored region.

    Interpretation used by the examples:
      * The border majority color is the background.
      * The other large color is the "thick field" to be thinned.
      * A small set of isolated 2x2 blocks defines:
          - a repeating color sequence (top row of 2x2 blocks, left to right),
          - a marker color (the remaining isolated 2x2 block).
      * Single border pixels of the first sequence color are anchors.
      * The output keeps the original grid, removes the legend blocks, draws a
        1-pixel skeleton of the thick field connected to the anchors, colors the
        skeleton by BFS distance from the anchors using the repeating sequence,
        and places marker cells just outside the field at exposed path ends.
    """

    g = np.array(grid, dtype=int, copy=True)
    if g.ndim != 2:
        return g

    H, W = g.shape
    if H < 2 or W < 2:
        return g

    out = g.copy()
    dirs4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def inb(r, c):
        return 0 <= r < H and 0 <= c < W

    # ----------------------------------------------------------------------
    # Basic component helpers
    # ----------------------------------------------------------------------

    def components(mask):
        """4-connected components as lists of coordinates."""
        if not mask.any():
            return []

        label = np.full((H, W), -1, dtype=int)
        comps = []

        for r in range(H):
            for c in range(W):
                if not mask[r, c] or label[r, c] != -1:
                    continue

                cid = len(comps)
                q = deque([(r, c)])
                label[r, c] = cid
                cells = []

                while q:
                    x, y = q.popleft()
                    cells.append((x, y))
                    for dx, dy in dirs4:
                        nx, ny = x + dx, y + dy
                        if inb(nx, ny) and mask[nx, ny] and label[nx, ny] == -1:
                            label[nx, ny] = cid
                            q.append((nx, ny))

                comps.append(cells)

        return comps

    def build_graph(cells):
        """4-neighbor adjacency for a set of cells."""
        cells = set(cells)
        graph = {v: [] for v in cells}

        for r, c in cells:
            for dr, dc in dirs4:
                nb = (r + dr, c + dc)
                if nb in cells:
                    graph[(r, c)].append(nb)

        return graph

    # ----------------------------------------------------------------------
    # Role detection: background, legend, anchors, thick field
    # ----------------------------------------------------------------------

    # Background is the most common border color.
    border_vals = np.concatenate([
        out[0, :].astype(int),
        out[-1, :].astype(int),
        out[:, 0].astype(int),
        out[:, -1].astype(int)
    ])
    vals, counts = np.unique(border_vals, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    # Find all isolated 2x2 blocks.  Legend blocks are small, so they are
    # normally not part of a larger rectangle of the same color.
    def isolated_2x2_blocks():
        blocks = []

        for r in range(H - 1):
            for c in range(W - 1):
                v = int(out[r, c])
                if v == bg:
                    continue

                if not (
                    out[r, c] == v and
                    out[r, c + 1] == v and
                    out[r + 1, c] == v and
                    out[r + 1, c + 1] == v
                ):
                    continue

                # Skip if part of a taller rectangle.
                if r > 0 and out[r - 1, c] == v and out[r - 1, c + 1] == v:
                    continue
                if r + 2 < H and out[r + 2, c] == v and out[r + 2, c + 1] == v:
                    continue

                # Skip if part of a wider rectangle.
                if c > 0 and out[r, c - 1] == v and out[r + 1, c - 1] == v:
                    continue
                if c + 2 < W and out[r, c + 2] == v and out[r + 1, c + 2] == v:
                    continue

                blocks.append((r, c, v))

        return blocks

    key_blocks = isolated_2x2_blocks()

    # If there are none, try a looser isolation test.
    if not key_blocks:
        for r in range(H - 1):
            for c in range(W - 1):
                v = int(out[r, c])
                if v == bg:
                    continue

                if not (
                    out[r, c] == v and
                    out[r, c + 1] == v and
                    out[r + 1, c] == v and
                    out[r + 1, c + 1] == v
                ):
                    continue

                # Isolated in at least two opposite directions.
                up = (r > 0 and out[r - 1, c] == v and out[r - 1, c + 1] == v)
                down = (r + 2 < H and out[r + 2, c] == v and out[r + 2, c + 1] == v)
                left = (c > 0 and out[r, c - 1] == v and out[r + 1, c - 1] == v)
                right = (c + 2 < W and out[r, c + 2] == v and out[r + 1, c + 2] == v)

                if (not up) or (not down) or (not left) or (not right):
                    key_blocks.append((r, c, v))

    key_cells = set()
    for r, c, _ in key_blocks:
        for dr in (0, 1):
            for dc in (0, 1):
                key_cells.add((r + dr, c + dc))

    sequence = []
    marker_color = -1

    if key_blocks:
        top_row = min(r for r, c, v in key_blocks)
        seq_blocks = sorted(
            [b for b in key_blocks if b[0] == top_row],
            key=lambda x: x[1]
        )
        sequence = [int(v) for _, _, v in seq_blocks]

        other_blocks = [b for b in key_blocks if b[0] != top_row]
        if other_blocks:
            marker_block = max(other_blocks, key=lambda x: (x[0], x[1]))
            marker_color = int(marker_block[2])

    # Fallback: use small non-background colors as sequence/marker.
    if not sequence:
        freq = {}
        for v in np.unique(out):
            v = int(v)
            if v == bg:
                continue
            cnt = int(np.count_nonzero(out == v))
            if cnt <= 12:
                freq[v] = cnt
        ordered = sorted(freq, key=lambda x: (freq[x], x))
        if ordered:
            sequence = [ordered[0]]
            if len(ordered) > 1:
                marker_color = ordered[1]

    if not sequence:
        return out

    if marker_color == -1:
        for v in np.unique(out):
            v = int(v)
            if v != bg and v not in sequence:
                marker_color = v
                break

        if marker_color == -1:
            return out

    legend_colors = set(sequence) | {int(marker_color)}

    # The thick field is the largest connected component whose color is not
    # background and not a legend color.
    field_color = -1
    best_area = 0

    for v in np.unique(out):
        v = int(v)
        if v == bg or v in legend_colors:
            continue

        mask = (out == v)
        comps = components(mask)
        if not comps:
            continue

        area = max(len(comp) for comp in comps)
        if area > best_area:
            best_area = area
            field_color = v

    # Fallback: use the color most adjacent to anchors.
    if field_color == -1:
        anchor_color = int(sequence[0])
        candidates = {}
        for r in range(H):
            for c in range(W):
                if int(out[r, c]) == anchor_color:
                    for dr, dc in dirs4:
                        nr, nc = r + dr, c + dc
                        if inb(nr, nc):
                            nv = int(out[nr, nc])
                            if nv != bg and nv not in legend_colors:
                                candidates[nv] = candidates.get(nv, 0) + 1
        if candidates:
            field_color = max(candidates, key=lambda x: candidates[x])

    if field_color == -1:
        return out

    field_mask = (out == field_color)

    # Anchors are single border pixels of the first sequence color.
    anchor_color = int(sequence[0])
    anchors = []

    border_cells = set()
    for r in range(H):
        border_cells.add((r, 0))
        border_cells.add((r, W - 1))
    for c in range(W):
        border_cells.add((0, c))
        border_cells.add((H - 1, c))

    for r, c in border_cells:
        if int(out[r, c]) == anchor_color and (r, c) not in key_cells:
            anchors.append((r, c))

    # Fallback: first two columns / rows if border anchors were missed.
    if not anchors:
        for r in range(H):
            for c in range(min(2, W)):
                if int(out[r, c]) == anchor_color and (r, c) not in key_cells:
                    anchors.append((r, c))
                    break

        for c in range(W):
            for r in range(min(2, H)):
                if int(out[r, c]) == anchor_color and (r, c) not in key_cells:
                    anchors.append((r, c))
                    break

    if not anchors:
        return out

    anchor_set = set(anchors)

    # Remove legend blocks now.
    for r, c, _ in key_blocks:
        out[r:r + 2, c:c + 2] = bg

    # ----------------------------------------------------------------------
    # Skeletonization
    # ----------------------------------------------------------------------

    def zhang_suen_skeleton(mask):
        """4-connected Zhang-Suen thinning, padded so border shapes survive."""
        if not mask.any():
            return np.zeros((H, W), dtype=bool)

        X = np.pad(mask.astype(np.uint8), 2, mode="constant", constant_values=0)
        h, w = X.shape

        while True:
            changed = False

            for step in (1, 2):
                delete = []

                for i in range(1, h - 1):
                    for j in range(1, w - 1):
                        if X[i, j] != 1:
                            continue

                        b2 = X[i, j - 1]
                        b3 = X[i + 1, j - 1]
                        b4 = X[i + 1, j]
                        b5 = X[i + 1, j + 1]
                        b6 = X[i, j + 1]
                        b7 = X[i - 1, j + 1]
                        b8 = X[i - 1, j]
                        b9 = X[i - 1, j - 1]

                        B = int(b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9)

                        A = (
                            int(b2 == 0 and b3 == 1) +
                            int(b3 == 0 and b4 == 1) +
                            int(b4 == 0 and b5 == 1) +
                            int(b5 == 0 and b6 == 1) +
                            int(b6 == 0 and b7 == 1) +
                            int(b7 == 0 and b8 == 1) +
                            int(b8 == 0 and b9 == 1) +
                            int(b9 == 0 and b2 == 1)
                        )

                        if 2 <= B <= 6 and A == 1:
                            if step == 1 and (b2 * b4 * b6) == 0 and (b4 * b6 * b8) == 0:
                                delete.append((i, j))
                            elif step == 2 and (b2 * b4 * b8) == 0 and (b2 * b6 * b8) == 0:
                                delete.append((i, j))

                if delete:
                    for i, j in delete:
                        X[i, j] = 0
                    changed = True

            if not changed:
                break

        return X[2:-2, 2:-2] == 1

    # The field mask should not include removed legend cells.
    for r, c in key_cells:
        field_mask[r, c] = False

    skeleton = zhang_suen_skeleton(field_mask)
    path_cells = set(map(tuple, np.argwhere(skeleton)))

    # Add anchors and connect them to the skeleton in the field.
    def add_line(cells, r1, c1, r2, c2):
        if r1 == r2:
            lo, hi = sorted((c1, c2))
            for c in range(lo, hi + 1):
                if inb(r1, c):
                    cells.add((r1, c))
        elif c1 == c2:
            lo, hi = sorted((r1, r2))
            for r in range(lo, hi + 1):
                if inb(r, c1):
                    cells.add((r, c1))

    if path_cells:
        # Connect each anchor to the nearest path cell, preferring horizontal
        # connections because the examples start along rows.
        path_list = list(path_cells)

        for ar, ac in anchors:
            # Same row first.
            same_row = [p for p in path_list if p[0] == ar and p[1] >= ac]
            if same_row:
                tr, tc = min(same_row, key=lambda p: p[1])
                add_line(path_cells, ar, ac, tr, tc)
                continue

            # Same column next.
            same_col = [p for p in path_list if p[1] == ac and p[0] >= ar]
            if same_col:
                tr, tc = min(same_col, key=lambda p: p[0])
                add_line(path_cells, ar, ac, tr, tc)
                continue

            # Nearest path cell, with an L-shaped connector.
            tr, tc = min(path_list, key=lambda p: (abs(p[0] - ar) + abs(p[1] - ac), p[0], p[1]))

            # Try horizontal-first and vertical-first; keep the one that stays
            # inside the field for the longest possible stretch.
            def connector_score(cells):
                score = 0
                for r, c in cells:
                    if field_mask[r, c]:
                        score += 1
                    if (r, c) in key_cells:
                        score -= 100
                return score

            h_first = set()
            add_line(h_first, ar, ac, ar, tc)
            add_line(h_first, ar, tc, tr, tc)

            v_first = set()
            add_line(v_first, ar, ac, tr, ac)
            add_line(v_first, tr, ac, tr, tc)

            if connector_score(h_first) >= connector_score(v_first):
                path_cells.update(h_first)
            else:
                path_cells.update(v_first)

    # Ensure the path is 4-connected and connected to anchors.
    path_bool = np.zeros((H, W), dtype=bool)
    for r, c in path_cells:
        if inb(r, c):
            path_bool[r, c] = True

    good_comps = []
    for comp in components(path_bool):
        if any(p in anchor_set for p in comp):
            good_comps.append(comp)

    if not good_comps:
        return out

    path_cells = set()
    for comp in good_comps:
        path_cells.update(comp)

    # Remove legend cells from the path.
    path_cells -= key_cells

    path_bool = np.zeros((H, W), dtype=bool)
    for r, c in path_cells:
        if inb(r, c):
            path_bool[r, c] = True

    # ----------------------------------------------------------------------
    # Color path by distance from anchors, using the repeating sequence.
    # ----------------------------------------------------------------------

    L = len(sequence)
    dist = np.full((H, W), -1, dtype=int)
    q = deque()

    for r, c in anchors:
        if path_bool[r, c]:
            dist[r, c] = 0
            q.append((r, c))

    while q:
        r, c = q.popleft()
        d = dist[r, c]
        for dr, dc in dirs4:
            nr, nc = r + dr, c + dc
            if inb(nr, nc) and path_bool[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = d + 1
                q.append((nr, nc))

    for r, c in path_cells:
        d = dist[r, c]
        if d < 0:
            d = 0
        out[r, c] = sequence[d % L]

    # ----------------------------------------------------------------------
    # Place marker cells just outside the field at exposed path ends.
    # ----------------------------------------------------------------------

    graph = build_graph(path_cells)

    # Terminals are path cells of degree 1, except anchors.
    terminals = [v for v, ns in graph.items() if len(ns) == 1 and v not in anchor_set]

    # Also consider junctions that have an exposed main-field dead end.  This
    # handles cases where a short branch is absorbed into a junction by the
    # skeletonizer but the thick field still has an endpoint there.
    candidate_end_points = list(terminals)

    for v, ns in graph.items():
        if v in anchor_set:
            continue
        r, c = v

        for dr, dc in dirs4:
            nr, nc = r + dr, c + dc
            if not inb(nr, nc):
                continue

            # If the neighboring cell is not part of the path but is field,
            # the path may still need an endpoint marker one step further out.
            if field_mask[nr, nc] and (nr, nc) not in path_cells:
                candidate_end_points.append((nr, nc))

    # Deduplicate while preserving order.
    seen = set()
    final_candidates = []
    for p in candidate_end_points:
        if p not in seen:
            seen.add(p)
            final_candidates.append(p)

    for r, c in final_candidates:
        if not inb(r, c):
            continue

        # For a terminal path cell, use the direction away from its neighbor.
        if (r, c) in path_cells and graph.get((r, c)) and len(graph[(r, c)]) == 1:
            nr, nc = graph[(r, c)][0]
            dr = r - nr
            dc = c - nc
        else:
            # For a field cell adjacent to the path, move away from the path.
            best = None
            for dr, dc in dirs4:
                pr, pc = r - dr, c - dc
                if inb(pr, pc) and path_bool[pr, pc]:
                    best = (dr, dc)
                    break
            if best is None:
                continue
            dr, dc = best

        rr, cc = r, c
        marker_pos = None

        # Move outward until leaving the field.
        while inb(rr, cc):
            if not field_mask[rr, cc]:
                marker_pos = (rr, cc)
                break
            rr += dr
            cc += dc

        # If the terminal itself is already outside the field, mark it.
        if marker_pos is None and inb(r, c) and not field_mask[r, c]:
            marker_pos = (r, c)

        if marker_pos is not None and marker_pos not in key_cells:
            mr, mc = marker_pos
            if inb(mr, mc) and not path_bool[mr, mc]:
                out[mr, mc] = marker_color

    # Ensure legend blocks remain removed.
    for r, c, _ in key_blocks:
        out[r:r + 2, c:c + 2] = bg

    return out.astype(int, copy=False)