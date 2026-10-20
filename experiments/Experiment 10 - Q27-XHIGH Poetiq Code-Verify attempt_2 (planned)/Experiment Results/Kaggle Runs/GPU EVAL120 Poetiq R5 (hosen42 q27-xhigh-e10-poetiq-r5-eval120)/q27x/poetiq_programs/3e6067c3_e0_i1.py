import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Detect numbered boxes and a bottom order line.
    For each consecutive pair of boxes in the order line, fill the straight
    corridor between them with the value of the earlier box.
    """
    grid = np.array(grid, dtype=int, copy=True)

    if grid.ndim != 2 or grid.size == 0:
        return grid

    H, W = grid.shape

    # ------------------------------------------------------------------
    # 1. Identify background color.
    #    Use the border color that is most frequent in the whole grid.
    # ------------------------------------------------------------------
    border = np.concatenate([
        grid[0, :].ravel(),
        grid[-1, :].ravel(),
        grid[:, 0].ravel(),
        grid[:, -1].ravel()
    ])
    border_vals, _ = np.unique(border, return_counts=True)
    bg = int(max(border_vals, key=lambda c: np.count_nonzero(grid == c)))

    nonbg = grid != bg

    # ------------------------------------------------------------------
    # 2. Connected components of non-background cells.
    #    Boxes are solid rectangular components with a uniform border
    #    and an interior digit color.
    # ------------------------------------------------------------------
    visited = np.zeros((H, W), dtype=bool)
    box_mask = np.zeros((H, W), dtype=bool)

    class Box:
        __slots__ = (
            "value",
            "r1", "r2", "c1", "c2",
            "pr1", "pr2", "pc1", "pc2"
        )

        def __init__(self, value, r1, r2, c1, c2, pr1, pr2, pc1, pc2):
            self.value = int(value)
            self.r1 = int(r1)
            self.r2 = int(r2)
            self.c1 = int(c1)
            self.c2 = int(c2)

            # Port / digit span: the span of the colored digit inside.
            self.pr1 = int(pr1)
            self.pr2 = int(pr2)
            self.pc1 = int(pc1)
            self.pc2 = int(pc2)

    boxes = []
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for r in range(H):
        for c in range(W):
            if not nonbg[r, c] or visited[r, c]:
                continue

            # BFS / DFS component.
            stack = [(r, c)]
            visited[r, c] = True
            count = 0
            minr = maxr = r
            minc = maxc = c

            while stack:
                cr, cc = stack.pop()
                count += 1

                if cr < minr:
                    minr = cr
                elif cr > maxr:
                    maxr = cr

                if cc < minc:
                    minc = cc
                elif cc > maxc:
                    maxc = cc

                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < H and 0 <= nc < W
                        and nonbg[nr, nc]
                        and not visited[nr, nc]
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            height = maxr - minr + 1
            width = maxc - minc + 1

            # Ignore isolated cells and non-rectangular / tiny components.
            if count < 2 or height < 2 or width < 2:
                continue

            # Solid rectangle?
            if count == height * width:
                # Mark every solid block as an obstacle.
                box_mask[minr:maxr + 1, minc:maxc + 1] = True

                sub = grid[minr:maxr + 1, minc:maxc + 1]

                # A valid box should have a uniform border color.
                sub_border = np.concatenate([
                    sub[0, :].ravel(),
                    sub[-1, :].ravel(),
                    sub[:, 0].ravel(),
                    sub[:, -1].ravel()
                ])

                border_colors, _ = np.unique(sub_border, return_counts=True)
                if len(border_colors) != 1:
                    continue

                container = int(border_colors[0])
                non_container = sub != container

                if not non_container.any():
                    continue

                # The box value is the most frequent non-container color.
                digit_vals, digit_counts = np.unique(
                    sub[non_container], return_counts=True
                )
                digit_val = int(digit_vals[np.argmax(digit_counts)])

                if digit_val == container:
                    continue

                # Digit span for corridor alignment.
                digit_cells = (sub == digit_val)
                ys, xs = np.where(digit_cells)
                if ys.size == 0:
                    continue

                pr1 = minr + int(ys.min())
                pr2 = minr + int(ys.max())
                pc1 = minc + int(xs.min())
                pc2 = minc + int(xs.max())

                boxes.append(Box(digit_val, minr, maxr, minc, maxc,
                                 pr1, pr2, pc1, pc2))

    # ------------------------------------------------------------------
    # 3. Extract the bottom order sequence.
    #    It is the row with the most remaining non-background cells.
    # ------------------------------------------------------------------
    remaining = nonbg & ~box_mask

    if not remaining.any():
        return grid

    row_counts = remaining.sum(axis=1)
    max_count = int(row_counts.max())

    if max_count == 0:
        return grid

    seq_row = int(np.where(row_counts == max_count)[0][-1])
    seq_cols = np.where(remaining[seq_row])[0]
    sequence = [int(grid[seq_row, c]) for c in seq_cols]

    if len(sequence) < 2:
        return grid

    box_values = set(b.value for b in boxes)
    if not all(v in box_values for v in sequence):
        return grid

    N = len(boxes)
    if N == 0:
        return grid

    # ------------------------------------------------------------------
    # 4. Corridor connection helper.
    #    A horizontal/vertical corridor is valid if the digit spans overlap
    #    and the cells between the two boxes are free of boxes.
    # ------------------------------------------------------------------
    def clear_rect(r1, r2, c1, c2):
        if r1 > r2 or c1 > c2:
            return False
        return not box_mask[r1:r2 + 1, c1:c2 + 1].any()

    def connection(a, b, strict=True):
        # Horizontal connection.
        if a.pr1 <= b.pr2 and b.pr1 <= a.pr2:
            if a.c2 < b.c1:
                left, right = a, b
            elif b.c2 < a.c1:
                left, right = b, a
            else:
                left = None

            if left is not None:
                r1 = max(a.pr1, b.pr1)
                r2 = min(a.pr2, b.pr2)

                if r1 <= r2:
                    c1 = left.c2 + 1
                    c2 = right.c1 - 1

                    if c1 <= c2 and (not strict or clear_rect(r1, r2, c1, c2)):
                        return (r1, r2, c1, c2)

        # Vertical connection.
        if a.pc1 <= b.pc2 and b.pc1 <= a.pc2:
            if a.r2 < b.r1:
                top, bottom = a, b
            elif b.r2 < a.r1:
                top, bottom = b, a
            else:
                top = None

            if top is not None:
                c1 = max(a.pc1, b.pc1)
                c2 = min(a.pc2, b.pc2)

                if c1 <= c2:
                    r1 = top.r2 + 1
                    r2 = bottom.r1 - 1

                    if r1 <= r2 and (not strict or clear_rect(r1, r2, c1, c2)):
                        return (r1, r2, c1, c2)

        return None

    def build_adjacency(strict):
        adj = [set() for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if connection(boxes[i], boxes[j], strict=strict) is not None:
                    adj[i].add(j)
                    adj[j].add(i)
        return adj

    # ------------------------------------------------------------------
    # 5. Find a path through boxes matching the sequence.
    # ------------------------------------------------------------------
    cand_by_val = {}
    for i, b in enumerate(boxes):
        cand_by_val.setdefault(b.value, []).append(i)

    for v in cand_by_val:
        cand_by_val[v].sort(key=lambda idx: (boxes[idx].pr1, boxes[idx].pc1))

    M = len(sequence)

    def find_path_for(adj):
        adj_by_val = [{} for _ in range(N)]
        for i in range(N):
            for j in adj[i]:
                adj_by_val[i].setdefault(boxes[j].value, []).append(j)

        for i in range(N):
            for v in adj_by_val[i]:
                adj_by_val[i][v].sort(
                    key=lambda idx: (boxes[idx].pr1, boxes[idx].pc1)
                )

        memo = set()

        def dfs(i, last, used_mask):
            if i == M:
                return []

            state = (i, last, used_mask)
            if state in memo:
                return None

            val = sequence[i]

            if i == 0:
                cands = list(cand_by_val.get(val, []))
            else:
                cands = list(adj_by_val[last].get(val, []))

            cands = [c for c in cands if not ((used_mask >> c) & 1)]

            # Forward pruning: the next value must be reachable.
            if i + 1 < M:
                next_val = sequence[i + 1]

                def has_next(c):
                    nm = used_mask | (1 << c)
                    return any(
                        not ((nm >> d) & 1)
                        for d in adj_by_val[c].get(next_val, [])
                    )

                cands = [c for c in cands if has_next(c)]

            def key(c):
                if i + 1 < M:
                    opts = [
                        d for d in adj_by_val[c].get(sequence[i + 1], [])
                        if not ((used_mask >> d) & 1)
                    ]
                    return (len(opts), boxes[c].pr1, boxes[c].pc1)
                return (0, boxes[c].pr1, boxes[c].pc1)

            cands.sort(key=key)

            for c in cands:
                sub = dfs(i + 1, c, used_mask | (1 << c))
                if sub is not None:
                    return [c] + sub

            memo.add(state)
            return None

        first_candidates = list(cand_by_val.get(sequence[0], []))

        if M > 1:
            next_val = sequence[1]

            def first_key(c):
                nm = 1 << c
                opts = [
                    d for d in adj_by_val[c].get(next_val, [])
                    if not ((nm >> d) & 1)
                ]
                return (0 if opts else 1, len(opts), boxes[c].pr1, boxes[c].pc1)

            first_candidates.sort(key=first_key)
            first_candidates = [
                c for c in first_candidates
                if any(
                    not (((1 << c) >> d) & 1)
                    for d in adj_by_val[c].get(next_val, [])
                )
            ]
        else:
            first_candidates.sort(key=lambda c: (boxes[c].pr1, boxes[c].pc1))

        for c in first_candidates:
            res = dfs(1, c, 1 << c)
            if res is not None:
                return [c] + res

        return None

    # Try strict clear-corridor adjacency first.
    path = find_path_for(build_adjacency(strict=True))

    # Fallback: if strict corridor checking is too restrictive, allow
    # alignment-only adjacency. Filling still protects original boxes and
    # sequence digits.
    if path is None:
        path = find_path_for(build_adjacency(strict=False))

    if path is None:
        return grid

    # ------------------------------------------------------------------
    # 6. Draw corridors.
    #    Protect original boxes and sequence digits from being overwritten.
    # ------------------------------------------------------------------
    protected = box_mask | remaining

    for idx in range(len(path) - 1):
        a = boxes[path[idx]]
        b = boxes[path[idx + 1]]

        rect = connection(a, b, strict=True)
        if rect is None:
            rect = connection(a, b, strict=False)

        if rect is None:
            continue

        r1, r2, c1, c2 = rect
        sub = grid[r1:r2 + 1, c1:c2 + 1]
        mask = ~protected[r1:r2 + 1, c1:c2 + 1]
        sub[mask] = a.value

    return grid