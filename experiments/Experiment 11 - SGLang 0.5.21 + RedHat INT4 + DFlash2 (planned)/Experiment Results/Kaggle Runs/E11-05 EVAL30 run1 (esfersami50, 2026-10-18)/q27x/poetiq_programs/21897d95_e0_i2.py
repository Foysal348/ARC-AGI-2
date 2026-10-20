import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Rule:
      * The grid is partitioned into colored 'blocks' (connected non-1 regions).
      * T-shaped arrows made of 1s (center may be a special 'payload' color)
        point in the direction of their lone 'head' arm.
      * Each arrow copies its carried color (payload if present, else the
        source block's dominant color) into the first block reached in that
        direction.  Arrows pointing out of the grid spray their payload onto
        every untouched block that shares the source block's color.
      * Finally the finished grid is transposed (identity for square grids).
    """
    if grid is None or grid.size == 0:
        return grid.copy() if grid is not None else grid

    H, W = grid.shape
    g = np.array(grid, dtype=np.int64)

    # --- 1. segment blocks: 4-connected components of non-1 pixels ---
    lab = np.zeros((H, W), dtype=np.int64)
    dom = {}
    for r in range(H):
        for c in range(W):
            if g[r, c] != 1 and lab[r, c] == 0:
                lid = lab[r, c] = len(dom) + 1
                cnt = {}
                q = [(r, c)]
                while q:
                    x, y = q.pop()
                    v = int(g[x, y])
                    cnt[v] = cnt.get(v, 0) + 1
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W and lab[nx, ny] == 0 and g[nx, ny] != 1:
                            lab[nx, ny] = lid
                            q.append((nx, ny))
                dom[lid] = max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    out = np.zeros((H, W), dtype=g.dtype)
    for r in range(H):
        for c in range(W):
            if lab[r, c]:
                out[r, c] = dom[lab[r, c]]

    # --- 2. find arrows (T-shapes / embedded payloads) ---
    seen = np.zeros((H, W), dtype=bool)
    arrows = []
    for r in range(H):
        for c in range(W):
            if g[r, c] == 1 and not seen[r, c]:
                cells = [(r, c)]
                seen[r, c] = True
                q = deque([(r, c)])
                while q:
                    x, y = q.popleft()
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < H and 0 <= ny < W and g[nx, ny] == 1 and not seen[nx, ny]:
                                seen[nx, ny] = True
                                cells.append((nx, ny))
                                q.append((nx, ny))
                S = set(cells)

                def find_dir(center):
                    n = [d for d in ((1, 0), (-1, 0), (0, 1), (0, -1))
                         if (center[0] + d[0], center[1] + d[1]) in S]
                    if len(n) != 3:
                        return None
                    for skip in range(3):
                        a, b, h = n[:skip] + n[skip + 1:]
                        if (a[0] == b[0] == center[0]) or (a[1] == b[1] == center[1]):
                            mid = ((center[0] + a[0] + b[0]) / 3.0,
                                   (center[1] + a[1] + b[1]) / 3.0)
                            if mid == (center[0], center[1]):
                                return h
                    return None

                if len(cells) == 4:
                    deg = {x: sum(1 for o in cells if abs(o[0] - x[0]) + abs(o[1] - x[1]) == 1)
                           for x in cells}
                    centers = [x for x in cells if deg[x] == 3]
                else:
                    centers = [max(cells, key=lambda x: deg.get(x, 0))] if deg else list(cells)
                d = next((d for cnd in centers if (d := find_dir(cnd)) is not None), None)
                if d is not None:
                    arrows.append((cells, d))

    # --- 3. apply colour transfers ---
    untouched = set(dom)
    for cells, d in arrows:
        center = None
        if len(cells) == 4:
            deg = {x: sum(1 for o in cells if abs(o[0] - x[0]) + abs(o[1] - x[1]) == 1) for x in cells}
            center = next((x for x in cells if deg[x] == 3), None)
        if center is None:
            continue
        if g[center[0], center[1]] == 1:
            center = next((x for x in cells
                           if all(g[center[0] + d[0], center[1] + d[1]] == 1 for _ in [0])
                               and False), None)
            # fall back: payload is a 1 => use source colour
            payload = None
        else:
            payload = int(g[center[0], center[1]])
        if payload is None or payload == 1:
            # carried colour = dominant colour of the source block
            src = next((lab[x, y] for x, y in cells if lab[x, y]), None)
            if src is None:
                continue
            carried = dom[src]
        else:
            src = None
            for x, y in cells:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < H and 0 <= ny < W and lab[nx, ny] and int(g[nx, ny]) == payload:
                        src = lab[nx, ny]
            carried = payload
        src = src or next((lab[x, y] for x, y in cells if lab[x, y]), None)
        if src is None:
            continue
        # walk in the arrow direction to the first different block
        x, y = center
        target = None
        while 0 <= x < H and 0 <= y < W:
            x += d[0]; y += d[1]
            if not (0 <= x < H and 0 <= y < W):
                break
            l = lab[x, y]
            if l == 0:
                continue
            if l == src:
                continue
            target = l
            break
        if target is not None and target in untouched:
            out[lab == target] = carried
            untouched.discard(target)
        elif target is None and payload is not None:
            # arrow points out of the grid: spray onto untouched same-colour blocks
            for b in sorted(untouched):
                if dom[b] == dom[src]:
                    out[lab == b] = carried
                    untouched.discard(b)

    # --- 4. transpose (identity for square grids) ---
    return np.ascontiguousarray(out.T).astype(grid.dtype)