import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Repair small corrupted periodic panels.

    The grid contains one or more rectangular panels enclosed by a frame color.
    Inside each panel is a small repeating 2D pattern using mainly a background
    color and one or more foreground colors.  A few cells are corrupted.  This
    transform detects each enclosed panel, infers its small periodic tile from
    the majority vote at each tile position, and reconstructs the panel.

    The tile search balances:
      - good fit to the observed panel cells,
      - simplicity/small tile area,
      - preservation of color diversity inside the panel.
    """
    if grid.size == 0:
        return grid.copy()

    g = np.array(grid, dtype=int).copy()
    h, w = g.shape
    if h == 0 or w == 0:
        return g

    # ------------------------------------------------------------------
    # 1. Identify the outer background and the panel frame color.
    # ------------------------------------------------------------------
    def most_common_border(arr: np.ndarray) -> int:
        hh, ww = arr.shape
        border = np.concatenate((
            arr[0, :],
            arr[-1, :],
            arr[:, 0],
            arr[:, -1]
        ))
        vals, counts = np.unique(border, return_counts=True)
        return int(vals[np.argmax(counts)])

    def find_frame_color(arr: np.ndarray, outer: int):
        hh, ww = arr.shape

        # In the visible examples/challenge the frame starts immediately inside
        # the outer border.
        if hh > 1 and ww > 1:
            c = int(arr[1, 1])
            if c != outer:
                return c

        # Fallback: colors adjacent to the outer background.
        adj_counts = {}
        for r in range(hh):
            for c in range(ww):
                if arr[r, c] == outer:
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < hh and 0 <= nc < ww:
                            v = int(arr[nr, nc])
                            if v != outer:
                                adj_counts[v] = adj_counts.get(v, 0) + 1
        if adj_counts:
            return max(adj_counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]

        # Fallback: non-outer color in the top-left interior block.
        cand = {}
        for i in range(min(hh, 4)):
            for j in range(min(ww, 4)):
                v = int(arr[i, j])
                if v != outer:
                    cand[v] = cand.get(v, 0) + 1
        if cand:
            return max(cand.items(), key=lambda kv: (kv[1], -kv[0]))[0]

        # Last fallback: second most common color in the whole grid.
        vals, counts = np.unique(arr, return_counts=True)
        order = np.argsort(-counts)
        for v in order:
            if int(v) != outer:
                return int(v)
        return None

    outer_color = most_common_border(g)
    frame_color = find_frame_color(g, outer_color)

    if frame_color is None or frame_color == outer_color:
        return g

    # ------------------------------------------------------------------
    # 2. Find enclosed non-frame components.  Each component is a panel
    #    interior.  The component touching the grid border is exterior.
    # ------------------------------------------------------------------
    mask = g != frame_color
    visited = np.zeros((h, w), dtype=bool)

    def bfs(start_r: int, start_c: int):
        q = deque([(start_r, start_c)])
        visited[start_r, start_c] = True
        cells = []

        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and mask[nr, nc]:
                    visited[nr, nc] = True
                    q.append((nr, nc))
        return cells

    panels = []
    for r in range(h):
        for c in range(w):
            if not visited[r, c] and mask[r, c]:
                cells = bfs(r, c)
                # Ignore exterior background component.
                if any(rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1
                       for rr, cc in cells):
                    continue
                # Ignore tiny noise components.
                if len(cells) >= 2:
                    panels.append(cells)

    out = g.copy()

    # ------------------------------------------------------------------
    # 3. Infer a small periodic tile for each panel.
    # ------------------------------------------------------------------
    def infer_tile(values: np.ndarray,
                   rows: np.ndarray,
                   cols: np.ndarray,
                   ph: int,
                   pw: int,
                   preferred_bg: int) -> np.ndarray:
        """
        Infer a periodic tile for one panel.

        parameters
        ----------
        values : panel cell values, length N
        rows   : row offsets within the panel bounding box, length N
        cols   : col offsets within the panel bounding box, length N
        ph/pw  : bounding box height/width
        preferred_bg : color preferred in ties, normally the outer background
        """
        n = len(values)
        if n == 0:
            return np.array([[preferred_bg]], dtype=int)

        max_period = 12

        th_candidates = sorted(set(list(range(1, min(ph, max_period) + 1)) + [ph]))
        tw_candidates = sorted(set(list(range(1, min(pw, max_period) + 1)) + [pw]))

        max_val = int(values.max()) if n else 0
        k = max(10, max_val + 1)

        global_counts = np.bincount(values, minlength=k).astype(int)
        panel_unique = len(np.unique(values))

        best_score = float("inf")
        best_tile = None
        best_area = None
        best_mism = None

        for th in th_candidates:
            for tw in tw_candidates:
                # Never consider a zero-size tile.
                if th <= 0 or tw <= 0:
                    continue

                # Count colors for each tile position.
                counts = np.zeros((th, tw, k), dtype=int)
                for v, rr, cc in zip(values, rows, cols):
                    counts[rr % th, cc % tw, int(v)] += 1

                tile = np.zeros((th, tw), dtype=int)
                for i in range(th):
                    for j in range(tw):
                        cnt = counts[i, j]
                        max_c = int(cnt.max())
                        if max_c == 0:
                            tile[i, j] = preferred_bg
                            continue

                        # Choose the most common color.  Ties prefer the outer
                        # background, then globally more common color, then lower id.
                        best = None
                        best_key = None
                        for color in range(k):
                            if cnt[color] != max_c:
                                continue
                            key = (
                                1 if color == preferred_bg else 0,
                                int(global_counts[color]),
                                -int(color)
                            )
                            if best_key is None or key > best_key:
                                best = color
                                best_key = key
                        tile[i, j] = int(best if best is not None else preferred_bg)

                # Mismatches against observed panel cells.
                tile_values = tile[rows % th, cols % tw]
                mism = int(np.sum(tile_values != values))

                area = th * tw
                score = (float(mism) / float(max(1, n))) * 10.0 + float(np.log(area))

                # Discourage tiles that erase colors present in the panel.
                tile_unique = len(np.unique(tile))
                if panel_unique > tile_unique:
                    score += 2.0 * (panel_unique - tile_unique)

                # Prefer smaller area / fewer mismatches in near ties.
                if (score < best_score - 1e-9 or
                    (abs(score - best_score) <= 1e-9 and
                     (area < best_area or (area == best_area and mism < best_mism)))):
                    best_score = score
                    best_tile = tile
                    best_area = area
                    best_mism = mism

        if best_tile is None:
            best_tile = np.array([[preferred_bg]], dtype=int)
        return best_tile

    # ------------------------------------------------------------------
    # 4. Reconstruct each panel from its inferred tile.
    # ------------------------------------------------------------------
    for cells in panels:
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1 = min(rs), max(rs) + 1
        c0, c1 = min(cs), max(cs) + 1

        ph = r1 - r0
        pw = c1 - c0

        rows = np.array([r - r0 for r, _ in cells], dtype=int)
        cols = np.array([c - c0 for _, c in cells], dtype=int)
        values = np.array([g[r, c] for r, c in cells], dtype=int)

        tile = infer_tile(values, rows, cols, ph, pw, outer_color)
        th, tw = tile.shape

        for (r, c), rr, cc in zip(cells, rows, cols):
            out[r, c] = tile[rr % th, cc % tw]

    return out