import numpy as np
from collections import deque
from functools import lru_cache

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform ARC-style grids containing:
      - a background color,
      - two marker blocks of one non-background color,
      - two path colors.

    For each marker block, find the longest simple path in the adjacent
    path-color component that starts at the marker. Recolor selected path
    pixels: the path color that forms more 2x2 solid blocks becomes 3,
    the other path color becomes 5.
    """
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape
    out = np.zeros((h, w), dtype=np.uint8)

    # 1. Background = most frequent color.
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[np.argmax(counts)])

    nonbg_counts = {int(c): int(n) for c, n in zip(colors, counts) if int(c) != background}
    if len(nonbg_counts) < 3:
        return np.full_like(grid, background, dtype=np.uint8)

    # 2. Marker color = least frequent non-background color, should have exactly two 8-connected components.
    marker_color = min(nonbg_counts, key=lambda c: (nonbg_counts[c], c))

    def component_id(mask):
        comp = -np.ones(mask.shape, dtype=int)
        cid = 0
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] and comp[i, j] == -1:
                    q = [(i, j)]
                    comp[i, j] = cid
                    while q:
                        y, x = q.pop()
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                if dy == 0 and dx == 0:
                                    continue
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                                    if mask[ny, nx] and comp[ny, nx] == -1:
                                        comp[ny, nx] = cid
                                        q.append((ny, nx))
                    cid += 1
        return comp, cid

    marker_mask = grid == marker_color
    marker_comp, marker_comp_count = component_id(marker_mask)

    # Fallback if marker assumption fails.
    if marker_comp_count != 2:
        # Choose color with exactly two components if any.
        for c, _ in sorted(nonbg_counts.items(), key=lambda kv: (kv[1], kv[0])):
            if c == background:
                continue
            m = grid == c
            _, cnt = component_id(m)
            if cnt == 2:
                marker_color = c
                marker_mask = m
                marker_comp, marker_comp_count = component_id(m)
                break

    if marker_comp_count != 2:
        return np.full_like(grid, background, dtype=np.uint8)

    # 3. Path colors: the two most frequent non-marker non-background colors.
    candidates = [(c, n) for c, n in nonbg_counts.items() if c != marker_color]
    candidates.sort(key=lambda x: (-x[1], x[0]))
    if len(candidates) < 2:
        return np.full_like(grid, background, dtype=np.uint8)

    path_colors = [int(candidates[0][0]), int(candidates[1][0])]

    # 4. Decide which path color is the "block" color -> 3.
    def two_by_two_count(color):
        cnt = 0
        g = grid
        for i in range(h - 1):
            for j in range(w - 1):
                if g[i, j] == color and g[i + 1, j] == color and g[i, j + 1] == color and g[i + 1, j + 1] == color:
                    cnt += 1
        return cnt

    block_color = path_colors[0] if two_by_two_count(path_colors[0]) >= two_by_two_count(path_colors[1]) else path_colors[1]
    line_color = path_colors[1] if block_color == path_colors[0] else path_colors[0]

    path_mask = np.isin(grid, path_colors)

    # For each path cell, precompute path-color-only 8-connected component ids.
    path_comp, _ = component_id(path_mask)

    # Starting cells adjacent to a marker component.
    starts = []
    for cid in range(marker_comp_count):
        ys, xs = np.where(marker_comp == cid)
        for y, x in zip(ys, xs):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and path_mask[ny, nx]:
                        starts.append((cid, (ny, nx)))

    # Deduplicate by (marker component, path cell, path component)
    seen_starts = set()
    selected = np.zeros((h, w), dtype=bool)

    @lru_cache(maxsize=None)
    def longest_simple_path(start, comp_id, limit=900):
        """
        DFS longest simple path in a small graph, with a soft cell limit.
        Returns (length, path tuple of coordinates).
        """
        best = [0, (start,)]

        # Neighbors of a cell restricted to same path component.
        def neighbors(p):
            y, x = p
            res = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and path_comp[ny, nx] == comp_id:
                        res.append((ny, nx))
            return res

        # If the component is small enough, exact DFS is fine.
        comp_cells = list(zip(*np.where(path_comp == comp_id)))
        if len(comp_cells) <= 60:
            visited = {start}
            stack = [(start, [start])]
            while stack:
                node, path = stack.pop()
                if len(path) > best[0]:
                    best[0] = len(path)
                    best[1] = tuple(path)
                if len(path) > limit:
                    continue
                for nb in neighbors(node):
                    if nb not in visited:
                        visited.add(nb)
                        stack.append((nb, path + [nb]))
                        visited.remove(nb)
        else:
            # Fallback: farthest simple greedy path using BFS depth in a tree-like approximation.
            # Use BFS from start to get parent pointers, then walk to farthest cell.
            parent = {start: None}
            q = deque([start])
            far = start
            while q:
                node = q.popleft()
                far = node
                for nb in neighbors(node):
                    if nb not in parent:
                        parent[nb] = node
                        q.append(nb)
            path = []
            cur = far
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            if len(path) > best[0]:
                best[0] = len(path)
                best[1] = tuple(path)

        return best[0], best[1]

    for marker_cid, start_cell in starts:
        key = (marker_cid, start_cell)
        if key in seen_starts:
            continue
        seen_starts.add(key)
        comp_id = int(path_comp[start_cell])
        _, path = longest_simple_path(start_cell, comp_id)
        for y, x in path:
            selected[y, x] = True

    # 5. Build output: background + unselected original colors + recolored selected path colors.
    out = grid.astype(np.uint8).copy()
    sel_block = selected & (grid == block_color)
    sel_line = selected & (grid == line_color)
    out[sel_block] = 3
    out[sel_line] = 5

    return out