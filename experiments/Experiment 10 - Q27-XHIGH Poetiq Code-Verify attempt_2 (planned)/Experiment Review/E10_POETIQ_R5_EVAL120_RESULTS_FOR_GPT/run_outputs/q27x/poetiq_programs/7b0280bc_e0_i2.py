import numpy as np
import heapq

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by finding the shortest object path between the two
    marker squares.

    Rule inferred from the examples:
    - The background is the most frequent color.
    - There is a marker color appearing as exactly two solid square blocks.
    - The other non-background colors form "path objects".
    - An object is a monochromatic 8-connected component.
    - Objects are adjacent if they touch orthogonally.
    - The required output is obtained by taking the shortest object path
      between the two marker objects and recoloring every object on that path.
    - Among the two path colors, the one forming square blocks of the same
      size as the marker blocks is recolored to 3; the other path color is
      recolored to 5.
    """
    if grid.size == 0 or grid.ndim != 2:
        return grid.copy()

    h, w = grid.shape
    out = grid.copy()

    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) < 3:
        return out

    background = int(colors[np.argmax(counts)])
    nonbg = [int(c) for c in colors if int(c) != background]
    if len(nonbg) < 2:
        return out

    # ------------------------------------------------------------------
    # Build monochromatic 8-connected objects for all non-background colors.
    # ------------------------------------------------------------------
    dirs8 = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),           (0, 1),
             (1, -1),  (1, 0),  (1, 1)]

    def component_ids(mask):
        comp = -np.ones(mask.shape, dtype=int)
        cid = 0
        for i in range(h):
            for j in range(w):
                if mask[i, j] and comp[i, j] == -1:
                    stack = [(i, j)]
                    comp[i, j] = cid
                    while stack:
                        y, x = stack.pop()
                        for dy, dx in dirs8:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if mask[ny, nx] and comp[ny, nx] == -1:
                                    comp[ny, nx] = cid
                                    stack.append((ny, nx))
                    cid += 1
        return comp, cid

    objects = []
    obj_id_grid = -np.ones((h, w), dtype=int)
    objs_by_color = {}

    for c in nonbg:
        mask = (grid == c)
        comp, cnt = component_ids(mask)
        objs_by_color.setdefault(int(c), [])

        for cid in range(cnt):
            m = (comp == cid)
            ys, xs = np.where(m)
            if ys.size == 0:
                continue

            min_y, max_y = int(ys.min()), int(ys.max())
            min_x, max_x = int(xs.min()), int(xs.max())
            hh = max_y - min_y + 1
            ww = max_x - min_x + 1
            area = int(m.sum())

            square_size = 0
            if hh == ww and hh >= 2 and area == hh * ww:
                square_size = hh

            oid = len(objects)
            obj_id_grid[ys, xs] = oid

            objects.append({
                "id": oid,
                "color": int(c),
                "mask": m,
                "area": area,
                "bbox": (min_y, min_x, max_y, max_x),
                "square_size": square_size,
            })
            objs_by_color[int(c)].append(objects[oid])

    if not objects:
        return out

    # ------------------------------------------------------------------
    # Build adjacency graphs:
    #   mode 4: orthogonal adjacency
    #   mode 8: 8-neighbor adjacency, used only as fallback
    # ------------------------------------------------------------------
    def build_adjacency(mode):
        adj = [set() for _ in range(len(objects))]
        if mode == 4:
            half_dirs = [(0, 1), (1, 0)]
        else:
            half_dirs = [(0, 1), (1, 0), (1, 1), (-1, 1)]

        for i in range(h):
            for j in range(w):
                a = obj_id_grid[i, j]
                if a < 0:
                    continue
                for dy, dx in half_dirs:
                    ni, nj = i + dy, j + dx
                    if 0 <= ni < h and 0 <= nj < w:
                        b = obj_id_grid[ni, nj]
                        if b >= 0 and b != a:
                            adj[a].add(b)
                            adj[b].add(a)
        return adj

    # ------------------------------------------------------------------
    # Dijkstra over objects.
    # Cost is primarily the number of path objects on the path.
    # Ties are broken by total area of those path objects.
    # ------------------------------------------------------------------
    def find_object_path(adj, start, end, marker_color):
        n = len(objects)
        inf = (10**12, 10**12)
        best = [inf] * n
        parent = [-1] * n

        path_obj_ids = set(
            obj["id"] for obj in objects if obj["color"] != marker_color
        )

        best[start] = (0, 0)
        heap = [(0, 0, start)]

        while heap:
            num, area, u = heapq.heappop(heap)
            if (num, area) != best[u]:
                continue

            if u == end:
                path = []
                cur = end
                while cur != start:
                    if cur == -1:
                        return None
                    if cur in path_obj_ids:
                        path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path

            for v in adj[u]:
                if v == start:
                    continue
                if v != end and objects[v]["color"] == marker_color:
                    continue

                add_num = 1 if v in path_obj_ids else 0
                add_area = objects[v]["area"] if v in path_obj_ids else 0
                new_cost = (num + add_num, area + add_area)

                if new_cost < best[v]:
                    best[v] = new_cost
                    parent[v] = u
                    heapq.heappush(heap, (new_cost[0], new_cost[1], v))

        return None

    # ------------------------------------------------------------------
    # Identify marker color:
    # exactly two square objects of the same size, and a valid path exists
    # between them through non-marker objects.
    # ------------------------------------------------------------------
    candidates = []
    for color, objs in objs_by_color.items():
        if len(objs) == 2:
            s1 = objs[0]["square_size"]
            s2 = objs[1]["square_size"]
            if s1 > 0 and s1 == s2:
                candidates.append((sum(o["area"] for o in objs), color, s1))

    candidates.sort(key=lambda x: (x[0], x[1]))

    result = None

    # Prefer orthogonal object adjacency; fall back to 8-neighbor if needed.
    for mode in (4, 8):
        adj = build_adjacency(mode)

        for _, color, size in candidates:
            objs = objs_by_color[color]
            if len(objs) != 2:
                continue
            path = find_object_path(adj, objs[0]["id"], objs[1]["id"], color)
            if path is not None:
                result = (color, size, objs, path)
                break

        if result is not None:
            break

    # Fallback if the strict marker condition was not met.
    if result is None:
        adj = build_adjacency(8)
        for color in sorted(nonbg, key=lambda c: sum(o["area"] for o in objs_by_color[c])):
            objs = objs_by_color[color]
            if len(objs) >= 2:
                # Choose the two most square-like objects as markers.
                objs_sorted = sorted(objs, key=lambda o: (-o["square_size"], -o["area"]))
                marker_objs = objs_sorted[:2]
                marker_size = max(o["square_size"] for o in marker_objs) or 2
                path = find_object_path(adj, marker_objs[0]["id"], marker_objs[1]["id"], color)
                if path is not None:
                    result = (color, marker_size, marker_objs, path)
                    break

    if result is None:
        return out

    marker_color, marker_size, marker_objs, path_objects = result

    if not path_objects:
        return out

    path_set = set(path_objects)

    # ------------------------------------------------------------------
    # Decide which path color is the "square" color.
    # It is the color that appears as solid marker-sized square objects
    # on the selected path.
    # ------------------------------------------------------------------
    path_colors = []
    seen = set()
    for oid in path_objects:
        c = objects[oid]["color"]
        if c not in seen:
            seen.add(c)
            path_colors.append(c)

    def count_squares_on_path(color, size=None):
        cnt = 0
        for oid in path_set:
            obj = objects[oid]
            if obj["color"] != color:
                continue
            if size is None or obj["square_size"] == size:
                cnt += 1
        return cnt

    square_color = None

    good = [c for c in path_colors if count_squares_on_path(c, marker_size) > 0]
    if good:
        square_color = max(
            good,
            key=lambda c: (
                count_squares_on_path(c, marker_size),
                -sum(objects[o]["area"] for o in path_set if objects[o]["color"] == c)
            )
        )
    else:
        good = [c for c in path_colors if count_squares_on_path(c) > 0]
        if good:
            square_color = max(
                good,
                key=lambda c: (
                    count_squares_on_path(c),
                    -sum(objects[o]["area"] for o in path_set if objects[o]["color"] == c)
                )
            )
        else:
            # Last-resort tie-break: prefer the color with the largest square object.
            square_color = max(
                path_colors,
                key=lambda c: max(
                    [objects[o]["square_size"] for o in path_set if objects[o]["color"] == c] + [0]
                )
            )

    # ------------------------------------------------------------------
    # Recolor selected path objects.
    # ------------------------------------------------------------------
    for oid in path_objects:
        obj = objects[oid]
        if obj["color"] == square_color:
            out[obj["mask"]] = 3
        else:
            out[obj["mask"]] = 5

    return out