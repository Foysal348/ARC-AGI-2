import numpy as np
from collections import deque
import heapq

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by removing wall cells and filling them with the
    dominant color of the nearest enclosed room.

    Rule:
      1. Detect the wall color, typically the color forming a central
         solid 3x3 block and long thin components.
      2. Detect the background color as the most frequent non-wall color.
      3. Label non-wall connected components. Ignore the main background
         component. Each remaining component containing non-background
         colors is a room.
      4. A room's fill color is the most frequent non-background color
         inside it.
      5. Replace every wall cell by the fill color of the closest room
         (4-neighbor Manhattan distance). Ties prefer the globally
         more frequent fill color, then larger room area.
      6. The detected central solid 3x3 wall block remains as a 3x3 ring;
         its center is filled with the globally dominant non-background
         color.
    """
    g = np.array(grid, dtype=int)
    if g.ndim != 2:
        return g.copy()

    H, W = g.shape
    if H == 0 or W == 0:
        return g.copy()

    out = g.copy()

    colors, cnts = np.unique(g, return_counts=True)
    counts = {int(c): int(n) for c, n in zip(colors, cnts)}
    if len(counts) <= 1:
        return out

    most_freq = max(counts, key=lambda c: (counts[c], c))
    center_r = (H - 1) / 2.0
    center_c = (W - 1) / 2.0
    dirs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # ------------------------------------------------------------------
    # 1. Label monochrome components to help detect the wall color.
    # ------------------------------------------------------------------
    comp_label = np.full((H, W), -1, dtype=int)
    comp_stats = []

    for r in range(H):
        for c in range(W):
            if comp_label[r, c] != -1:
                continue

            color = int(g[r, c])
            cid = len(comp_stats)
            q = deque([(r, c)])
            comp_label[r, c] = cid

            area = 0
            min_r = max_r = r
            min_c = max_c = c

            while q:
                cr, cc = q.popleft()
                area += 1
                min_r = min(min_r, cr)
                max_r = max(max_r, cr)
                min_c = min(min_c, cc)
                max_c = max(max_c, cc)

                for dr, dc in dirs4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if comp_label[nr, nc] == -1 and g[nr, nc] == color:
                            comp_label[nr, nc] = cid
                            q.append((nr, nc))

            bbox_area = (max_r - min_r + 1) * (max_c - min_c + 1)
            density = area / bbox_area if bbox_area else 1.0
            comp_stats.append({
                "color": color,
                "area": area,
                "bbox": bbox_area,
                "density": density,
            })

    def is_3x3_block(r, c, v):
        return bool(np.all(g[r:r + 3, c:c + 3] == v))

    def inside_4x4_block(r, c, v):
        if H < 4 or W < 4:
            return False
        r0 = max(0, r - 1)
        r1 = min(H - 4, r + 1)
        c0 = max(0, c - 1)
        c1 = min(W - 4, c + 1)
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if bool(np.all(g[rr:rr + 4, cc:cc + 4] == v)):
                    return True
        return False

    # ------------------------------------------------------------------
    # 2. Detect wall color from maximal 3x3 blocks.
    # ------------------------------------------------------------------
    block_candidates = []

    if H >= 3 and W >= 3:
        for r in range(H - 2):
            for c in range(W - 2):
                v = int(g[r, c])
                if not is_3x3_block(r, c, v):
                    continue
                if inside_4x4_block(r, c, v):
                    continue

                cid = comp_label[r, c]
                st = comp_stats[cid]

                dist_to_center = abs((r + 1) - center_r) + abs((c + 1) - center_c)
                large_component = 0 if st["bbox"] >= 16 else 1

                key = (
                    large_component,
                    dist_to_center,
                    st["density"],
                    -st["bbox"],
                    0 if v != most_freq else 1,
                    -counts[v],
                )
                block_candidates.append((key, v, r, c))

    if block_candidates:
        wall_color = min(block_candidates, key=lambda x: x[0])[1]
    else:
        # Fallback: choose a thin, large component, avoiding the most
        # frequent color unless nothing else exists.
        best_wall_key = None
        wall_color = most_freq
        for c, n in counts.items():
            best_component_key = None
            for st in comp_stats:
                if st["color"] != c:
                    continue
                large_component = 0 if st["bbox"] >= 16 else 1
                ck = (large_component, st["density"], -st["bbox"])
                if best_component_key is None or ck < best_component_key:
                    best_component_key = ck

            if best_component_key is None:
                continue

            key = (
                0 if c != most_freq else 1,
                best_component_key[0],
                best_component_key[1],
                best_component_key[2],
                -n,
            )
            if best_wall_key is None or key < best_wall_key:
                best_wall_key = key
                wall_color = c

    # ------------------------------------------------------------------
    # 3. Choose the special central 3x3 wall block, if any.
    # ------------------------------------------------------------------
    special_blocks = [
        (r, c)
        for _, v, r, c in block_candidates
        if v == wall_color
    ]

    special = None
    if special_blocks:
        special = min(
            special_blocks,
            key=lambda rc: abs((rc[0] + 1) - center_r) + abs((rc[1] + 1) - center_c)
        )

    # ------------------------------------------------------------------
    # 4. Background color: most frequent color other than wall.
    # ------------------------------------------------------------------
    non_wall_colors = [c for c in counts if c != wall_color]
    if non_wall_colors:
        bg_color = max(non_wall_colors, key=lambda c: (counts[c], c))
    else:
        bg_color = wall_color

    # ------------------------------------------------------------------
    # 5. Label non-wall components (rooms plus background).
    # ------------------------------------------------------------------
    nw_label = np.full((H, W), -1, dtype=int)
    rooms = []

    for r in range(H):
        for c in range(W):
            if g[r, c] == wall_color or nw_label[r, c] != -1:
                continue

            rid = len(rooms)
            q = deque([(r, c)])
            nw_label[r, c] = rid

            color_counts = {}
            area = 0

            while q:
                cr, cc = q.popleft()
                area += 1
                col = int(g[cr, cc])
                color_counts[col] = color_counts.get(col, 0) + 1

                for dr, dc in dirs4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if nw_label[nr, nc] == -1 and g[nr, nc] != wall_color:
                            nw_label[nr, nc] = rid
                            q.append((nr, nc))

            rooms.append({"counts": color_counts, "area": area})

    # Exclude the main background component.
    bg_room_id = -1
    if rooms:
        best_bg_count = -1
        best_bg_area = -1
        for rid, room in enumerate(rooms):
            bc = room["counts"].get(bg_color, 0)
            if bc > best_bg_count or (bc == best_bg_count and room["area"] > best_bg_area):
                best_bg_count = bc
                best_bg_area = room["area"]
                bg_room_id = rid
        if best_bg_count == 0:
            bg_room_id = -1

    # ------------------------------------------------------------------
    # 6. Determine source rooms and their fill colors.
    # ------------------------------------------------------------------
    temp_nonbg_counts = []
    source_rooms = []

    for rid, room in enumerate(rooms):
        nonbg = {}
        if rid != bg_room_id:
            nonbg = {
                c: n
                for c, n in room["counts"].items()
                if c != bg_color
            }
        temp_nonbg_counts.append(nonbg)
        if nonbg:
            source_rooms.append(rid)

    if not source_rooms:
        if special is not None:
            sr, sc = special
            for rr in range(sr, sr + 3):
                for cc in range(sc, sc + 3):
                    if not (rr == sr + 1 and cc == sc + 1):
                        out[rr, cc] = wall_color
            out[sr + 1, sc + 1] = bg_color
        return out

    fg_total = {}
    for rid in source_rooms:
        for c, n in temp_nonbg_counts[rid].items():
            fg_total[c] = fg_total.get(c, 0) + n

    room_fill = {}
    room_nonbg_area = {}
    is_source_room = np.zeros(len(rooms), dtype=bool)

    for rid in source_rooms:
        nonbg = temp_nonbg_counts[rid]
        room_nonbg_area[rid] = sum(nonbg.values())
        fill = max(nonbg, key=lambda c: (nonbg[c], fg_total.get(c, 0), -c))
        room_fill[rid] = fill
        is_source_room[rid] = True

    # Source pixels for distance transform: non-background pixels in source rooms.
    source_id = np.full((H, W), -1, dtype=int)
    for r in range(H):
        for c in range(W):
            rid = nw_label[r, c]
            if rid != -1 and is_source_room[rid] and g[r, c] != bg_color:
                source_id[r, c] = rid

    if not np.any(source_id != -1):
        if special is not None:
            sr, sc = special
            for rr in range(sr, sr + 3):
                for cc in range(sc, sc + 3):
                    if not (rr == sr + 1 and cc == sc + 1):
                        out[rr, cc] = wall_color
            out[sr + 1, sc + 1] = bg_color
        return out

    # ------------------------------------------------------------------
    # 7. Multi-source distance transform over rooms.
    # ------------------------------------------------------------------
    priority_scores = []
    for rid in source_rooms:
        fill = room_fill[rid]
        score = (
            fg_total.get(fill, 0),
            room_nonbg_area[rid],
            -fill,
        )
        priority_scores.append((score, rid))

    priority_scores.sort(reverse=True)
    rank = {rid: i for i, (_, rid) in enumerate(priority_scores)}

    INF = 10 ** 9
    dist = np.full((H, W), INF, dtype=int)
    best_rank = np.full((H, W), 10 ** 9, dtype=int)
    best_room = np.full((H, W), -1, dtype=int)
    heap = []

    for r in range(H):
        for c in range(W):
            rid = source_id[r, c]
            if rid != -1:
                rk = rank[rid]
                dist[r, c] = 0
                best_rank[r, c] = rk
                best_room[r, c] = rid
                heapq.heappush(heap, (0, rk, rid, r, c))

    while heap:
        d, rk, rid, r, c = heapq.heappop(heap)

        if d != dist[r, c] or rk != best_rank[r, c] or rid != best_room[r, c]:
            continue

        for dr, dc in dirs4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                nd = d + 1
                if (
                    dist[nr, nc] == INF
                    or nd < dist[nr, nc]
                    or (nd == dist[nr, nc] and rk < best_rank[nr, nc])
                ):
                    dist[nr, nc] = nd
                    best_rank[nr, nc] = rk
                    best_room[nr, nc] = rid
                    heapq.heappush(heap, (nd, rk, rid, nr, nc))

    # ------------------------------------------------------------------
    # 8. Replace wall cells by nearest room fill color.
    # ------------------------------------------------------------------
    for r in range(H):
        for c in range(W):
            if g[r, c] == wall_color:
                rid = best_room[r, c]
                if rid != -1:
                    out[r, c] = room_fill[rid]

    # ------------------------------------------------------------------
    # 9. Apply the special central 3x3 ring, if detected.
    # ------------------------------------------------------------------
    if special is not None:
        sr, sc = special

        for rr in range(sr, sr + 3):
            for cc in range(sc, sc + 3):
                if not (rr == sr + 1 and cc == sc + 1):
                    out[rr, cc] = wall_color

        color_room_area = {}
        for rid in source_rooms:
            fill = room_fill[rid]
            color_room_area[fill] = color_room_area.get(fill, 0) + room_nonbg_area[rid]

        if fg_total:
            center_color = max(
                fg_total,
                key=lambda c: (
                    fg_total[c],
                    color_room_area.get(c, 0),
                    -c,
                )
            )
        else:
            center_color = bg_color

        out[sr + 1, sc + 1] = center_color

    return out