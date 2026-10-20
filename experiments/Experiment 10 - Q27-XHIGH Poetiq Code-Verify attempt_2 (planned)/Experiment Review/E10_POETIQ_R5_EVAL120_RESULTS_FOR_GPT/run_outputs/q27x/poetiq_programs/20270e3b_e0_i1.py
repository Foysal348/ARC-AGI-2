import numpy as np
from collections import deque


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid according to the observed ARC rule:

    1. Color 7 forms two congruent components. One is the source cut marker,
       the other is the destination cut marker.
    2. If the source component is translated to the destination component by
       vector t = (tr, tc), then the movable 4-flap is translated by
       v = (tr - 1, tc).
    3. The movable flap consists of all 4 cells in the region:
         row >= top_of_source_7_component - 1
         col >= left_of_source_7_component
    4. After moving the flap, keep only 4-cells connected to at least one
       moved 4-cell. This removes disconnected leftover pieces.
    5. Crop to the bounding box of the kept cells and return the grid with
       background color 1.
    """
    if grid is None or grid.size == 0:
        return np.array([], dtype=int)

    # Convert to plain list of lists for easier BFS.
    g = [list(row) for row in grid]
    H, W = len(g), len(g[0])

    # Determine background color (almost always 1 in these examples).
    bg = 1
    colors = {}
    for r in range(H):
        for c in range(W):
            v = g[r][c]
            if v != 4 and v != 7:
                colors[v] = colors.get(v, 0) + 1
    if colors:
        bg = max(colors, key=colors.get)

    # Collect all 4 cells.
    all4 = set()
    for r in range(H):
        for c in range(W):
            if g[r][c] == 4:
                all4.add((r, c))

    # Find connected components of 7 (4-neighbor connectivity).
    seven_cells = [(r, c) for r in range(H) for c in range(W) if g[r][c] == 7]
    seen = set()
    comps = []

    for start in seven_cells:
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        comp = set()
        while q:
            r, c = q.popleft()
            comp.add((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if g[nr][nc] == 7 and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append((nr, nc))
        comps.append(comp)

    # If there are not two marker components, gracefully return a clean grid.
    if len(comps) < 2:
        out = np.array(g, dtype=int)
        out[out == 7] = bg
        return out

    def reachable_from_starts(combined: set, starts: set) -> set:
        """Return all cells in `combined` connected to any start cell."""
        visited = set()
        stack = [s for s in starts if s in combined]
        for s in stack:
            visited.add(s)
        while stack:
            r, c = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (r + dr, c + dc)
                if nb in combined and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        return visited

    candidates = []

    # Try every ordered pair of 7-components.
    for src in comps:
        for dst in comps:
            if src is dst or len(src) != len(dst):
                continue

            src_min_r = min(r for r, c in src)
            src_min_c = min(c for r, c in src)
            dst_min_r = min(r for r, c in dst)
            dst_min_c = min(c for r, c in dst)

            tr = dst_min_r - src_min_r
            tc = dst_min_c - src_min_c

            # Verify exact translation from source marker to destination marker.
            translated_src = {(r + tr, c + tc) for r, c in src}
            if translated_src != dst:
                continue

            # Flap translation vector.
            vr = tr - 1
            vc = tc

            # Movable flap: all 4s to the right of and from one row above source.
            moving4 = set()
            for r in range(max(0, src_min_r - 1), H):
                for c in range(src_min_c, W):
                    if g[r][c] == 4:
                        moving4.add((r, c))

            if not moving4:
                continue

            shifted = {(r + vr, c + vc) for r, c in moving4}

            # Every destination 7-cell must be covered by the shifted flap.
            if not dst.issubset(shifted):
                continue

            stationary4 = all4 - moving4

            # Reject candidates where the moved flap overlaps existing stationary 4s.
            if shifted & stationary4:
                continue

            combined = shifted | stationary4
            if not combined:
                continue

            # Keep moved cells and any stationary cells connected to moved cells.
            kept = reachable_from_starts(combined, shifted)
            if not kept:
                continue

            # Destination marker must remain part of the final object.
            if not dst.issubset(kept):
                continue

            min_r = min(r for r, c in kept)
            max_r = max(r for r, c in kept)
            min_c = min(c for r, c in kept)
            max_c = max(c for r, c in kept)

            area = (max_r - min_r + 1) * (max_c - min_c + 1)
            candidates.append((area, len(kept), kept))

    # Fallback if the exact component-translation rule does not find anything:
    # try single 7-cell anchors with the same flap idea.
    if not candidates:
        seven_set = {(r, c) for r, c in seven_cells}
        for sr, sc in seven_cells:
            if sr + 1 >= H or g[sr + 1][sc] != 4:
                continue

            for dr2, dc2 in seven_cells:
                if (dr2, dc2) == (sr, sc):
                    continue

                vr = dr2 - sr - 1
                vc = dc2 - sc

                moving4 = set()
                for r in range(max(0, sr - 1), H):
                    for c in range(sc, W):
                        if g[r][c] == 4:
                            moving4.add((r, c))

                if not moving4:
                    continue

                shifted = {(r + vr, c + vc) for r, c in moving4}
                if (dr2, dc2) not in shifted:
                    continue

                stationary4 = all4 - moving4
                if shifted & stationary4:
                    continue

                combined = shifted | stationary4
                if not combined:
                    continue

                kept = reachable_from_starts(combined, shifted)
                if not kept or (dr2, dc2) not in kept:
                    continue

                min_r = min(r for r, c in kept)
                max_r = max(r for r, c in kept)
                min_c = min(c for r, c in kept)
                max_c = max(c for r, c in kept)
                area = (max_r - min_r + 1) * (max_c - min_c + 1)
                candidates.append((area, len(kept), kept))

    # If no valid candidate, return input with markers removed.
    if not candidates:
        out = np.array(g, dtype=int)
        out[out == 7] = bg
        return out

    # Choose the candidate with smallest bounding box; tie-break by fewer cells.
    candidates.sort(key=lambda x: (x[0], x[1]))
    kept = candidates[0][2]

    min_r = min(r for r, c in kept)
    max_r = max(r for r, c in kept)
    min_c = min(c for r, c in kept)
    max_c = max(c for r, c in kept)

    out = np.full((max_r - min_r + 1, max_c - min_c + 1), bg, dtype=int)
    for r, c in kept:
        out[r - min_r, c - min_c] = 4

    return out