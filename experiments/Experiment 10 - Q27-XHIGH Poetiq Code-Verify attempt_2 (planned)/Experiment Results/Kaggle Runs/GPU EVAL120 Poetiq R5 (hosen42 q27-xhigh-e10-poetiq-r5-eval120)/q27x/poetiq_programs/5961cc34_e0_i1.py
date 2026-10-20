import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid according to the following rule:

    1. The single yellow cell (4) with a short red tail (2) is a directed
       starting marker. The direction of motion is from the red tail toward
       the yellow cell. In all given examples this is upward, but the code
       also handles horizontal directions.

    2. A red path is drawn from the tail, through the yellow cell, in that
       direction until it hits a blue/green object. When the path hits such
       an object, the object is recolored red and the path continues from
       the object's green side in the outward direction indicated by that
       green side. This repeats until the path reaches a border or re-enters
       an already used path.

    3. After the directed path is complete, objects on the right-hand side
       of the directed path are recolored red. Objects on the other side are
       removed. This right-side region is found by flood-filling from a point
       immediately to the right of the directed path.

    All final non-background cells are color 2; everything else is background.
    """
    grid = np.asarray(grid)
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape

    # Background is the most frequent color (8 in the examples).
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    target = 2

    out = np.full((H, W), bg, dtype=grid.dtype)

    def inb(r, c):
        return 0 <= r < H and 0 <= c < W

    # ------------------------------------------------------------
    # Find all blue/green objects.
    # Each object is a connected component of colors 1 and 3.
    # A green side gives the outward direction for that object.
    # ------------------------------------------------------------
    object_mask = (grid == 1) | (grid == 3)
    visited = np.zeros((H, W), dtype=bool)
    blocks = []
    block_id_at = np.full((H, W), -1, dtype=int)
    neigh4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(H):
        for c in range(W):
            if not object_mask[r, c] or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            cells = []

            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in neigh4:
                    nr, nc = cr + dr, cc + dc
                    if inb(nr, nc) and object_mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            # We only treat components containing blue as objects.
            if not any(grid[r, c] == 1 for r, c in cells):
                continue

            bid = len(blocks)
            blocks.append({"cells": cells, "dir": None, "starts": [], "port_len": 0})
            for r, c in cells:
                block_id_at[r, c] = bid

    # Determine green port direction and start cells for each block.
    for blk in blocks:
        ones = [(r, c) for r, c in blk["cells"] if grid[r, c] == 1]
        threes = [(r, c) for r, c in blk["cells"] if grid[r, c] == 3]

        if not ones or not threes:
            continue

        rmin = min(r for r, _ in ones)
        rmax = max(r for r, _ in ones)
        cmin = min(c for _, c in ones)
        cmax = max(c for _, c in ones)

        top = [p for p in threes if p[0] < rmin]
        bottom = [p for p in threes if p[0] > rmax]
        left = [p for p in threes if p[1] < cmin]
        right = [p for p in threes if p[1] > cmax]

        candidates = [
            ((-1, 0), top),
            ((1, 0), bottom),
            ((0, -1), left),
            ((0, 1), right),
        ]

        # Choose the side containing the most green cells.
        direction, port_cells = max(candidates, key=lambda x: len(x[1]))

        if not port_cells:
            # Fallback: use centroid comparison if green cells are not
            # strictly outside the blue bounding box.
            tr = sum(r for r, _ in threes) / len(threes)
            tc = sum(c for _, c in threes) / len(threes)
            orow = (rmin + rmax) / 2.0
            ocol = (cmin + cmax) / 2.0

            if abs(tr - orow) >= abs(tc - ocol):
                direction = (-1, 0) if tr < orow else (1, 0)
            else:
                direction = (0, -1) if tc < ocol else (0, 1)

            port_cells = threes

        dr, dc = direction
        blk["dir"] = direction
        blk["port_len"] = len(port_cells)
        blk["starts"] = [(r + dr, c + dc) for r, c in port_cells]

    # ------------------------------------------------------------
    # Find the yellow starting cell and the direction indicated by
    # the red tail.
    # ------------------------------------------------------------
    pos4 = np.argwhere(grid == 4)
    if len(pos4) == 0:
        # No valid source marker: return an empty background grid.
        return out

    cy, cx = int(pos4[0][0]), int(pos4[0][1])

    # Color the source marker itself.
    out[grid == 4] = target
    out[grid == 2] = target

    best_tail_dir = None
    best_tail_len = 0

    for td in neigh4:
        r, c = cy + td[0], cx + td[1]
        if inb(r, c) and grid[r, c] == 2:
            length = 0
            rr, cc = r, c
            while inb(rr, cc) and grid[rr, cc] == 2:
                length += 1
                rr += td[0]
                cc += td[1]
            if length > best_tail_len:
                best_tail_len = length
                best_tail_dir = td

    if best_tail_dir is not None:
        arrow = (-best_tail_dir[0], -best_tail_dir[1])
    else:
        # Fallback: use the centroid of all red cells if present, else up.
        pos2 = np.argwhere(grid == 2)
        if len(pos2):
            mr, mc = pos2.mean(axis=0)
            dy = cy - mr
            dx = cx - mc
            if abs(dy) >= abs(dx):
                arrow = (-1, 0) if dy > 0 else (1, 0) if dy < 0 else (-1, 0)
            else:
                arrow = (0, -1) if dx > 0 else (0, 1) if dx < 0 else (-1, 0)
        else:
            arrow = (-1, 0)

    adr, adc = arrow

    # Directed path segments, used to find the right-hand side.
    # Each segment is (r0, c0, r1, c1, dr, dc), directed from start to end.
    segments = []

    # Add the tail segment, directed from the far tail end toward the 4.
    if best_tail_dir is not None:
        tdr, tdc = best_tail_dir
        fr, fc = cy, cx
        rr, cc = cy + tdr, cx + tdc
        while inb(rr, cc) and grid[rr, cc] == 2:
            fr, fc = rr, cc
            rr += tdr
            cc += tdc
        if (fr, fc) != (cy, cx):
            segments.append((fr, fc, cy, cx, adr, adc))

    # ------------------------------------------------------------
    # Trace the red path.
    # ------------------------------------------------------------
    activated = set()
    queue = deque()

    def activate(bid):
        if bid not in activated:
            activated.add(bid)
            for r, c in blocks[bid]["cells"]:
                out[r, c] = target
            queue.append(bid)

    def trace_ray(start_cells, dr, dc):
        """
        Extend a ray from start_cells in direction (dr, dc).
        Returns block IDs hit by the ray.
        """
        lanes = []
        for r, c in start_cells:
            if inb(r, c):
                # current, start, end
                lanes.append([r, c, r, c, r, c])

        if not lanes:
            return []

        while True:
            hit_ids = set()
            non_block = []
            crossing = False

            for lane in lanes:
                r, c = lane[0], lane[1]
                bid = block_id_at[r, c]
                if bid != -1:
                    hit_ids.add(bid)
                elif out[r, c] == target:
                    crossing = True
                else:
                    non_block.append((r, c))

            if hit_ids:
                for r, c in non_block:
                    out[r, c] = target

                for lane in lanes:
                    r, c = lane[0], lane[1]
                    sr, sc = lane[2], lane[3]
                    if (r, c) != (sr, sc):
                        segments.append((sr, sc, r, c, dr, dc))
                return list(hit_ids)

            if crossing:
                for lane in lanes:
                    er, ec = lane[4], lane[5]
                    sr, sc = lane[2], lane[3]
                    if (er, ec) != (sr, sc):
                        segments.append((sr, sc, er, ec, dr, dc))
                return []

            # No block or crossing: color the current cells and advance.
            for lane in lanes:
                r, c = lane[0], lane[1]
                out[r, c] = target
                lane[4], lane[5] = r, c

            next_lanes = []
            for lane in lanes:
                r, c = lane[0], lane[1]
                nr, nc = r + dr, c + dc
                if inb(nr, nc):
                    next_lanes.append([nr, nc, lane[2], lane[3], lane[4], lane[5]])
                else:
                    sr, sc = lane[2], lane[3]
                    er, ec = lane[4], lane[5]
                    if (er, ec) != (sr, sc):
                        segments.append((sr, sc, er, ec, dr, dc))

            if not next_lanes:
                return []
            lanes = next_lanes

    # Start from the cell in front of the yellow head.
    cr, cc = cy + adr, cx + adc
    if inb(cr, cc):
        for bid in trace_ray([(cr, cc)], adr, adc):
            activate(bid)

    # Follow exits of activated objects.
    while queue:
        bid = queue.popleft()
        blk = blocks[bid]
        if blk["dir"] is None or not blk["starts"]:
            continue

        dr, dc = blk["dir"]
        for nbid in trace_ray(blk["starts"], dr, dc):
            activate(nbid)

    # ------------------------------------------------------------
    # Find the right-hand side of the directed path by flood fill.
    # ------------------------------------------------------------
    seed = None

    def try_seed_from_segment(seg):
        r0, c0, r1, c1, dr, dc = seg
        length = max(abs(r1 - r0), abs(c1 - c0))
        if length == 0:
            return None

        # Sample points along the segment and step right by 1..3 cells.
        for t in range(1, length + 1):
            r = r0 + dr * t
            c = c0 + dc * t
            if not inb(r, c):
                continue

            # Right side vector for direction (dr, dc) is (dc, -dr).
            for k in range(1, 4):
                sr = r + dc * k
                sc = c - dr * k
                if inb(sr, sc) and out[sr, sc] != target:
                    return (sr, sc)
        return None

    for seg in segments:
        seed = try_seed_from_segment(seg)
        if seed is not None:
            break

    # Fallback seed searches.
    if seed is None:
        for t in range(0, min(10, H, W) + 1):
            r = cy + adr * t
            c = cx + adc * t
            if not inb(r, c):
                break
            for k in range(1, 4):
                sr = r + adc * k
                sc = c - adr * k
                if inb(sr, sc) and out[sr, sc] != target:
                    seed = (sr, sc)
                    break
            if seed is not None:
                break

    if seed is None:
        for bid in activated:
            blk = blocks[bid]
            if blk["dir"] is None:
                continue
            dr, dc = blk["dir"]
            for sr0, sc0 in blk["starts"]:
                if not inb(sr0, sc0):
                    continue
                for t in range(0, min(10, H, W) + 1):
                    r = sr0 + dr * t
                    c = sc0 + dc * t
                    if not inb(r, c):
                        break
                    for k in range(1, 4):
                        sr = r + dc * k
                        sc = c - dr * k
                        if inb(sr, sc) and out[sr, sc] != target:
                            seed = (sr, sc)
                            break
                    if seed is not None:
                        break
                if seed is not None:
                    break
            if seed is not None:
                break

    region = np.zeros((H, W), dtype=bool)

    if seed is not None:
        sr, sc = seed
        region[sr, sc] = True
        stack = [(sr, sc)]

        while stack:
            r, c = stack.pop()
            for dr, dc in neigh4:
                nr, nc = r + dr, c + dc
                if inb(nr, nc) and not region[nr, nc] and out[nr, nc] != target:
                    region[nr, nc] = True
                    stack.append((nr, nc))

    # Recolor unactivated objects lying in the right-hand region.
    if seed is not None:
        for bid, blk in enumerate(blocks):
            if bid in activated:
                continue

            keep = False
            for r, c in blk["cells"]:
                if region[r, c]:
                    keep = True
                    break

            if keep:
                for r, c in blk["cells"]:
                    out[r, c] = target
    else:
        # If no right-side region could be found, keep only single-cell-port
        # objects as a conservative fallback.
        for bid, blk in enumerate(blocks):
            if bid not in activated and blk["port_len"] == 1:
                for r, c in blk["cells"]:
                    out[r, c] = target

    return out