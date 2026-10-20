import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Trace a beam from the yellow/red source marker through blue/green objects.

    Rules:
    1. The source is a single 4 with a straight tail of 2s.
       The beam travels away from the tail, through the 4.
    2. Objects are connected components of colors 1 and 3 that contain at least one 1.
    3. If a beam hits an object:
       - the whole object becomes color 2,
       - a new beam exits from the green cells (3) of that object,
       - the exit direction is the open side of the green port,
       - the number of parallel lanes equals the number of exposed green cells.
    4. A beam stops when it leaves the grid, hits an already drawn beam,
       or hits an object. Activated objects may spawn new beams.
    5. Everything not drawn by the beam/source is background.
    """
    grid = np.asarray(grid)
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape

    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    target = 2

    out = np.full((H, W), bg, dtype=grid.dtype)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def inb(r, c):
        return 0 <= r < H and 0 <= c < W

    # Source markers.
    pos4 = np.argwhere(grid == 4)
    if len(pos4) == 0:
        return out

    out[grid == 4] = target
    out[grid == 2] = target

    def source_beam_dir(cy, cx):
        """
        Find the direction of the red tail next to the 4.
        The beam direction is opposite the tail.
        """
        best_tail_dir = None
        best_len = 0

        for dr, dc in dirs:
            r, c = cy + dr, cx + dc
            if inb(r, c) and grid[r, c] == 2:
                length = 1
                rr, cc = r, c
                while inb(rr + dr, cc + dc) and grid[rr + dr, cc + dc] == 2:
                    rr += dr
                    cc += dc
                    length += 1
                if length > best_len:
                    best_len = length
                    best_tail_dir = (dr, dc)

        if best_tail_dir is not None:
            return (-best_tail_dir[0], -best_tail_dir[1])

        # Fallback: point toward the nearest 2 if no adjacent tail is found.
        pos2 = np.argwhere(grid == 2)
        if len(pos2) == 0:
            return (-1, 0)

        dist = np.abs(pos2[:, 0].astype(int) - cy) + np.abs(pos2[:, 1].astype(int) - cx)
        idx = int(np.argmin(dist))
        nr, nc = int(pos2[idx, 0]), int(pos2[idx, 1])

        dy = nr - cy
        dx = nc - cx

        if abs(dy) >= abs(dx):
            if dy != 0:
                return (1, 0) if dy > 0 else (-1, 0)
        else:
            if dx != 0:
                return (0, 1) if dx > 0 else (0, -1)

        return (-1, 0)

    # Build object components from colors 1 and 3.
    object_mask = (grid == 1) | (grid == 3)
    block_id = np.full((H, W), -1, dtype=int)
    visited = np.zeros((H, W), dtype=bool)
    blocks = []

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

                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if inb(nr, nc) and object_mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            # Objects must contain at least one blue cell.
            if any(grid[rr, cc] == 1 for rr, cc in cells):
                bid = len(blocks)
                for rr, cc in cells:
                    block_id[rr, cc] = bid
                blocks.append({
                    "cells": cells,
                    "exit_dir": None,
                    "starts": []
                })

    def determine_exit(cells):
        """
        Determine the outward direction of an object's green port and the
        starting cells for the beam leaving that port.
        """
        threes = [(r, c) for r, c in cells if grid[r, c] == 3]
        if not threes:
            return None, []

        # Count green cells that are open to non-object space in each direction.
        exposed = {d: [] for d in dirs}
        for r, c in threes:
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if not inb(nr, nc) or not object_mask[nr, nc]:
                    exposed[(dr, dc)].append((r, c))

        # Votes from adjacent blue cells: a green cell facing blue means
        # the beam should exit away from that blue neighbor.
        votes = {d: 0 for d in dirs}
        for r, c in threes:
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if inb(nr, nc) and grid[nr, nc] == 1:
                    votes[(-dr, -dc)] += 1

        # Centroid fallback/tie-break: green centroid relative to blue centroid.
        ones = [(r, c) for r, c in cells if grid[r, c] == 1]
        centroid_dir = None
        if ones:
            orow = sum(r for r, c in ones) / len(ones)
            ocol = sum(c for _, c in ones) / len(ones)
            trow = sum(r for r, c in threes) / len(threes)
            tcol = sum(c for _, c in threes) / len(threes)

            dy = trow - orow
            dx = tcol - ocol

            if abs(dy) >= abs(dx):
                if dy < 0:
                    centroid_dir = (-1, 0)
                elif dy > 0:
                    centroid_dir = (1, 0)
            else:
                if dx < 0:
                    centroid_dir = (0, -1)
                elif dx > 0:
                    centroid_dir = (0, 1)

        def score(d):
            return (len(exposed[d]), votes[d], 1 if d == centroid_dir else 0)

        best_dir = max(dirs, key=score)

        if score(best_dir)[0] == 0:
            # No exposed green cells; fallback to all green cells.
            start_list = [(r + best_dir[0], c + best_dir[1]) for r, c in threes]
        else:
            start_list = [(r + best_dir[0], c + best_dir[1]) for r, c in exposed[best_dir]]

        # Unique starting cells.
        seen = set()
        starts = []
        for p in start_list:
            if p not in seen:
                seen.add(p)
                starts.append(p)

        return best_dir, starts

    for blk in blocks:
        d, starts = determine_exit(blk["cells"])
        blk["exit_dir"] = d
        blk["starts"] = starts

    # Trace beams.
    from collections import deque
    q = deque()

    for p in pos4:
        cy, cx = int(p[0]), int(p[1])
        bdir = source_beam_dir(cy, cx)
        sr, sc = cy + bdir[0], cx + bdir[1]
        if inb(sr, sc):
            q.append(([(sr, sc)], bdir))

    activated = np.zeros(len(blocks), dtype=bool)

    while q:
        starts, (dr, dc) = q.popleft()

        lanes = []
        seen_start = set()
        for r, c in starts:
            if inb(r, c) and (r, c) not in seen_start:
                seen_start.add((r, c))
                lanes.append((r, c))

        while lanes:
            next_lanes = []
            next_seen = set()

            for r, c in lanes:
                if not inb(r, c):
                    continue

                bid = block_id[r, c]

                if bid != -1:
                    # Hit an object: activate it and spawn its exit beam.
                    if not activated[bid]:
                        activated[bid] = True
                        for rr, cc in blocks[bid]["cells"]:
                            out[rr, cc] = target

                        d = blocks[bid]["exit_dir"]
                        sts = blocks[bid]["starts"]
                        if d is not None and sts:
                            q.append((sts, d))

                    # Lane stops at the object.
                    continue

                if out[r, c] == target:
                    # Stop at already drawn beam/source.
                    continue

                # Draw beam and advance.
                out[r, c] = target
                nr, nc = r + dr, c + dc
                if inb(nr, nc) and (nr, nc) not in next_seen:
                    next_seen.add((nr, nc))
                    next_lanes.append((nr, nc))

            lanes = next_lanes

    return out