import numpy as np
from collections import deque, defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid by routing the colored 3x3 payloads through the
    8/5/9 "machine".

    Observed rule:
    - The machine is drawn with colors 8, 5, and 9.
      * 8 and 5 form thin 1-cell-wide wires.
      * 9 forms thick straight wire segments.
      * At 4-way intersections, wires cross but do NOT connect.
    - Payloads are 3x3 blocks/diamonds in colors other than {8,5,9}.
    - For every connected wire path connecting two ports:
        * the color present at either end is propagated to both ends,
        * the payload at each port is rendered as the 3x3 diamond:
              0 C 0
              C 0 C
              0 C 0
        * empty ports are filled with the routed color.
    """
    grid = np.array(grid, dtype=int)
    if grid.ndim != 2:
        return grid

    H, W = grid.shape
    if H < 3 or W < 3:
        return grid

    MACHINE = {8, 5, 9}
    mask = np.isin(grid, list(MACHINE))

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def in_bounds(r, c):
        return 0 <= r < H and 0 <= c < W

    def is_machine(r, c):
        return in_bounds(r, c) and mask[r, c]

    # Thin machine pixels: 8 and 5
    thin = np.isin(grid, [8, 5])

    # Connected components of thick 9-regions.
    comp_id = -np.ones((H, W), dtype=int)
    ncomp = 0
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 9 and comp_id[r, c] == -1:
                q = deque([(r, c)])
                comp_id[r, c] = ncomp
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if in_bounds(nr, nc) and grid[nr, nc] == 9 and comp_id[nr, nc] == -1:
                            comp_id[nr, nc] = ncomp
                            q.append((nr, nc))
                ncomp += 1

    # For each 9-component and direction, precompute straight exits.
    # entry_exit[comp][dir] maps boundary cell -> exit cell (last 9 in that dir)
    entry_exit = defaultdict(lambda: [defaultdict() for _ in range(4)])
    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    for comp in range(ncomp):
        cells = list(zip(*np.where(comp_id == comp)))
        # If there are too many cells, skip; in this task they are compact rectangles.
        if not cells:
            continue
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)

        # Up exit: for each column, if topmost cell in component, store bottommost in same column
        for c in range(cmin, cmax + 1):
            col_cells = [r for r, cc in cells if cc == c]
            if col_cells:
                top = min(col_cells)
                bottom = max(col_cells)
                entry_exit[comp][0][(top, c)] = (bottom, c)  # up
                entry_exit[comp][2][(bottom, c)] = (top, c)  # down
            # Left/right exits
        for r in range(rmin, rmax + 1):
            row_cells = [c for rr, c in cells if rr == r]
            if row_cells:
                left = min(row_cells)
                right = max(row_cells)
                entry_exit[comp][1][(r, left)] = (r, right)  # right
                entry_exit[comp][3][(r, right)] = (r, left)  # left

    # -------------------------------------------------------------------------
    # Find path endpoints: thin 8/5 pixels with exactly one machine neighbor
    # -------------------------------------------------------------------------
    endpoints = []

    for r in range(H):
        for c in range(W):
            if not thin[r, c]:
                continue

            nbrs = []
            for d, (dr, dc) in enumerate(dirs):
                nr, nc = r + dr, c + dc
                if in_bounds(nr, nc) and mask[nr, nc]:
                    nbrs.append(d)

            if len(nbrs) == 1:
                endpoints.append((r, c))

    # Group endpoints that belong to the same physical path.
    # Trace from each endpoint until reaching another endpoint.
    #
    # Tracing rules:
    # - If entering a 9-region, jump straight through it to the opposite side.
    # - In thin wires, prefer going straight; if no straight move, take the bend.
    # - At crossings (4-way), going straight keeps the paths separated.

    parent = {ep: -1 for ep in endpoints}
    path_groups = []
    visited_endpoints = set()

    def trace(start):
        """Return list of cells from start to the first other endpoint reached."""
        path = [start]
        r, c = start

        # initial direction: only neighbor
        d0 = None
        for d, (dr, dc) in enumerate(dirs):
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and mask[nr, nc]:
                d0 = d
                break
        if d0 is None:
            return path, None

        # Move into first machine cell
        dr, dc = dirs[d0]
        nr, nc = r + dr, c + dc
        prev = (r, c)
        r, c = nr, nc
        path.append((r, c))

        # If first step entered a 9-region, jump straight through it.
        if grid[r, c] == 9:
            comp = comp_id[r, c]
            exit_cell = entry_exit[comp][d0].get((r, c))
            if exit_cell is not None:
                r, c = exit_cell
                prev = (r, c)
                # continue one step beyond the 9-component if possible
                dr, dc = dirs[d0]
                nr, nc = r + dr, c + dc
                if in_bounds(nr, nc) and mask[nr, nc]:
                    prev = (r, c)
                    r, c = nr, nc
                    path.append((r, c))

        steps = 0
        limit = 100000

        while steps < limit:
            steps += 1

            if (r, c) in endpoints and (r, c) != start:
                return path, (r, c)

            # Choose next move.
            # Direction into current cell is from prev to (r,c).
            pr, pc = prev
            in_dir = None
            for d, (dr, dc) in enumerate(dirs):
                if r + dr == pr and c + dc == pc:
                    in_dir = d
                    break

            # Opposite direction is the straight continuation.
            straight_dir = (in_dir + 2) % 4 if in_dir is not None else None

            candidates = []

            if straight_dir is not None:
                dr, dc = dirs[straight_dir]
                nr, nc = r + dr, c + dc
                if in_bounds(nr, nc) and mask[nr, nc]:
                    candidates.append(straight_dir)

            # If no straight candidate, allow bends (perpendicular directions).
            if not candidates and in_dir is not None:
                for d in ((in_dir + 1) % 4, (in_dir + 3) % 4):
                    dr, dc = dirs[d]
                    nr, nc = r + dr, c + dc
                    if in_bounds(nr, nc) and mask[nr, nc]:
                        candidates.append(d)

            if not candidates:
                return path, None

            # Prefer straight if present; otherwise the only bend.
            d = candidates[0]
            dr, dc = dirs[d]
            nr, nc = r + dr, c + dc
            prev = (r, c)
            r, c = nr, nc
            path.append((r, c))

            # If we entered a 9-region, jump straight through it.
            if grid[r, c] == 9:
                comp = comp_id[r, c]
                exit_cell = entry_exit[comp][d].get((r, c))
                if exit_cell is not None:
                    er, ec = exit_cell
                    prev = (r, c)
                    r, c = er, ec
                    # Continue one step beyond the 9-component if possible.
                    dr, dc = dirs[d]
                    nr, nc = r + dr, c + dc
                    if in_bounds(nr, nc) and mask[nr, nc]:
                        prev = (r, c)
                        r, c = nr, nc
                        path.append((r, c))

        return path, None

    # Build path pairing between endpoints.
    pair_map = {}
    for ep in endpoints:
        if ep in visited_endpoints:
            continue
        path, end2 = trace(ep)
        if end2 is None:
            continue
        visited_endpoints.add(ep)
        visited_endpoints.add(end2)
        pair_map[ep] = end2
        pair_map[end2] = ep

    # -------------------------------------------------------------------------
    # Infer port rectangles from endpoints.
    # A port is a 3x3 region just outside the machine at the endpoint.
    # -------------------------------------------------------------------------
    def port_rect(ep):
        r, c = ep
        # Direction from endpoint into machine
        d_in = None
        for d, (dr, dc) in enumerate(dirs):
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and mask[nr, nc]:
                d_in = d
                break

        if d_in is None:
            return None

        # The 3x3 port lies opposite to d_in.
        # dirs: 0 up, 1 right, 2 down, 3 left
        if d_in == 2:       # machine is below -> port above
            r1, r2 = r - 3, r
            c1, c2 = c - 1, c + 2
        elif d_in == 0:      # machine is above -> port below
            r1, r2 = r + 1, r + 4
            c1, c2 = c - 1, c + 2
        elif d_in == 1:      # machine is right -> port left
            r1, r2 = r - 1, r + 2
            c1, c2 = c - 3, c
        elif d_in == 3:      # machine is left -> port right
            r1, r2 = r - 1, r + 2
            c1, c2 = c + 1, c + 4
        else:
            return None

        if not (0 <= r1 < r2 <= H and 0 <= c1 < c2 <= W):
            return None

        # The port rectangle should not contain machine pixels.
        if mask[r1:r2, c1:c2].any():
            return None

        return (r1, r2, c1, c2)

    def read_color(rect):
        r1, r2, c1, c2 = rect
        sub = grid[r1:r2, c1:c2]
        vals = sub[sub != 0]
        # Ignore machine colors just in case.
        vals = vals[~np.isin(vals, MACHINE)]
        if vals.size == 0:
            return None
        vals, counts = np.unique(vals, return_counts=True)
        return int(vals[np.argmax(counts)])

    # Map endpoints to rectangles and colors.
    ep_rect = {}
    ep_color = {}
    for ep in endpoints:
        rect = port_rect(ep)
        if rect is None:
            continue
        ep_rect[ep] = rect
        ep_color[ep] = read_color(rect)

    # -------------------------------------------------------------------------
    # Produce output.
    # Start from input, then for each paired port pair, set both ports to diamond.
    # -------------------------------------------------------------------------
    out = grid.copy()

    def set_diamond(rect, color):
        if rect is None or color is None:
            return
        r1, r2, c1, c2 = rect
        out[r1:r2, c1:c2] = 0
        # plus shape
        out[r1 + 1, c1] = color
        out[r1 + 1, c1 + 2] = color
        out[r1, c1 + 1] = color
        out[r1 + 2, c1 + 1] = color

    used = set()
    for ep, other in pair_map.items():
        if ep in used:
            continue
        used.add(ep)
        used.add(other)

        color = ep_color.get(ep)
        if color is None:
            color = ep_color.get(other)

        if color is None:
            continue

        set_diamond(ep_rect.get(ep), color)
        set_diamond(ep_rect.get(other), color)

    # If any endpoints were not paired but have a color, normalize their own port.
    for ep in endpoints:
        if ep in used:
            continue
        color = ep_color.get(ep)
        if color is not None:
            set_diamond(ep_rect.get(ep), color)

    return out