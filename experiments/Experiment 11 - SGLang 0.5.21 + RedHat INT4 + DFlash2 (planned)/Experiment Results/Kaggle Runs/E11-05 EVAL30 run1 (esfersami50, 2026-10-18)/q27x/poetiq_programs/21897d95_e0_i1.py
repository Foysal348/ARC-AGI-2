import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid into the required output mosaic.

    The rule inferred from the examples:
    - The non-1 colored regions form a fixed block layout.
    - Small 1-pixel "arrows" indicate that the color of a region should be
      copied into the neighbouring region in the arrow's direction.
    - In the examples, the output is the same block layout, but each block is
      filled with the colour arriving through an incoming arrow.
    - If several arrows target the same block, the examples are consistent
      with the last processed arrow winning; processing order is top-to-bottom,
      left-to-right over the detected arrow sources.

    This implementation detects the rectangular blocks implicitly by taking
    the connected non-1 background regions, then finds the 1-arrow shapes and
    applies their colour transfers to the block mosaic.  For the given task
    family, this reproduces the demonstrated outputs.
    """

    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape
    out = np.zeros((H, W), dtype=grid.dtype)

    # Background is 0 for the purpose of block detection; 1 is treated as a
    # special marker/arrow colour, not as part of a block.
    non1 = grid != 1

    # Connected components of non-1 pixels (4-connectivity).
    labels = np.zeros((H, W), dtype=int)
    label = 0
    block_color = {}

    for r in range(H):
        for c in range(W):
            if non1[r, c] and labels[r, c] == 0:
                label += 1
                stack = [(r, c)]
                labels[r, c] = label
                colour_counts = {}
                while stack:
                    x, y = stack.pop()
                    val = int(grid[x, y])
                    if val != 1:
                        colour_counts[val] = colour_counts.get(val, 0) + 1
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if non1[nx, ny] and labels[nx, ny] == 0:
                                labels[nx, ny] = label
                                stack.append((nx, ny))
                # The dominant non-1 colour in the component is its block colour.
                if colour_counts:
                    block_color[label] = max(colour_counts, key=colour_counts.get)

    # Determine the set of block colours present.  The output is a blocky
    # mosaic, so fill every pixel with the colour of its connected component
    # first.
    for r in range(H):
        for c in range(W):
            lab = labels[r, c]
            if lab != 0:
                out[r, c] = block_color.get(lab, 0)
            else:
                # 1-pixels are removed from the output.
                out[r, c] = 0

    # Detect the 1-arrow shapes.  In the examples they are small 3-cell or
    # 4-cell patterns of 1s.  The direction is determined by the side where
    # the "head" 1 lies relative to the three-cell shaft.
    def arrow_direction(cells):
        # cells: list of (r,c)
        if len(cells) < 3:
            return None

        # If exactly 4 cells, try the T-shape used in the examples.
        if len(cells) == 4:
            # Look for a 3-cell straight shaft plus one perpendicular cell.
            for i in range(len(cells)):
                a = cells[i]
                b = cells[(i + 1) % len(cells)]
                c = cells[(i + 2) % len(cells)]
                d = cells[(i + 3) % len(cells)]

                # Check if a,b,c form a straight line and d is adjacent to the middle.
                def is_line(p, q, r):
                    return (p[0] == q[0] == r[0]) or (p[1] == q[1] == r[1])

                if is_line(a, b, c):
                    # middle is the one adjacent to both
                    if (abs(a[0] - c[0]) == 1 and abs(a[1] - c[1]) == 1) or True:
                        pass

            # Simpler: try each cell as the head, and see if the other three
            # form a straight line with the middle adjacent to the head.
            for head in cells:
                others = [x for x in cells if x != head]
                # others must be a straight line of length 3
                if len(others) != 3:
                    continue
                o = set(others)
                for mid in others:
                    # the remaining two should be on opposite sides of mid
                    rem = [x for x in others if x != mid]
                    if len(rem) != 2:
                        continue
                    r1, r2 = rem
                    if (r1[0] == mid[0] == r2[0] and r1[1] < mid[1] < r2[1]) or \
                       (r1[0] == mid[0] == r2[0] and r2[1] < mid[1] < r1[1]) or \
                       (r1[1] == mid[1] == r2[1] and r1[0] < mid[0] < r2[0]) or \
                       (r1[1] == mid[1] == r2[1] and r2[0] < mid[0] < r1[0]):
                        # shaft found; head must be orthogonally adjacent to mid
                        if abs(head[0] - mid[0]) + abs(head[1] - mid[1]) == 1:
                            if head[1] > mid[1]:
                                return (0, 1)
                            if head[1] < mid[1]:
                                return (0, -1)
                            if head[0] > mid[0]:
                                return (1, 0)
                            if head[0] < mid[0]:
                                return (-1, 0)
        elif len(cells) == 3:
            # A 3-cell L or U shape can occur when a T is truncated at the
            # boundary.  Infer direction from the missing corner of the
            # bounding box, favouring the direction of the "elbow".
            rs = [r for r, c in cells]
            cs = [c for r, c in cells]
            rmin, rmax = min(rs), max(rs)
            cmin, cmax = min(cs), max(cs)

            # If it spans 2 rows and 2 cols, it's an L/T fragment.
            if rmax - rmin == 1 and cmax - cmin == 1:
                # Count how many cells are in the 2x2 box.
                # The missing cell direction is usually the arrow direction
                # in the examples where the shape is truncated.
                for dr in (-1, 0):
                    for dc in (-1, 0):
                        if (rmin + dr, cmin + dc) not in cells:
                            # Missing corner; direction towards the missing
                            # corner is a reasonable heuristic for the given
                            # puzzle family.
                            return (dr, dc)
            elif rmax - rmin == 1 and cmax - cmin == 0:
                # vertical line of 2 plus one?
                if len(cells) == 3:
                    # Determine side protrusion
                    mid = (rmin + rmax) // 2
                    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                        if (mid + dr, cmin + dc) in cells:
                            return (dr, dc)
            elif rmax - rmin == 0 and cmax - cmin == 1:
                mid = (cmin + cmax) // 2
                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                    if (rmin + dr, mid + dc) in cells:
                        return (dr, dc)

        return None

    # Find connected components of 1s (8-connectivity, because some arrows
    # are diagonal in the examples).
    visited = np.zeros((H, W), dtype=bool)
    arrows = []

    for r in range(H):
        for c in range(W):
            if grid[r, c] == 1 and not visited[r, c]:
                stack = [(r, c)]
                visited[r, c] = True
                cells = []
                while stack:
                    x, y = stack.pop()
                    cells.append((x, y))
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < H and 0 <= ny < W:
                                if grid[nx, ny] == 1 and not visited[nx, ny]:
                                    visited[nx, ny] = True
                                    stack.append((nx, ny))

                d = arrow_direction(cells)
                if d is not None:
                    arrows.append((cells, d))

    # For each arrow, find the source block colour and apply it to the
    # destination block(s).  The source colour is the non-1 colour adjacent
    # to the arrow shaft/centre.  In the examples this is the dominant colour
    # of the block containing the arrow, except when a single special colour
    # is embedded in the arrow shape.
    def payload_color(cells):
        # Collect non-1 colours adjacent to the arrow cells.
        adj = {}
        for x, y in cells:
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and grid[nx, ny] != 1:
                    val = int(grid[nx, ny])
                    adj[val] = adj.get(val, 0) + 1
        if not adj:
            return None

        # Prefer a singleton embedded colour (e.g. 7, 9, 6) over the large
        # background block colour when the arrow is tightly around it.
        # Heuristic: if there is a colour that appears only once and is not
        # the most common adjacent colour, it is often the special payload.
        counts = sorted(adj.items(), key=lambda kv: (-kv[1], kv[0]))
        # If there is a low-count non-1 colour and the arrow cells surround it
        # (i.e. it is adjacent to multiple 1s), use it.
        for x, y in cells:
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and grid[nx, ny] != 1:
                    val = int(grid[nx, ny])
                    if adj[val] == 1:
                        # check if this pixel is surrounded by at least 2 1s
                        around = 0
                        for ax, ay in ((1,0),(-1,0),(0,1),(0,-1)):
                            xx, yy = nx + ax, ny + ay
                            if 0 <= xx < H and 0 <= yy < W and grid[xx, yy] == 1:
                                around += 1
                        if around >= 2:
                            return val
        # Otherwise use the dominant adjacent colour.
        return counts[0][0]

    # Build a block-id -> pixel list and block-id -> colour map for filling.
    block_pixels = {}
    for r in range(H):
        for c in range(W):
            lab = labels[r, c]
            if lab != 0:
                block_pixels.setdefault(lab, []).append((r, c))

    # For destination determination, use the block containing the arrow's
    # "tail" (the shaft centre) and move one block in the arrow direction.
    # Since blocks are rectangular, we can approximate by scanning in the
    # direction from the arrow centre until leaving the source block.
    def source_block(cells):
        # Use the most central cell of the arrow as the anchor.
        if len(cells) == 4:
            # If it is a T, the shaft centre is the cell adjacent to three
            # others.
            for cell in cells:
                neigh = sum(1 for other in cells if other != cell and
                            abs(other[0] - cell[0]) + abs(other[1] - cell[1]) == 1)
                if neigh == 3:
                    return labels[cell[0], cell[1]], cell
        # Fallback: use the average centre rounded.
        r = int(round(sum(p[0] for p in cells) / len(cells)))
        c = int(round(sum(p[1] for p in cells) / len(cells)))
        return labels[r, c], (r, c)

    def destination_block(src_lab, anchor, direction):
        dx, dy = direction
        x, y = anchor
        # Step one pixel at a time until entering a different block.
        while 0 <= x < H and 0 <= y < W:
            x += dx
            y += dy
            if not (0 <= x < H and 0 <= y < W):
                break
            lab = labels[x, y]
            if lab != 0 and lab != src_lab:
                return lab
        return None

    # Apply arrows.  Order matters slightly; top-to-bottom, left-to-right is
    # consistent with the examples.
    arrows.sort(key=lambda item: (min(r for r, c in item[0]), min(c for r, c in item[0])))
    for cells, d in arrows:
        payload = payload_color(cells)
        if payload is None:
            continue
        src_lab, anchor = source_block(cells)
        dst_lab = destination_block(src_lab, anchor, d)
        if dst_lab is None:
            # If no distinct destination is found (e.g. arrow points out of
            # bounds or into the same block), leave the output unchanged.
            continue
        for px, py in block_pixels.get(dst_lab, []):
            out[px, py] = payload

    return out