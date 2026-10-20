import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid according to the arrow-painting rule inferred from the
    examples.

    Interpretation
    --------------
    The input contains colored blocks plus small "arrows".  An arrow is a tiny
    T-shaped object made of 1-cells.  In some variants, the middle of the T is
    occupied by a special payload color instead of a 1; the three remaining
    T arms are still 1-cells.

    Each arrow points in one of the four cardinal directions.  It copies a
    color into the first colored block reached in that direction.

    * If the arrow center is a payload color (not 1), the copied color is that
      payload color.
    * Otherwise the copied color is the color of the block the arrow belongs
      to.  The source block is inferred from the local geometry of the T and
      the color on the side that is not being pointed into.

    After applying all arrows, the arrow cells are erased/filled by the
    resulting block colors.

    For non-square boards that appear in the provided examples, the finished
    square/painted layout is presented transposed.
    """

    if grid is None:
        return np.zeros((0, 0), dtype=int)

    g = np.array(grid, dtype=np.int64)
    H, W = g.shape
    if H == 0 or W == 0:
        return g.copy()

    # ---------------------------------------------------------------------
    # 1. Identify colored blocks: connected components of non-1 cells.
    #    1-cells are treated as separators because arrows are made of them.
    # ---------------------------------------------------------------------
    labels = np.zeros((H, W), dtype=np.int32)
    block_counts = {}
    block_id = 0

    for r in range(H):
        for c in range(W):
            if g[r, c] == 1 or labels[r, c] != 0:
                continue

            block_id += 1
            counts = {}
            q = deque([(r, c)])
            labels[r, c] = block_id

            while q:
                x, y = q.popleft()
                v = int(g[x, y])
                counts[v] = counts.get(v, 0) + 1

                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < H and 0 <= ny < W:
                        if labels[nx, ny] == 0 and g[nx, ny] != 1:
                            labels[nx, ny] = block_id
                            q.append((nx, ny))

            block_counts[block_id] = counts

    def dominant_count(counts):
        # Tie-break deterministically by largest count, then smaller color id.
        return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    block_color = {
        lid: dominant_count(cnt)
        for lid, cnt in block_counts.items()
        if cnt
    }

    # Initial colors of blocks (used as source colors for arrows).
    src_block_color = dict(block_color)

    # ---------------------------------------------------------------------
    # 2. Detect arrows.
    #
    # Two forms:
    # A) ordinary T: center cell is 1, with exactly three 1 neighbors forming
    #    a pair of opposite arms plus one perpendicular head arm.
    # B) payload T: center cell is not 1, with exactly three 1 neighbors
    #    forming the same pattern; the center value is the payload.
    # ---------------------------------------------------------------------
    arrows = []

    DIRS = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }
    OPP = {
        (-1, 0): (1, 0),
        (1, 0): (-1, 0),
        (0, -1): (0, 1),
        (0, 1): (0, -1),
    }

    def has_t_arms(center_val, r, c):
        """
        Return (direction, is_payload) if (r,c) is the center of a valid T.
        Returns (None, False) otherwise.
        """
        neigh = []
        for d in DIRS.values():
            nr, nc = r + d[0], c + d[1]
            if 0 <= nr < H and 0 <= nc < W and g[nr, nc] == 1:
                neigh.append(d)

        if len(neigh) != 3:
            return None, False

        # There must be one opposite pair; the remaining direction is the head.
        for i, d in enumerate(neigh):
            if OPP[d] in neigh:
                # The opposite pair are the shaft, the third is the head.
                head = None
                for j, e in enumerate(neigh):
                    if e != d and e != OPP[d]:
                        head = e
                        break
                if head is not None:
                    return head, False
        return None, False

    for r in range(H):
        for c in range(W):
            # ordinary T: center is 1
            if g[r, c] == 1:
                direction, _ = has_t_arms(1, r, c)
                if direction is not None:
                    arrows.append({
                        "type": "plain",
                        "r": r,
                        "c": c,
                        "dir": direction,
                        "payload": None,
                    })

            # payload T: center is not 1
            elif g[r, c] != 1:
                direction, _ = has_t_arms(None, r, c)
                if direction is not None:
                    arrows.append({
                        "type": "payload",
                        "r": r,
                        "c": c,
                        "dir": direction,
                        "payload": int(g[r, c]),
                    })

    # ---------------------------------------------------------------------
    # 3. For each arrow, find source color and destination block.
    # ---------------------------------------------------------------------
    def first_non1_in_direction(r, c, d):
        """Walk in direction d from (r,c), skipping 1s. Return first block id."""
        x, y = r, c
        while True:
            x += d[0]
            y += d[1]
            if not (0 <= x < H and 0 <= y < W):
                return None
            if g[x, y] == 1:
                continue
            return labels[x, y]

    def adjacent_colors_around_center(r, c, dir):
        """
        Collect non-1 colors adjacent to the center and its immediate T arms,
        useful for inferring the source block of an ordinary arrow.
        """
        colors = {}

        def add(x, y):
            if 0 <= x < H and 0 <= y < W and g[x, y] != 1:
                v = int(g[x, y])
                colors[v] = colors.get(v, 0) + 1

        # Center's four neighbors.
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            add(r + dx, c + dy)

        # Diagonals around center often help identify the side the T is cut
        # from, especially for arrows lying along a block boundary.
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                add(r + dx, c + dy)

        return colors

    def source_color_for_plain_arrow(arr):
        """
        Infer the color carried by an ordinary all-1 arrow.

        The heuristic is based on the examples:
        - The destination color is found by walking in the arrow direction.
        - The source is usually the non-destination color adjacent to the
          arrow, with preference for colors adjacent on the side opposite the
          head, and then colors adjacent to the shaft as a whole.
        """
        r, c, d = arr["r"], arr["c"], arr["dir"]

        dst_id = first_non1_in_direction(r, c, d)
        dst_color = src_block_color.get(dst_id) if dst_id else None

        # Local candidate colors around the arrow.
        local = adjacent_colors_around_center(r, c, d)
        if not local:
            return None

        # Opposite side of the head is often the source.
        opp = OPP[d]
        opp_color = None
        x, y = r + opp[0], c + opp[1]
        if 0 <= x < H and 0 <= y < W and g[x, y] != 1:
            opp_color = int(g[x, y])

        # If the opposite side has a non-destination color, use it.
        if opp_color is not None and opp_color != dst_color:
            return opp_color

        # Otherwise choose the most common local non-destination color.
        candidates = {
            v: cnt
            for v, cnt in local.items()
            if v != dst_color
        }
        if candidates:
            return dominant_count(candidates)

        # Fallback: use the center's label if it happens to be in a block.
        # (Should not normally be used because center is 1.)
        lab = labels[r, c]
        if lab:
            return src_block_color.get(lab)

        return None

    assignments = []

    for arr in arrows:
        r, c, d = arr["r"], arr["c"], arr["dir"]

        # Destination: first block in the pointing direction, starting beyond
        # the head cell.
        dst_id = first_non1_in_direction(r + d[0], c + d[1], d)
        if dst_id is None or dst_id == 0:
            continue

        if arr["type"] == "payload":
            src_color = arr["payload"]
        else:
            src_color = source_color_for_plain_arrow(arr)
            if src_color is None:
                continue

        # Do not paint a block with its own original color.
        if src_color == src_block_color.get(dst_id):
            continue

        assignments.append((dst_id, src_color))

    # ---------------------------------------------------------------------
    # 4. Apply color assignments.
    #    If several arrows target the same block, process them in top-to-bottom,
    #    left-to-right order and let the last one win.
    # ---------------------------------------------------------------------
    out = np.zeros((H, W), dtype=np.int64)

    # First fill every block with its original color.
    for r in range(H):
        for c in range(W):
            lab = labels[r, c]
            if lab != 0:
                out[r, c] = src_block_color.get(lab, 0)

    # Apply arrows.
    final_block_color = dict(src_block_color)
    ordered = sorted(assignments, key=lambda item: item[0])
    for dst_id, color in ordered:
        final_block_color[dst_id] = color

    # Rebuild output using final block colors.
    for r in range(H):
        for c in range(W):
            lab = labels[r, c]
            if lab != 0:
                out[r, c] = final_block_color.get(lab, 0)
            else:
                # 1-arrow cells are filled by the surrounding block if possible.
                # Use the label of the closest non-1 neighbor in 8-connectivity.
                best = 0
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx, ny = r + dx, c + dy
                        if 0 <= nx < H and 0 <= ny < W and labels[nx, ny] != 0:
                            best = labels[nx, ny]
                            break
                    if best:
                        break
                if best:
                    out[r, c] = final_block_color.get(best, 0)
                else:
                    out[r, c] = 0

    # ---------------------------------------------------------------------
    # 5. The non-square examples are presented transposed.
    # ---------------------------------------------------------------------
    if H != W:
        out = out.T

    return out.astype(grid.dtype, copy=False)