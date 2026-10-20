import numpy as np
from collections import deque


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble outer frames and inner patterns.

    Rule
    ----
    1. Background is the most frequent color.
    2. Find all connected components of non-background colors.
       Each component is an object; its bounding box defines its size.
    3. The larger half of the objects are outer frames, and the smaller half
       are inner patterns.
    4. Pair outers with inners by color:
         - process outer frames in increasing color order;
         - for each outer, choose the unused inner frame with the next larger
           color;
         - if no larger color is available, wrap to the smallest unused color.
    5. Preserve the spatial row/column layout of the outer frames, but compact
       it by removing background gaps.
    6. For each paired outer/inner object, fill a rectangle with the outer
       color and paste the inner pattern centered inside it.  Background holes
       in the inner pattern remain as the outer color.
    """
    if grid.size == 0:
        return np.zeros_like(grid)
    if grid.ndim != 2:
        return grid.copy()

    H, W = grid.shape

    # ------------------------------------------------------------
    # 1. Background color
    # ------------------------------------------------------------
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[np.argmax(counts)])

    # ------------------------------------------------------------
    # 2. Object extraction
    # ------------------------------------------------------------
    objects = []

    for color_val in colors:
        color = int(color_val)
        if color == background:
            continue

        mask = grid == color
        seen = np.zeros((H, W), dtype=bool)

        for r in range(H):
            for c in range(W):
                if not mask[r, c] or seen[r, c]:
                    continue

                # BFS over one connected component of this color.
                q = deque([(r, c)])
                seen[r, c] = True

                top = bottom = r
                left = right = c

                while q:
                    y, x = q.popleft()

                    if y < top:
                        top = y
                    if y > bottom:
                        bottom = y
                    if x < left:
                        left = x
                    if x > right:
                        right = x

                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            if mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                q.append((ny, nx))

                h = bottom - top + 1
                w = right - left + 1

                # Use all pixels of this color inside the bounding box.
                # This keeps disconnected same-color details inside a frame
                # if they appear in the same rectangular object.
                obj_mask = grid[top:bottom + 1, left:right + 1] == color

                objects.append({
                    "color": color,
                    "top": top,
                    "left": left,
                    "bottom": bottom,
                    "right": right,
                    "h": h,
                    "w": w,
                    "area": h * w,
                    "mask": obj_mask,
                })

    if not objects:
        return np.zeros_like(grid)

    # Stable identifiers.
    for i, obj in enumerate(objects):
        obj["uid"] = i

    # ------------------------------------------------------------
    # 3. Split into outer frames and inner patterns
    # ------------------------------------------------------------
    sorted_objects = sorted(
        objects,
        key=lambda o: (-o["area"], o["color"], o["top"], o["left"]),
    )

    n = len(sorted_objects)

    if n == 1:
        outers = [sorted_objects[0]]
        inners = []
    else:
        m = n // 2
        outers = sorted_objects[:m]
        inners = sorted_objects[m:m + m]

    if not outers:
        return np.zeros_like(grid)

    # ------------------------------------------------------------
    # 4. Pair outer frames with inner patterns by color
    # ------------------------------------------------------------
    assigned = {}
    pool = list(inners)

    for outer in sorted(outers, key=lambda o: (o["color"], o["top"], o["left"])):
        if not pool:
            break

        c = outer["color"]

        # Next larger unused inner color.
        candidates = [p for p in pool if p["color"] > c]

        # If none exists, wrap to the smallest unused color.
        if not candidates:
            candidates = [p for p in pool if p["color"] != c]

        if candidates:
            chosen = min(
                candidates,
                key=lambda p: (p["color"], p["top"], p["left"]),
            )
        else:
            chosen = min(
                pool,
                key=lambda p: (p["color"], p["top"], p["left"]),
            )

        assigned[outer["uid"]] = chosen
        pool = [p for p in pool if p["uid"] != chosen["uid"]]

    # ------------------------------------------------------------
    # 5. Layout of outer frames
    # ------------------------------------------------------------
    def same_row(a, b):
        # Objects are in the same visual row if their tops are close.
        return abs(a["top"] - b["top"]) <= max(1, min(a["h"], b["h"]) // 2)

    sorted_outers = sorted(outers, key=lambda o: (o["top"], o["left"]))
    row_groups = []

    for obj in sorted_outers:
        placed = False
        for group in row_groups:
            if same_row(obj, group[0]):
                group.append(obj)
                placed = True
                break
        if not placed:
            row_groups.append([obj])

    row_groups.sort(
        key=lambda g: (
            min(o["top"] for o in g),
            min(o["left"] for o in g),
        )
    )

    for group in row_groups:
        group.sort(key=lambda o: o["left"])

    row_heights = [max(o["h"] for o in group) for group in row_groups]
    row_widths = [sum(o["w"] for o in group) for group in row_groups]

    out_h = sum(row_heights)
    out_w = max(row_widths) if row_widths else 1

    output = np.zeros((out_h, out_w), dtype=grid.dtype)

    # ------------------------------------------------------------
    # 6. Build combined blocks
    # ------------------------------------------------------------
    def make_block(outer, inner):
        block = np.full((outer["h"], outer["w"]), outer["color"], dtype=grid.dtype)

        if inner is None:
            # Fallback for unpaired outer: keep its original component pixels.
            block[:] = 0
            block[outer["mask"]] = outer["color"]
            return block

        ih = inner["h"]
        iw = inner["w"]

        if ih <= outer["h"] and iw <= outer["w"]:
            off_r = (outer["h"] - ih) // 2
            off_c = (outer["w"] - iw) // 2
            region = block[off_r:off_r + ih, off_c:off_c + iw]
            region[inner["mask"]] = inner["color"]
        else:
            # Robust fallback: crop the larger inner pattern to the outer size.
            start_r = (ih - outer["h"]) // 2
            start_c = (iw - outer["w"]) // 2
            crop_mask = inner["mask"][
                start_r:start_r + outer["h"],
                start_c:start_c + outer["w"],
            ]
            block[crop_mask] = inner["color"]

        return block

    y = 0
    for ri, group in enumerate(row_groups):
        x = 0
        for outer in group:
            inner = assigned.get(outer["uid"])
            block = make_block(outer, inner)
            bh, bw = block.shape

            # Top-align within the row; rows are stacked without gaps.
            output[y:y + bh, x:x + bw] = block
            x += bw

        y += row_heights[ri]

    return output