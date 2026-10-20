import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a grid containing background 4 and object pixels 1.

    Rule:
    1. Find 4-connected components of 1.
    2. Flood-fill background 4 from the grid border to identify exterior
       background. Background not connected to the border is a hole.
    3. A component is "holed" if it is adjacent to at least one hole.
    4. Output:
       - holed components become color 8;
       - non-holed components remain color 1;
       - holes become color 6;
       - exterior background cells adjacent (8-neighborhood) to any object
         become color 2, forming an outline.
    """
    grid = np.array(grid, dtype=int, copy=True)
    if grid.size == 0:
        return grid

    H, W = grid.shape
    background = (grid == 4)
    objects = (grid == 1)

    if not objects.any():
        return grid

    four_neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Label object components using 4-connectivity.
    obj_label = np.zeros((H, W), dtype=int)
    n_objects = 0

    for r in range(H):
        for c in range(W):
            if objects[r, c] and obj_label[r, c] == 0:
                n_objects += 1
                obj_label[r, c] = n_objects
                stack = [(r, c)]

                while stack:
                    y, x = stack.pop()
                    for dy, dx in four_neighbors:
                        ny, nx = y + dy, x + dx
                        if (
                            0 <= ny < H
                            and 0 <= nx < W
                            and objects[ny, nx]
                            and obj_label[ny, nx] == 0
                        ):
                            obj_label[ny, nx] = n_objects
                            stack.append((ny, nx))

    # Flood-fill exterior background from the border using 4-connectivity.
    outside = np.zeros((H, W), dtype=bool)
    q = deque()

    for r in range(H):
        if background[r, 0] and not outside[r, 0]:
            outside[r, 0] = True
            q.append((r, 0))
        if background[r, W - 1] and not outside[r, W - 1]:
            outside[r, W - 1] = True
            q.append((r, W - 1))

    for c in range(W):
        if background[0, c] and not outside[0, c]:
            outside[0, c] = True
            q.append((0, c))
        if background[H - 1, c] and not outside[H - 1, c]:
            outside[H - 1, c] = True
            q.append((H - 1, c))

    while q:
        y, x = q.popleft()
        for dy, dx in four_neighbors:
            ny, nx = y + dy, x + dx
            if (
                0 <= ny < H
                and 0 <= nx < W
                and background[ny, nx]
                and not outside[ny, nx]
            ):
                outside[ny, nx] = True
                q.append((ny, nx))

    # Holes are background cells not connected to the border.
    holes = background & ~outside

    # Mark which object components enclose at least one hole.
    component_has_hole = np.zeros(n_objects + 1, dtype=bool)

    if holes.any():
        hole_cells = np.argwhere(holes)
        for y, x in hole_cells:
            for dy, dx in four_neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    cid = obj_label[ny, nx]
                    if cid > 0:
                        component_has_hole[cid] = True

    out = grid.copy()

    # Recolor object components.
    for cid in range(1, n_objects + 1):
        mask = (obj_label == cid)
        out[mask] = 8 if component_has_hole[cid] else 1

    # Recolor holes.
    out[holes] = 6

    # Build an 8-neighbor dilation of the object mask.
    padded = np.pad(objects, 1, mode="constant", constant_values=0).astype(bool)
    dilated_objects = np.zeros((H, W), dtype=bool)

    for dy in range(3):
        for dx in range(3):
            dilated_objects |= padded[dy:dy + H, dx:dx + W]

    # Outline: exterior background cells adjacent to any object.
    outline = outside & dilated_objects & ~objects
    out[outline] = 2

    return out