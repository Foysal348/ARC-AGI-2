import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by highlighting the outer boundary of the region containing 6.

    - The region is the 4-connected component of cells {8, 6} containing the 6.
    - 1-wall components that do not touch this region are erased to 8.
    - Kept wall components that touch the grid border are treated as outer boundary.
    - Paint 7 on region cells that are on the grid border or in the 8-neighborhood
      of those border-connected kept wall cells.
    """
    grid = np.asarray(grid)
    if grid.ndim != 2:
        return grid.copy()

    new_grid = grid.copy()
    h, w = grid.shape
    if h == 0 or w == 0:
        return new_grid

    six_pos = np.argwhere(grid == 6)
    if len(six_pos) == 0:
        return new_grid

    def flood4(mask: np.ndarray, seeds) -> np.ndarray:
        """4-connected flood fill."""
        visited = np.zeros(mask.shape, dtype=bool)
        q = deque()

        for r, c in seeds:
            r, c = int(r), int(c)
            if 0 <= r < h and 0 <= c < w and mask[r, c] and not visited[r, c]:
                visited[r, c] = True
                q.append((r, c))

        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    q.append((nr, nc))

        return visited

    def dilate8(mask: np.ndarray) -> np.ndarray:
        """8-neighbor dilation, including the original mask."""
        out = np.zeros(mask.shape, dtype=bool)
        if mask.size == 0:
            return out
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        for dr in range(3):
            for dc in range(3):
                out |= padded[dr:dr + h, dc:dc + w]
        return out

    def label8(mask: np.ndarray):
        """Label 8-connected components of a boolean mask."""
        labels = np.zeros(mask.shape, dtype=int)
        comp = 0

        for r in range(h):
            for c in range(w):
                if mask[r, c] and labels[r, c] == 0:
                    comp += 1
                    labels[r, c] = comp
                    q = deque([(r, c)])

                    while q:
                        r0, c0 = q.popleft()
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = r0 + dr, c0 + dc
                                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                                    labels[nr, nc] = comp
                                    q.append((nr, nc))

        return labels, comp

    # Region containing the 6.
    floor = (grid == 8) | (grid == 6)
    room = flood4(floor, six_pos)

    # Wall components.
    wall = (grid == 1)
    wall_labels, n_wall_comps = label8(wall)

    edge = np.zeros((h, w), dtype=bool)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True

    # Keep wall components that touch the room (8-neighbor touch).
    room_neighborhood = dilate8(room)
    adjacent_wall = wall & room_neighborhood

    touches_room = np.zeros(n_wall_comps + 1, dtype=bool)
    if np.any(adjacent_wall):
        touches_room[wall_labels[adjacent_wall]] = True

    keep_walls = wall & touches_room[wall_labels]

    # Among kept walls, mark components that touch the grid border.
    border_wall = wall & edge
    touches_border = np.zeros(n_wall_comps + 1, dtype=bool)
    if np.any(border_wall):
        touches_border[wall_labels[border_wall]] = True

    border_walls = keep_walls & touches_border[wall_labels]

    # Erase walls that do not belong to the room's wall set.
    new_grid[wall & ~keep_walls] = 8

    # Paint the outer boundary of the room.
    wall_halo = dilate8(border_walls)
    boundary = room & (edge | wall_halo)
    new_grid[boundary & (grid == 8)] = 7

    return new_grid