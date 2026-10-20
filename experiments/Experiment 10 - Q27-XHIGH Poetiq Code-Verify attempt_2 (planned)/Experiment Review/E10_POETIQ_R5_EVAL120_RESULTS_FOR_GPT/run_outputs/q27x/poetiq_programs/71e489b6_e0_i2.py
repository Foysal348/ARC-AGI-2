import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by outlining thin zero holes with 7s and removing
    non-solid 1s.

    Rule:
    1. A zero cell is "thin" if it is not part of any 2x2 all-zero block.
    2. A thin zero cell is a seed if at least 3 of its 4 orthogonal neighbors
       are 1, with outside-the-grid treated as 1.
    3. Select every 4-connected component of thin zeros that contains a seed.
    4. Set all 8-neighbors of selected zeros to 7, while selected zeros stay 0.
    5. Keep only 1s that are part of at least one 2x2 all-1 block, unless
       overwritten by 7. All other cells become 0.
    """
    grid = np.array(grid, dtype=int, copy=True)
    if grid.size == 0:
        return grid

    h, w = grid.shape

    def in_2x2(value_mask: np.ndarray) -> np.ndarray:
        """
        Return a mask of cells that belong to at least one 2x2 block
        where all four cells are True in value_mask.
        """
        mask = np.zeros(value_mask.shape, dtype=bool)
        if value_mask.shape[0] < 2 or value_mask.shape[1] < 2:
            return mask

        quad = (
            value_mask[:-1, :-1] &
            value_mask[1:, :-1] &
            value_mask[:-1, 1:] &
            value_mask[1:, 1:]
        )

        mask[:-1, :-1] |= quad
        mask[1:, :-1] |= quad
        mask[:-1, 1:] |= quad
        mask[1:, 1:] |= quad
        return mask

    one = grid == 1
    zero = grid == 0

    # Zero cells that are too thick are not hole centers.
    thick_zero = in_2x2(zero)
    thin_zero = zero & ~thick_zero

    # Count orthogonal 1-neighbors. Outside the grid is treated as 1.
    one_pad = np.pad(one.astype(np.int8), 1, mode="constant", constant_values=1)
    one_count = (
        one_pad[:-2, 1:-1] +  # up
        one_pad[2:, 1:-1] +   # down
        one_pad[1:-1, :-2] +  # left
        one_pad[1:-1, 2:]     # right
    )

    seed = thin_zero & (one_count >= 3)

    # Select 4-connected thin-zero components containing at least one seed.
    selected = np.zeros((h, w), dtype=bool)

    if np.any(seed):
        visited = np.zeros((h, w), dtype=bool)
        directions = ((1, 0), (-1, 0), (0, 1), (0, -1))

        for r in range(h):
            for c in range(w):
                if not thin_zero[r, c] or visited[r, c]:
                    continue

                stack = [(r, c)]
                visited[r, c] = True
                component = []
                has_seed = False

                while stack:
                    cr, cc = stack.pop()
                    component.append((cr, cc))

                    if seed[cr, cc]:
                        has_seed = True

                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < h and
                            0 <= nc < w and
                            thin_zero[nr, nc] and
                            not visited[nr, nc]
                        ):
                            visited[nr, nc] = True
                            stack.append((nr, nc))

                if has_seed:
                    for cr, cc in component:
                        selected[cr, cc] = True

    # Solid 1s are 1s belonging to at least one 2x2 all-1 block.
    solid_one = in_2x2(one)

    out = np.zeros((h, w), dtype=int)
    out[solid_one] = 1

    # Dilate selected holes by their 8-neighborhood and paint with 7.
    if np.any(selected):
        padded = np.pad(selected, 1, mode="constant", constant_values=False)
        seven = np.zeros((h, w), dtype=bool)

        for dr in range(3):
            for dc in range(3):
                seven |= padded[dr:dr + h, dc:dc + w]

        out[seven] = 7

        # Hole centers remain 0.
        out[selected] = 0

    return out