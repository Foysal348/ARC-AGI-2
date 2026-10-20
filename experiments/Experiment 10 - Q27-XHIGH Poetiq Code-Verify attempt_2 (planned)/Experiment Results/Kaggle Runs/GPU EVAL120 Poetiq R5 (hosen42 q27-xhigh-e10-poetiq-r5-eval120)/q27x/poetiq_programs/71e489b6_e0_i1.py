import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid as follows:

    1. Remove 1 pixels that are not part of any 2x2 all-1 block.
    2. Consider zero pixels in the resulting grid.
       - A zero is "thin" if it is not part of any 2x2 all-zero block.
       - A thin zero is a "seed" if at least 3 of its 4 cardinal neighbours
         are 1, treating outside-the-grid as 1.
       - Select every 4-connected component of thin zeros that contains a seed.
    3. For every selected zero pixel:
       - keep it as 0
       - set all 8-neighbours to 7

    The output uses only 0, 1, and 7.
    """
    grid = np.array(grid, dtype=int, copy=True)
    if grid.size == 0:
        return grid

    h, w = grid.shape

    def in_2x2_blocks(arr: np.ndarray, value: int) -> np.ndarray:
        """Return mask of cells belonging to at least one 2x2 block of `value`."""
        mask = np.zeros(arr.shape, dtype=bool)
        if arr.shape[0] < 2 or arr.shape[1] < 2:
            return mask

        a = arr == value
        quad = (
            a[:-1, :-1] &
            a[1:, :-1] &
            a[:-1, 1:] &
            a[1:, 1:]
        )

        mask[:-1, :-1] |= quad
        mask[1:, :-1] |= quad
        mask[:-1, 1:] |= quad
        mask[1:, 1:] |= quad
        return mask

    # ------------------------------------------------------------------
    # Stage 1: keep only 1s that are part of some 2x2 all-1 block.
    # ------------------------------------------------------------------
    solid_1 = in_2x2_blocks(grid, 1)
    base = grid.copy()
    base[(grid == 1) & (~solid_1)] = 0

    # ------------------------------------------------------------------
    # Stage 2: find thin zero components that contain a capped zero.
    # ------------------------------------------------------------------
    zero = base == 0
    thick_zero = in_2x2_blocks(base, 0)
    thin_zero = zero & (~thick_zero)

    # Count cardinal 1-neighbours.  The padding value is 1, so the outside
    # of the grid behaves like a wall of 1s.
    one = (base == 1).astype(int)
    padded = np.pad(one, pad_width=1, mode="constant", constant_values=1)

    one_count = (
        padded[:-2, 1:-1] +  # up
        padded[2:, 1:-1] +   # down
        padded[1:-1, :-2] +  # left
        padded[1:-1, 2:]     # right
    )

    seed_zero = thin_zero & (one_count >= 3)

    # Label thin-zero components by simple BFS/DFS.
    selected_zero = np.zeros((h, w), dtype=bool)
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
                y, x = stack.pop()
                component.append((y, x))

                if seed_zero[y, x]:
                    has_seed = True

                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < h and
                        0 <= nx < w and
                        thin_zero[ny, nx] and
                        not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            if has_seed:
                for y, x in component:
                    selected_zero[y, x] = True

    # ------------------------------------------------------------------
    # Stage 3: draw 7 in the 8-neighborhood of selected zeros.
    # ------------------------------------------------------------------
    out = base.copy()
    seven_mask = np.zeros((h, w), dtype=bool)

    for r, c in np.argwhere(selected_zero):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    seven_mask[nr, nc] = True

    out[seven_mask] = 7

    # Selected zero centers must remain 0 even if they are neighbours of
    # another selected zero.
    out[selected_zero] = 0

    return out