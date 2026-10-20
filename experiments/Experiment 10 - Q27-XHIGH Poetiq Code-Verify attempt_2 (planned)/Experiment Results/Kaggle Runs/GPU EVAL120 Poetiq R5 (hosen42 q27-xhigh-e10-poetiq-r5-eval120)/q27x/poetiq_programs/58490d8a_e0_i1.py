import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule:
    1. Find the rectangular blank region: the bounding box of all 0 cells.
    2. Each nonzero pixel inside this region is a color seed/label.
    3. For each seed color, count the number of 8-connected components of that
       color outside the blank region.
    4. In the output, create a zero grid of the blank region's size and place
       that many pixels of the seed color in the seed's row, starting at the
       seed's column and then every 2 columns.
    """
    grid = np.array(grid, dtype=np.int64, copy=True)

    if grid.ndim != 2 or grid.size == 0:
        return grid

    # Locate the blank zero region.
    zero_coords = np.argwhere(grid == 0)
    if zero_coords.size == 0:
        return grid

    rmin = int(zero_coords[:, 0].min())
    cmin = int(zero_coords[:, 1].min())
    rmax = int(zero_coords[:, 0].max())
    cmax = int(zero_coords[:, 1].max())

    height = rmax - rmin + 1
    width = cmax - cmin + 1

    # Mask for inside/outside the blank region.
    inside = np.zeros(grid.shape, dtype=bool)
    inside[rmin:rmax + 1, cmin:cmax + 1] = True
    outside = ~inside

    output = np.zeros((height, width), dtype=grid.dtype)

    # Seed pixels inside the blank region.
    seeds = np.argwhere((grid != 0) & inside)
    if seeds.size == 0:
        return output

    # Use the first occurrence of each color as its seed position.
    seed_positions = {}
    for r, c in seeds:
        color = int(grid[r, c])
        if color not in seed_positions:
            seed_positions[color] = (int(r - rmin), int(c - cmin))

    def count_components(color: int) -> int:
        """Count 8-connected components of a color outside the blank region."""
        mask = (grid == color) & outside
        if not np.any(mask):
            return 0

        visited = np.zeros(grid.shape, dtype=bool)
        h, w = grid.shape
        count = 0

        rs, cs = np.where(mask)
        for start_r, start_c in zip(rs, cs):
            start_r = int(start_r)
            start_c = int(start_c)

            if visited[start_r, start_c]:
                continue

            count += 1
            visited[start_r, start_c] = True
            stack = [(start_r, start_c)]

            while stack:
                r, c = stack.pop()

                for dr in (-1, 0, 1):
                    nr = r + dr
                    if nr < 0 or nr >= h:
                        continue

                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue

                        nc = c + dc
                        if nc < 0 or nc >= w:
                            continue

                        if mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))

        return count

    # Fill the output canvas with counted markers.
    for color, (sr, sc) in seed_positions.items():
        n = count_components(color)
        for k in range(n):
            col = sc + 2 * k
            if col >= width:
                break
            output[sr, col] = color

    return output