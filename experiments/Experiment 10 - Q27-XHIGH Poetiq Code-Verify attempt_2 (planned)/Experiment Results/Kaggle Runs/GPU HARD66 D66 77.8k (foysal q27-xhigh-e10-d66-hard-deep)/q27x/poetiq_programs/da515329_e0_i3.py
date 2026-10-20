import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Generate the recursive square-spiral fractal shown in the examples.

    The input contains one small cross made of 8s.  The cross has an empty
    centre.  Let L be its arm length.  The output is obtained by drawing
    four square spirals, one starting in each cardinal direction from the
    cross centre.

    Each spiral:
      * starts at the cross centre,
      * first segment length is A = L, except that L == 1 is treated as 2,
      * turns clockwise after every segment,
      * increases the segment length by 2 after every turn.

    The cells visited by the four spirals are XOR-ed: a cell is 8 if it is
    covered by an odd number of spiral segments, otherwise 0.  This XOR
    behaviour is necessary to reproduce the one-cell gaps and cancellations
    visible in the examples.

    Segments are clipped to the input grid, but the logical spiral endpoint
    is still used for the next segment, matching the reference outputs.
    """
    if grid is None or grid.size == 0:
        return np.asarray(grid)

    grid = np.asarray(grid)
    out = np.zeros_like(grid)
    h, w = grid.shape

    # Find all 8 cells; the input should contain exactly one cross.
    mask = grid == 8
    if not mask.any():
        return out

    ys, xs = np.where(mask)
    cr = int(np.median(ys))
    cc = int(np.median(xs))

    # Arm length: count consecutive 8s upward from the empty centre.
    arm = 0
    r = cr - 1
    while r >= 0 and grid[r, cc] == 8:
        arm += 1
        r -= 1

    # Fallbacks for malformed inputs.
    if arm == 0:
        # Try downward.
        r = cr + 1
        while r < h and grid[r, cc] == 8:
            arm += 1
            r += 1

    if arm == 0:
        # Try rightward.
        c = cc + 1
        while c < w and grid[cr, c] == 8:
            arm += 1
            c += 1

    if arm == 0:
        return out

    # In the examples, arm length 1 behaves like arm length 2 in the fractal.
    A = arm if arm > 1 else 2

    # Directions: up, right, down, left.
    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def turn_cw(dr, dc):
        # Clockwise rotation in grid coordinates.
        return dc, -dr

    def draw_spiral(start_dr, start_dc, initial_dir):
        dr, dc = initial_dir
        r = cr
        c = cc
        length = A

        # A safe upper bound; ARC grids are small.
        for _ in range(2000):
            if length > 4 * (h + w) + 100:
                break

            nr = r + dr * length
            nc = c + dc * length

            # Draw the portion of the segment that intersects the grid.
            # The starting cell is not drawn; the endpoint is drawn if inside.
            rr = r
            ccnt = c
            for _i in range(length):
                rr += dr
                ccnt += dc
                if 0 <= rr < h and 0 <= ccnt < w:
                    out[rr, ccnt] ^= 8

            # Move to the logical endpoint, even if outside the grid.
            r, c = nr, nc
            dr, dc = turn_cw(dr, dc)
            length += 2

    # Four spirals, one for each initial arm direction.
    for d in dirs:
        draw_spiral(0, 0, d)

    return out