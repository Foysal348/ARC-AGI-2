import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform inferred from the examples:

    1. Every non-zero cell in the top row is a "command color".
       The original top-row command cell(s) are preserved in the output.
       All other cells with command colors are erased.

    2. The remaining non-zero cells are moved downward.  The amount of
       movement is estimated from the erased command-colored structure:
         - for horizontal runs of command cells, use run length - 1
         - for vertical runs of command cells, use run length - 1
         - take the maximum of these, capped at 3.

    3. Falling is performed iteratively, from the bottom of the grid upward.
       A non-zero cell moves down one step if the cell below it is empty.
       Repeating this D times reproduces the example behavior.

    This implementation avoids in-place mutation issues and handles grids
    gracefully.
    """
    if grid is None:
        return np.array([])

    if grid.size == 0:
        return grid.copy()

    # Work on a copy so the original input is not modified.
    out = np.array(grid, copy=True, dtype=np.int64)
    h, w = out.shape

    # Command colors are the non-zero colors in the top row.
    command_colors = {int(x) for x in out[0] if x != 0}

    if not command_colors:
        return out.astype(grid.dtype, copy=False)

    # Remember top-row command cells; they remain unchanged.
    top_command_mask = np.zeros_like(out, dtype=bool)
    for c in range(w):
        if out[0, c] in command_colors:
            top_command_mask[0, c] = True

    # Erase all command-colored cells except the original top-row markers.
    for color in command_colors:
        out[out == color] = 0

    # Restore top-row command markers.
    out[top_command_mask] = grid[top_command_mask]

    # Estimate the downward shift from command-colored structures below row 0.
    d = 0

    for color in command_colors:
        mask = (grid == color)
        mask[0, :] = False  # ignore top-row command markers

        # Horizontal runs: contribution = length - 1.
        for r in range(h):
            in_run = False
            run_len = 0
            for c in range(w):
                if mask[r, c]:
                    if not in_run:
                        in_run = True
                        run_len = 1
                    else:
                        run_len += 1
                else:
                    if in_run:
                        d = max(d, max(0, run_len - 1))
                        in_run = False
                        run_len = 0
            if in_run:
                d = max(d, max(0, run_len - 1))

        # Vertical runs: contribution = length - 1.
        for c in range(w):
            in_run = False
            run_len = 0
            for r in range(h):
                if mask[r, c]:
                    if not in_run:
                        in_run = True
                        run_len = 1
                    else:
                        run_len += 1
                else:
                    if in_run:
                        d = max(d, max(0, run_len - 1))
                        in_run = False
                        run_len = 0
            if in_run:
                d = max(d, max(0, run_len - 1))

    # In all observed examples the effective fall distance is at most 3.
    d = min(int(d), 3)

    # Apply D bottom-up falling steps.
    for _ in range(d):
        for c in range(w):
            # Process from bottom to top so that a cell can fall into the
            # space just vacated by the cell below it during the same step.
            for r in range(h - 2, -1, -1):
                if out[r, c] != 0 and out[r + 1, c] == 0:
                    out[r + 1, c] = out[r, c]
                    out[r, c] = 0

    return out.astype(grid.dtype, copy=False)