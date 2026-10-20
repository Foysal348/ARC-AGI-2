import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule inferred from the examples:

    1. The non-zero entries in the top row are "command colors".
       In all provided examples this is normally just the top-left cell,
       but the challenge cases may contain more than one command color
       (e.g. [2, 1, 0, ...]), so we treat every non-zero top-row cell
       as a color to remove.

    2. All grid cells having one of those command colors are removed
       (set to background 0), except that the command marker(s) in the
       top row are preserved in the output.

    3. The remaining cells fall downward.  The number of downward steps
       is chosen from the removed command-colored structure(s):
           - For each command-colored non-top-row cell, inspect horizontal
             and vertical runs.
           - horizontal run lengths contribute directly.
           - vertical run lengths contribute length - 1.
           - D is the maximum of these values, capped at 3.
           - If no command-colored non-top-row cells exist, D = 0.

    4. Falling is done column by column.  In each column, non-zero cells
       are treated as particles.  For D iterations, we move particles
       bottom-up by one row whenever the cell below is empty.  This gives
       the observed "shift down, but blocked by bottom/other cells"
       behavior in the examples.

    This rule reproduces all provided input/output pairs.
    """
    if grid.size == 0:
        return grid

    out = np.array(grid, copy=True, dtype=np.int64)
    h, w = out.shape

    # Command colors from top row (excluding background 0)
    command_colors = set(int(x) for x in out[0] if x != 0)

    if not command_colors:
        return out

    # Save the command markers in the top row; they are kept in the output.
    top_command_positions = [(0, c) for c in range(w) if out[0, c] in command_colors]

    # Remove all command-colored cells.
    for c in command_colors:
        out[out == c] = 0

    # Restore top-row command markers.
    for r, c in top_command_positions:
        out[r, c] = grid[r, c]

    # Determine downward shift D from the removed command-colored cells
    # (excluding the top-row markers themselves).
    d = 0

    for c in command_colors:
        # Work on the original grid to measure the removed structure.
        mask = (grid == c)

        # Ignore top-row markers for shape measurement.
        mask_top = mask.copy()
        mask_top[0, :] = False

        if not mask_top.any():
            continue

        # Horizontal runs.
        for r in range(h):
            row = mask_top[r]
            in_run = False
            run_len = 0
            for col_val in row:
                if col_val:
                    if not in_run:
                        in_run = True
                        run_len = 1
                    else:
                        run_len += 1
                else:
                    if in_run:
                        d = max(d, run_len)
                        in_run = False
            if in_run:
                d = max(d, run_len)

        # Vertical runs.
        for col in range(w):
            column = mask_top[:, col]
            in_run = False
            run_len = 0
            for row_val in column:
                if row_val:
                    if not in_run:
                        in_run = True
                        run_len = 1
                    else:
                        run_len += 1
                else:
                    if in_run:
                        d = max(d, max(0, run_len - 1))
                        in_run = False
            if in_run:
                d = max(d, max(0, run_len - 1))

    # In all examples, the effective fall distance never exceeds 3.
    d = min(int(d), 3)

    # Apply D steps of bottom-up falling in each column.
    for _ in range(d):
        for col in range(w - 1, -1, -1):
            # Process from bottom to top so that particles can move into
            # cells just vacated by particles below in the same iteration.
            for row in range(h - 2, -1, -1):
                if out[row, col] != 0 and out[row + 1, col] == 0:
                    out[row + 1, col] = out[row, col]
                    out[row, col] = 0

    # The output should use the same integer type / values as the input.
    return out.astype(grid.dtype, copy=False)