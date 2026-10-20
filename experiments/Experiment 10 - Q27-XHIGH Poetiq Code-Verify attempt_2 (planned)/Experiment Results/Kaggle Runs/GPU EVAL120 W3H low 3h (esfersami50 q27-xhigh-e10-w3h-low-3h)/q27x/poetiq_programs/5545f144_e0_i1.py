import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Identify separator columns (uniform non-background vertical lines),
    split into panels, and output positions where the object appears
    in a strict majority (>50%) of panels.
    """
    grid = grid.copy()
    rows, cols = grid.shape
    
    # Find background color
    values, counts = np.unique(grid, return_counts=True)
    bg_color = values[np.argmax(counts)]
    
    # Find separator columns
    sep_cols = []
    for c in range(cols):
        col_vals = grid[:, c]
        if len(np.unique(col_vals)) == 1 and col_vals[0] != bg_color:
            sep_cols.append(c)
    
    if not sep_cols:
        return grid
    
    # Panel boundaries (skip separator columns)
    boundaries = [0] + sep_cols + [cols]
    panel_starts = []
    for i in range(len(boundaries) - 1):
        if boundaries[i+1] - boundaries[i] > 0:
            panel_starts.append(boundaries[i])
    
    n_panels = len(panel_starts)
    panel_width = boundaries[1] - boundaries[0] if len(boundaries) > 1 else cols
    
    # Determine separator colors to exclude
    sep_colors = set(int(grid[r, c]) for c in sep_cols for r in range(rows))
    
    output = np.full((rows, panel_width), bg_color, dtype=grid.dtype)
    
    for r in range(rows):
        for c in range(panel_width):
            count = 0
            obj_val = None
            for p in range(n_panels):
                actual_col = panel_starts[p] + c
                val = int(grid[r, actual_col])
                if val != bg_color and val not in sep_colors:
                    count += 1
                    obj_val = val
            if count > n_panels / 2 and obj_val is not None:
                output[r, c] = obj_val
    
    return output