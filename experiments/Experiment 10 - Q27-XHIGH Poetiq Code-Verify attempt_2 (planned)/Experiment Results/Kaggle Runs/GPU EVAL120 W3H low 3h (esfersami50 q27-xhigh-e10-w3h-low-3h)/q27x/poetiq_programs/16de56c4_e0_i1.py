import numpy as np
from math import gcd
from functools import reduce

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform by creating periodic patterns from non-zero pixels.
    
    For each row: period = GCD of non-zero column positions, value = rightmost non-zero value
    For each column: period = GCD of non-zero row positions, value = bottommost non-zero value
    Patterns fill from position 0 in steps of period, up to the extent.
    """
    grid = np.array(grid)
    result = np.zeros_like(grid)
    rows, cols = grid.shape
    
    # Process each row
    for r in range(rows):
        nz = [c for c in range(cols) if grid[r, c] != 0]
        if not nz:
            continue
        period = reduce(gcd, nz) if nz else 1
        if period == 0:
            period = 1
        value = grid[r, nz[-1]]
        max_pos = max(nz)
        # Fill from 0 to max_pos (rounded up to next multiple of period)
        extent = ((max_pos + period - 1) // period) * period
        for c in range(0, min(extent, cols - 1) + 1, period):
            result[r, c] = value
    
    # Process each column (overwrite where result is 0)
    for c in range(cols):
        nz = [r for r in range(rows) if grid[r, c] != 0]
        if not nz:
            continue
        period = reduce(gcd, nz) if nz else 1
        if period == 0:
            period = 1
        value = grid[nz[-1], c]
        max_pos = max(nz)
        extent = ((max_pos + period - 1) // period) * period
        for r in range(0, min(extent, rows - 1) + 1, period):
            if result[r, c] == 0:
                result[r, c] = value
    
    return result