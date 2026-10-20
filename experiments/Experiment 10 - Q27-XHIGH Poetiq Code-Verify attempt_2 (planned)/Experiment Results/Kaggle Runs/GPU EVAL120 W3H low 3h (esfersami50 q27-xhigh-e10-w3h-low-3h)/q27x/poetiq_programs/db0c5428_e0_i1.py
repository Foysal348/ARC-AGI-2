import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by extending the 9x9 pattern outward using 180-degree
    rotational symmetry around the pattern center.
    
    For each pixel outside the original pattern, compute its point-reflection
    across the pattern center, map it back into the pattern via modular
    arithmetic, and copy the value. Only certain 3x3 block positions are filled
    (those where block_row+block_col is even, or on the center row/column).
    """
    # Find background color from border
    border = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
    bg_color = np.bincount(border).argmax()
    
    # Find bounding box of non-background region
    mask = (grid != bg_color)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0 = int(np.argmax(rows))
    r1 = int(len(rows) - np.argmax(rows[::-1]) - 1)
    c0 = int(np.argmax(cols))
    c1 = int(len(cols) - np.argmax(cols[::-1]) - 1)
    
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    
    # Center of the pattern
    cr = (r0 + r1) / 2.0
    cc = (c0 + c1) / 2.0
    
    # Block size (one tile)
    block_size = h // 3  # = 3 for 9x9 pattern
    
    # Block grid origin
    block_row_start = r0 - block_size
    block_col_start = c0 - block_size
    
    # Create output
    output = np.full_like(grid, bg_color)
    
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            # If within original pattern, keep value
            if r0 <= r <= r1 and c0 <= c <= c1:
                output[r, c] = grid[r, c]
            else:
                # Determine block position in the 5x5 block grid
                br = (r - block_row_start) // block_size
                bc = (c - block_col_start) // block_size
                
                # Must be within the 5x5 block grid
                if not (0 <= br <= 4 and 0 <= bc <= 4):
                    continue
                
                # Skip unfilled blocks: (br+bc) odd AND not on center row/col
                if (br + bc) % 2 == 1 and br != 2 and bc != 2:
                    continue
                
                # Reflect pixel across pattern center
                reflected_r = 2.0 * cr - r
                reflected_c = 2.0 * cc - c
                
                # Map reflected position into pattern via modular arithmetic
                mapped_r = int((reflected_r - r0) % h) + r0
                mapped_c = int((reflected_c - c0) % w) + c0
                
                output[r, c] = grid[mapped_r, mapped_c]
    
    return output