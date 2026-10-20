import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid by:
    1. Dividing into 4 equal quadrants (TL, TR, BL, BR)
    2. Creating a pattern from BR with 0s filled from corresponding BL positions
    3. Determining output height from TL non-zero cell count
    4. Determining output width from max(TR non-zero count, BR non-zero column count)
    5. Tiling the pattern to fill the output dimensions and cropping
    """
    h, w = grid.shape
    mid_h, mid_w = h // 2, w // 2
    
    # Extract quadrants
    tl = grid[:mid_h, :mid_w]
    tr = grid[:mid_h, mid_w:]
    bl = grid[mid_h:, :mid_w]
    br = grid[mid_h:, mid_w:]
    
    # Determine output dimensions
    output_h = int(np.count_nonzero(tl))
    br_nonzero_cols = int(np.sum(np.any(br != 0, axis=0)))
    output_w = max(int(np.count_nonzero(tr)), br_nonzero_cols)
    
    # Handle edge case of empty output
    if output_h == 0 or output_w == 0:
        return np.zeros((0, 0), dtype=grid.dtype)
    
    # Create the pattern: BR with 0s filled from corresponding BL positions
    pattern = br.copy()
    zero_mask = (pattern == 0)
    pattern[zero_mask] = bl[zero_mask]
    
    # Tile the pattern to fill output_h x output_w
    ph, pw = pattern.shape
    tiles_h = (output_h + ph - 1) // ph
    tiles_w = (output_w + pw - 1) // pw
    
    tiled = np.tile(pattern, (tiles_h, tiles_w))
    
    # Crop to output size
    result = tiled[:output_h, :output_w]
    
    return result