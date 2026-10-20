import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform an ARC grid by:
    1. Taking the 5x5 top-left sprite
    2. Swapping its two active colors to create a stamp
    3. Taking the right part of the input (after 6-column separator)
    4. Stamping the swapped pattern at 3x3 scale wherever the template blob (color 8) exists
    5. Preserving background colors and special content (like color 1)
    """
    if grid.size == 0:
        return grid.copy()
    
    rows, cols = grid.shape
    
    # Extract the 5x5 sprite from top-left
    sprite = grid[:5, :5].copy()
    
    # Find the two active colors in the sprite (ignore 0 and 6)
    active_colors = {}
    for i in range(5):
        for j in range(5):
            val = sprite[i, j]
            if val not in (0, 6):
                active_colors[val] = active_colors.get(val, 0) + 1
    
    if len(active_colors) != 2:
        # Fallback: just return the right part
        return grid[:, 6:].copy()
    
    # Get the two active colors
    color_a, color_b = sorted(active_colors.keys())
    
    # Create the stamp by swapping the two colors in the 3x3 core
    stamp = np.zeros((3, 3), dtype=grid.dtype)
    for i in range(3):
        for j in range(3):
            val = sprite[1 + i, 1 + j]
            if val == color_a:
                stamp[i, j] = color_b
            elif val == color_b:
                stamp[i, j] = color_a
    
    # Take the right part of the input (after the 6-column separator)
    # The 6-column separator is typically at column 5
    canvas = grid[:, 6:].copy()
    
    # Identify the template blob (color 8) in the canvas
    blob_mask = (canvas == 8)
    
    # Stamp the pattern at 3x3 scale wherever the blob exists
    for i in range(canvas.shape[0]):
        for j in range(canvas.shape[1]):
            if blob_mask[i, j]:
                # Calculate which 3x3 cell this pixel belongs to
                cell_row = i // 3
                cell_col = j // 3
                # Calculate position within the 3x3 cell
                pos_row = i % 3
                pos_col = j % 3
                # Stamp the pattern
                canvas[i, j] = stamp[pos_row, pos_col]
    
    # Preserve special content colors (like 1) that should override the stamp
    # Look for cells with color 1 in the original canvas (before stamping)
    original_canvas = grid[:, 6:].copy()
    for i in range(canvas.shape[0]):
        for j in range(canvas.shape[1]):
            if original_canvas[i, j] == 1:
                canvas[i, j] = 1
    
    return canvas