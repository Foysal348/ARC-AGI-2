import numpy as np
from scipy import ndimage

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by moving non-background objects to top or bottom edges.
    
    Objects whose vertical center is above the grid midpoint are moved to the bottom
    (bottom-aligned). Objects whose vertical center is at or below the grid midpoint
    are moved to the top (top-aligned). Horizontal positions and shapes are preserved.
    """
    if grid.size == 0:
        return grid.copy()
    
    rows, cols = grid.shape
    
    # Find background color (most common color)
    flat = grid.flatten()
    colors, counts = np.unique(flat, return_counts=True)
    bg_color = colors[np.argmax(counts)]
    
    # Create mask of non-background pixels
    mask = (grid != bg_color)
    
    # Find connected components (4-connectivity)
    labeled, num_features = ndimage.label(mask, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    
    if num_features == 0:
        return np.full_like(grid, bg_color)
    
    # Create output grid filled with background
    output = np.full_like(grid, bg_color)
    
    midpoint = rows / 2.0
    
    for i in range(1, num_features + 1):
        component_mask = (labeled == i)
        pixel_coords = np.where(component_mask)
        
        if len(pixel_coords[0]) == 0:
            continue
        
        min_row = pixel_coords[0].min()
        max_row = pixel_coords[0].max()
        min_col = pixel_coords[1].min()
        max_col = pixel_coords[1].max()
        
        comp_height = max_row - min_row + 1
        comp_width = max_col - min_col + 1
        
        # Extract the component pattern
        component = grid[min_row:max_row+1, min_col:max_col+1].copy()
        
        # Determine vertical center
        center_row = (min_row + max_row) / 2.0
        
        # Above midpoint → bottom-aligned, at or below → top-aligned
        if center_row < midpoint:
            new_min_row = rows - comp_height
        else:
            new_min_row = 0
        
        new_max_row = new_min_row + comp_height - 1
        new_min_col = min_col
        new_max_col = min_col + comp_width - 1
        
        output[new_min_row:new_max_row+1, new_min_col:new_max_col+1] = component
    
    return output