import numpy as np
from scipy import ndimage

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Find the largest connected component (8-connectivity) of non-zero pixels,
    then crop the grid to the bounding box of that component.
    
    This effectively removes 'stray' or 'noise' pixels that are far from
    the main structure, keeping only the main object and its interior.
    """
    grid = grid.copy()
    
    # Create a binary mask of non-zero pixels
    mask = (grid != 0).astype(np.uint8)
    
    # Label connected components using 8-connectivity
    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, num_features = ndimage.label(mask, structure=structure)
    
    if num_features == 0:
        return grid
    
    # Find the largest connected component (by pixel count)
    component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_component = np.argmax(component_sizes) + 1
    
    # Get the bounding box of the largest component
    component_mask = (labeled == largest_component)
    rows = np.any(component_mask, axis=1)
    cols = np.any(component_mask, axis=0)
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Crop to the bounding box
    return grid[rmin:rmax+1, cmin:cmax+1]