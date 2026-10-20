import numpy as np
from scipy import ndimage

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid by tiling one object (the 'tile') according to
    the pattern of another object (the 'mask').
    
    Algorithm:
    1. Find the background color (most common color).
    2. Find the two objects using 8-connected component labeling.
    3. The 'tile' is the object with higher density (fewer background pixels in bbox).
    4. The 'mask' is the other object, whose non-background pattern determines placement.
    5. Output size: (mask_rows * tile_rows) x (mask_cols * tile_cols)
    6. For each 'on' cell in the mask, place the tile; otherwise fill with background.
    """
    # Step 1: Determine background color (most common color in the grid)
    colors, counts = np.unique(grid, return_counts=True)
    bg_color = colors[np.argmax(counts)]
    
    # Step 2: Find connected components of non-background pixels (8-connectivity)
    non_bg = (grid != bg_color).astype(int)
    structure = np.ones((3, 3), dtype=int)  # 8-connected neighborhood
    labeled, num_features = ndimage.label(non_bg, structure=structure)
    
    # Step 3: Get bounding boxes and compute density for each component
    objects = []
    for i in range(1, num_features + 1):
        component = (labeled == i)
        rows = np.where(component.any(axis=1))[0]
        cols = np.where(component.any(axis=0))[0]
        r_min, r_max = rows[0], rows[-1]
        c_min, c_max = cols[0], cols[-1]
        bbox = grid[r_min:r_max+1, c_min:c_max+1].copy()
        bg_count = np.sum(bbox == bg_color)
        total = bbox.shape[0] * bbox.shape[1]
        objects.append({
            'bbox': bbox,
            'density': (total - bg_count) / total
        })
    
    # Step 4: Identify tile (higher density = more solid) and mask (lower density)
    if objects[0]['density'] >= objects[1]['density']:
        tile = objects[0]['bbox']
        mask_obj = objects[1]['bbox']
    else:
        tile = objects[1]['bbox']
        mask_obj = objects[0]['bbox']
    
    # Step 5: Create the binary pattern from the mask object
    pattern = (mask_obj != bg_color).astype(int)
    
    # Step 6: Build the output grid by tiling
    tile_h, tile_w = tile.shape
    mask_h, mask_w = pattern.shape
    output = np.full((mask_h * tile_h, mask_w * tile_w), bg_color, dtype=grid.dtype)
    
    for i in range(mask_h):
        for j in range(mask_w):
            if pattern[i, j]:
                output[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w] = tile
    
    return output