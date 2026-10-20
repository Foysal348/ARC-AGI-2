import numpy as np
from scipy import ndimage

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform input grid by:
    1. Finding all objects (connected components of non-zero pixels)
    2. Classifying each as hollow (border with empty interior) or filled (solid)
    3. Arranging hollow objects in the left column, filled in the right column
    4. Within each column, objects are sorted top-to-bottom by position
    5. Empty slots are filled with zeros
    """
    # Step 1: Find connected components of non-zero pixels
    mask = (grid != 0).astype(int)
    labeled, num_objects = ndimage.label(mask)
    
    objects = []
    for i in range(1, num_objects + 1):
        rows = np.where(labeled == i)[0]
        cols = np.where(labeled == i)[1]
        top, bottom = rows.min(), rows.max()
        left, right = cols.min(), cols.max()
        
        color = grid[top, left]
        height = bottom - top + 1
        width = right - left + 1
        
        # Determine if hollow or filled by checking the interior
        is_hollow = False
        if height > 2 and width > 2:
            inner = grid[top + 1:bottom, left + 1:right]
            if np.all(inner == 0):
                is_hollow = True
        
        obj_grid = grid[top:bottom + 1, left:right + 1].copy()
        
        objects.append({
            'top': top,
            'hollow': is_hollow,
            'height': height,
            'width': width,
            'grid': obj_grid
        })
    
    # Step 2: Separate into hollow and filled, sort by vertical position
    hollow_objects = sorted([o for o in objects if o['hollow']], key=lambda o: o['top'])
    filled_objects = sorted([o for o in objects if not o['hollow']], key=lambda o: o['top'])
    
    # Step 3: Build output grid
    num_rows = max(len(hollow_objects), len(filled_objects))
    
    if num_rows == 0:
        return np.zeros((1, 1), dtype=grid.dtype)
    
    all_objects = hollow_objects + filled_objects
    obj_height = all_objects[0]['height']
    obj_width = all_objects[0]['width']
    
    out_height = num_rows * obj_height
    out_width = 2 * obj_width
    
    output = np.zeros((out_height, out_width), dtype=grid.dtype)
    
    # Place hollow objects in the left column (sorted top to bottom)
    for i, obj in enumerate(hollow_objects):
        r_start = i * obj_height
        output[r_start:r_start + obj_height, 0:obj_width] = obj['grid']
    
    # Place filled objects in the right column (sorted top to bottom)
    for i, obj in enumerate(filled_objects):
        r_start = i * obj_height
        output[r_start:r_start + obj_height, obj_width:2 * obj_width] = obj['grid']
    
    return output