import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by:
    1. Identifying hollow rectangles (borders) and small objects (other components).
    2. Removing all small objects.
    3. For each rectangle, counting connected components of the same color outside
       the rectangle, and placing that many pixels in the center row of the interior
       at column offsets 7, 5, 3, 1, ... from the left interior edge.
    """
    rows, cols = grid.shape
    
    # Find background color (most common)
    colors, counts = np.unique(grid, return_counts=True)
    bg_color = colors[np.argmax(counts)]
    
    # Find connected components of non-background pixels using BFS
    visited = np.zeros((rows, cols), dtype=bool)
    components = []  # List of (color, set_of_positions)
    
    for r in range(rows):
        for c in range(cols):
            if not visited[r, c] and grid[r, c] != bg_color:
                color = grid[r, c]
                component = set()
                queue = deque([(r, c)])
                visited[r, c] = True
                while queue:
                    cr, cc = queue.popleft()
                    component.add((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if not visited[nr, nc] and grid[nr, nc] == color:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                components.append((color, component))
    
    # Classify components as rectangles or small objects
    rectangles = []   # (color, set_of_positions, (min_r, max_r, min_c, max_c))
    small_objects = []  # (color, set_of_positions)
    
    for color, component in components:
        min_r = min(r for r, c in component)
        max_r = max(r for r, c in component)
        min_c = min(c for r, c in component)
        max_c = max(c for r, c in component)
        
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        
        # Must be at least 3x3 to be a valid hollow rectangle
        if height < 3 or width < 3:
            small_objects.append((color, component))
            continue
        
        # Check if it forms the complete border of its bounding box
        border_pixels = set()
        for c in range(min_c, max_c + 1):
            border_pixels.add((min_r, c))
            border_pixels.add((max_r, c))
        for r in range(min_r, max_r + 1):
            border_pixels.add((r, min_c))
            border_pixels.add((r, max_c))
        
        if component == border_pixels:
            rectangles.append((color, component, (min_r, max_r, min_c, max_c)))
        else:
            small_objects.append((color, component))
    
    # Create output grid filled with background
    output = np.full((rows, cols), bg_color, dtype=grid.dtype)
    
    # Draw rectangle borders
    for color, component, bbox in rectangles:
        for r, c in component:
            output[r, c] = color
    
    # For each rectangle, place interior pixels based on count of matching small objects
    for color, component, (min_r, max_r, min_c, max_c) in rectangles:
        count = sum(1 for obj_color, _ in small_objects if obj_color == color)
        
        if count == 0:
            continue
        
        # Interior bounds
        int_min_r = min_r + 1
        int_max_r = max_r - 1
        int_min_c = min_c + 1
        int_max_c = max_c - 1
        
        # Center row of the interior
        center_r = (int_min_r + int_max_r) // 2
        
        # Place pixels at column offsets 7, 5, 3, 1, ... from left interior edge
        for i in range(count):
            col_offset = 7 - 2 * i
            if col_offset < 0:
                break
            c = int_min_c + col_offset
            if int_min_c <= c <= int_max_c:
                output[center_r, c] = color
    
    return output