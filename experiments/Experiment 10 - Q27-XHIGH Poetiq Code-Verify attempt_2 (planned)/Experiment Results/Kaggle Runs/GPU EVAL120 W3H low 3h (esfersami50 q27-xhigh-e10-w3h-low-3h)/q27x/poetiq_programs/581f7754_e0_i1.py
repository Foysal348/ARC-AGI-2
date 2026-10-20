import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """Move shapes so their centers align with an isolated center on the axis of least spread."""
    rows, cols = grid.shape
    result = np.full((rows, cols), -1, dtype=int)  # -1 means "copy from input"
    
    # Find background color
    colors, counts = np.unique(grid, return_counts=True)
    bg_color = colors[np.argmax(counts)]
    
    # Find all non-background colors
    non_bg_colors = [c for c in colors if c != bg_color]
    
    # Get 4-connected component
    def get_component(start, color):
        component = set()
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            if (r, c) in component or not (0 <= r < rows and 0 <= c < cols):
                continue
            if grid[r, c] != color:
                continue
            component.add((r, c))
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = r+dr, c+dc
                if (nr, nc) not in component:
                    queue.append((nr, nc))
        return component
    
    # Find all connected components for each non-bg color
    all_components = {}  # color -> list of (component_set, is_assigned)
    visited = set()
    for c in non_bg_colors:
        comps = []
        for r in range(rows):
            for col in range(cols):
                if grid[r, col] == c and (r, col) not in visited:
                    comp = get_component((r, col), c)
                    visited.update(comp)
                    comps.append(comp)
        all_components[c] = comps
    
    # For each component, check if it encloses a pixel of another color
    # A pixel is "enclosed" if flood fill from it (through non-shape pixels) can't reach boundary
    def is_enclosed(pixel, shape_color):
        """Check if pixel is enclosed by shape_color pixels."""
        r, c = pixel
        # Flood fill from pixel through non-shape-color pixels
        visited_f = set()
        queue = deque([pixel])
        while queue:
            cr, cc = queue.popleft()
            if (cr, cc) in visited_f:
                continue
            if not (0 <= cr < rows and 0 <= cc < cols):
                return False  # reached boundary
            visited_f.add((cr, cc))
            if grid[cr, cc] == shape_color:
                continue  # can't pass through shape
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = cr+dr, c+dc
                if (nr, nc) not in visited_f:
                    queue.append((nr, nc))
        return True  # didn't reach boundary
    
    # Find shapes: (component_pixels, center_position, shape_color, center_color)
    shapes = []
    center_pixels_in_shapes = set()
    
    for shape_color in non_bg_colors:
        for comp in all_components[shape_color]:
            # Find all non-bg pixels adjacent to this component that aren't part of it
            candidate_centers = set()
            for (r, c) in comp:
                for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if grid[nr, nc] != bg_color and grid[nr, nc] != shape_color and (nr, nc) not in comp:
                            candidate_centers.add((nr, nc))
            
            # Check which candidates are enclosed
            for pixel in candidate_centers:
                if is_enclosed(pixel, shape_color):
                    shapes.append((comp, pixel, shape_color, grid[pixel]))
                    center_pixels_in_shapes.add(pixel)
                    break  # One center per shape
    
    # Find isolated centers (center-colored pixels not in any shape)
    # The center color is the color that appears as centers in shapes
    center_colors = set(s[3] for s in shapes)
    
    isolated_center = None
    for cc in center_colors:
        # Find all pixels of this color
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == cc and (r, c) not in center_pixels_in_shapes:
                    # Check if it's truly isolated (not adjacent to many same-color pixels forming a shape)
                    isolated_center = (r, c)
                    break
            if isolated_center:
                break
        if isolated_center:
            break
    
    if not isolated_center or not shapes:
        return grid.copy()
    
    # Determine alignment axis
    center_positions = [s[1] for s in shapes]
    row_spread = max(p[0] for p in center_positions) - min(p[0] for p in center_positions)
    col_spread = max(p[1] for p in center_positions) - min(p[1] for p in center_positions)
    
    if col_spread <= row_spread:
        align_axis = 'col'
        target = isolated_center[1]
    else:
        align_axis = 'row'
        target = isolated_center[0]
    
    # Move shapes
    # First, mark all shape pixels and center pixels to remove from input
    shape_pixel_mask = np.zeros((rows, cols), dtype=bool)
    for comp, center, sc, cc in shapes:
        for (r, c) in comp:
            shape_pixel_mask[r, c] = True
        shape_pixel_mask[center[0], center[1]] = True
    
    # Start with background
    result = np.full((rows, cols), bg_color, dtype=int)
    
    # Copy non-shape pixels from input
    for r in range(rows):
        for c in range(cols):
            if not shape_pixel_mask[r, c]:
                result[r, c] = grid[r, c]
    
    # Place moved shapes
    for comp, center, sc, cc in shapes:
        if align_axis == 'col':
            shift = target - center[1]
            new_center = (center[0], target)
        else:
            shift = target - center[0]
            new_center = (target, center[1])
        
        # Calculate offset from center to each pixel
        for (r, c) in comp:
            nr = r + (new_center[0] - center[0])
            nc = c + (new_center[1] - center[1])
            if 0 <= nr < rows and 0 <= nc < cols:
                result[nr, nc] = sc
        
        # Place center
        if 0 <= new_center[0] < rows and 0 <= new_center[1] < cols:
            result[new_center[0], new_center[1]] = cc
    
    return result