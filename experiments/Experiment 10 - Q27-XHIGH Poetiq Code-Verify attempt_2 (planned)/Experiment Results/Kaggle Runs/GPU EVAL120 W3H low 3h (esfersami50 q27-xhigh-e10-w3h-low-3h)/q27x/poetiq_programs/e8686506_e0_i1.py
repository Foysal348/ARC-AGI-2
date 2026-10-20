import numpy as np
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid by:
    1. Finding the background color (most common)
    2. Identifying all 8-connected components of non-background colors
    3. Determining output grid size from the largest component's bounding box
    4. Placing each component at its unique valid position
    5. Filling remaining empty cells with isolated pixels
    """
    # Step 1: Find background color (most common)
    colors, counts = np.unique(grid, return_counts=True)
    bg_color = colors[np.argmax(counts)]
    
    # Step 2: Find all 8-connected components of non-background pixels
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    components = []  # List of (color, list of (r,c) positions)
    
    def get_component(start_r, start_c, color):
        """BFS to find 8-connected component."""
        component = []
        queue = [(start_r, start_c)]
        visited[start_r, start_c] = True
        while queue:
            r, c = queue.pop(0)
            component.append((r, c))
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and grid[nr, nc] == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
        return component
    
    for r in range(rows):
        for c in range(cols):
            if not visited[r, c] and grid[r, c] != bg_color:
                comp = get_component(r, c, grid[r, c])
                components.append((grid[r, c], comp))
    
    # Step 3: Determine output grid size from the largest component's bounding box
    def get_bbox(comp):
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        return min(rs), min(cs), max(rs), max(cs)
    
    # Find the component with the largest bounding box area
    best_comp = None
    best_area = 0
    for color, comp in components:
        r1, c1, r2, c2 = get_bbox(comp)
        area = (r2 - r1 + 1) * (c2 - c1 + 1)
        if area > best_area:
            best_area = area
            best_comp = (color, comp)
    
    # Output grid dimensions from the largest component's bounding box
    r1, c1, r2, c2 = get_bbox(best_comp[1])
    out_h = r2 - r1 + 1
    out_w = c2 - c1 + 1
    
    # Create output grid (0 = empty)
    out = np.zeros((out_h, out_w), dtype=int)
    
    # Step 4: Place the largest component at (0,0)
    color, comp = best_comp
    for (r, c) in comp:
        out[r - r1, c - c1] = color
    
    # Place remaining components sorted by size (largest first)
    other_comps = [(color, comp) for color, comp in components if comp != best_comp[1]]
    other_comps.sort(key=lambda x: len(x[1]), reverse=True)
    
    def get_shape(comp):
        """Get the shape pattern as a set of relative (dr, dc) positions."""
        r_min = min(p[0] for p in comp)
        c_min = min(p[1] for p in comp)
        return frozenset((p[0] - r_min, p[1] - c_min) for p in comp)
    
    def can_place(out, shape, color, row, col):
        """Check if shape can be placed at (row, col) - all cells must be empty."""
        for (dr, dc) in shape:
            r, c = row + dr, col + dc
            if r < 0 or r >= out.shape[0] or c < 0 or c >= out.shape[1]:
                return False
            if out[r, c] != 0:
                return False
        return True
    
    # For each remaining component, find its unique valid position
    placed = []
    for color, comp in other_comps:
        shape = get_shape(comp)
        valid_positions = []
        for row in range(out_h):
            for col in range(out_w):
                if can_place(out, shape, color, row, col):
                    valid_positions.append((row, col))
        
        if len(valid_positions) == 1:
            row, col = valid_positions[0]
            for (dr, dc) in shape:
                out[row + dr, col + dc] = color
            placed.append((color, comp))
        elif len(valid_positions) > 1:
            # For multiple valid positions, we need to handle it
            # This can happen for symmetric shapes (e.g., two identical 4x1 lines)
            # We'll handle this by trying all combinations
            pass
    
    # Handle components that had multiple valid positions
    # Process remaining unplaced components
    unplaced = [(color, comp) for color, comp in other_comps if comp not in [p[1] for p in placed]]
    
    def place_components(out, unplaced):
        """Recursively place components, trying all valid positions."""
        if not unplaced:
            return True
        color, comp = unplaced[0]
        shape = get_shape(comp)
        for row in range(out_h):
            for col in range(out_w):
                if can_place(out, shape, color, row, col):
                    # Place it
                    for (dr, dc) in shape:
                        out[row + dr, col + dc] = color
                    if place_components(out, unplaced[1:]):
                        return True
                    # Undo
                    for (dr, dc) in shape:
                        out[row + dr, col + dc] = 0
        return False
    
    if unplaced:
        place_components(out, unplaced)
    
    # Step 5: Fill remaining empty cells with isolated pixels (1x1 components)
    remaining_empty = [(r, c) for r in range(out_h) for c in range(out_w) if out[r, c] == 0]
    
    # Collect all unplaced pixels (from 1x1 components or any leftover)
    placed_pixels = set()
    for color, comp in components:
        for (r, c) in comp:
            placed_pixels.add((r, c))
    
    # Find which components were actually placed in the output
    # Fill remaining with the appropriate color
    # For isolated pixels, just fill remaining empty cells
    # Determine which color to use based on the count matching
    filler_pixels = []
    for color, comp in components:
        if len(comp) == 1:  # Isolated pixel
            filler_pixels.append(color)
    
    # If there are remaining empty cells and filler pixels, fill them
    if remaining_empty and filler_pixels:
        # Use the most common filler color, or assign based on count
        if len(remaining_empty) == len(filler_pixels):
            for i, (r, c) in enumerate(remaining_empty):
                out[r, c] = filler_pixels[i]
        elif len(filler_pixels) > 0:
            fill_color = filler_pixels[0]
            for (r, c) in remaining_empty:
                out[r, c] = fill_color
    
    return out