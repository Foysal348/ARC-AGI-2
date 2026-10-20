import numpy as np
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by applying gravity in the direction indicated by grid[0][0].
    Objects (connected components of same color) fall in that direction until they
    hit the boundary or another settled object.
    
    Direction mapping (numpad): 2=down, 4=left, 6=right, 8=up,
    1=down-left, 3=down-right, 7=up-left, 9=up-right, 5=center(no move)
    """
    grid = grid.copy()
    rows, cols = grid.shape
    
    # Direction vectors based on numpad
    direction = grid[0, 0]
    dir_map = {
        1: (1, -1), 2: (1, 0), 3: (1, 1),
        4: (0, -1), 5: (0, 0), 6: (0, 1),
        7: (-1, -1), 8: (-1, 0), 9: (-1, 1)
    }
    dr, dc = dir_map.get(direction, (0, 0))
    
    if dr == 0 and dc == 0:
        return grid
    
    # Find all connected components (4-connectivity) excluding [0][0]
    visited = np.zeros((rows, cols), dtype=bool)
    visited[0, 0] = True  # Skip the direction indicator
    
    components = []
    
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0 and not visited[r, c]:
                # BFS to find connected component
                color = grid[r, c]
                component = []
                queue = [(r, c)]
                visited[r, c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    component.append((cr, cc))
                    for dr2, dc2 in [(0,1),(0,-1),(1,0),(-1,0)]:
                        nr, nc = cr + dr2, cc + dc2
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if not visited[nr, nc] and grid[nr, nc] == color:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                components.append((color, component))
    
    # For each component, determine its "position" along the gravity axis
    # and process from the boundary inward
    
    def component_position(comp_cells, dr, dc):
        """Get the position of a component along the gravity direction."""
        if dr != 0:
            positions = [r for r, c in comp_cells]
            return max(positions) if dr > 0 else min(positions)
        else:
            positions = [c for r, c in comp_cells]
            return max(positions) if dc > 0 else min(positions)
    
    # Sort components: those closest to the boundary (in gravity direction) first
    if dr > 0:  # falling down: process bottom-most first
        components.sort(key=lambda x: -component_position(x[1], dr, dc))
    elif dr < 0:  # falling up: process top-most first
        components.sort(key=lambda x: component_position(x[1], dr, dc))
    elif dc > 0:  # falling right: process right-most first
        components.sort(key=lambda x: -component_position(x[1], dr, dc))
    elif dc < 0:  # falling left: process left-most first
        components.sort(key=lambda x: component_position(x[1], dr, dc))
    
    # Create output grid
    output = np.zeros((rows, cols), dtype=int)
    output[0, 0] = grid[0, 0]
    
    settled = np.zeros((rows, cols), dtype=bool)
    settled[0, 0] = True
    
    for color, comp_cells in components:
        # Try to slide the component in the gravity direction
        # Find maximum slide amount
        max_slide = 0
        for slide in range(0, max(rows, cols)):
            # Check if component can be at offset 'slide'
            valid = True
            for (r, c) in comp_cells:
                nr, nc = r + dr * slide, c + dc * slide
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    valid = False
                    break
                if settled[nr, nc]:
                    valid = False
                    break
            if not valid:
                break
            max_slide = slide
        
        # Place component at max_slide position
        for (r, c) in comp_cells:
            nr, nc = r + dr * max_slide, c + dc * max_slide
            if 0 <= nr < rows and 0 <= nc < cols:
                output[nr, nc] = color
                settled[nr, nc] = True
    
    return output