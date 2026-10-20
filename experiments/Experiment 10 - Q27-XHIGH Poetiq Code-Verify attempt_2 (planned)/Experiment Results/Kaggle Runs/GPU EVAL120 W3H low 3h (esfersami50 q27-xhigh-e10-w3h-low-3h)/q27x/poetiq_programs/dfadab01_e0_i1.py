import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Place fixed shapes at marker positions.
    
    Markers are single-pixel connected components (isolated non-zero pixels).
    Each marker color maps to a specific 4x4 shape:
      - Color 2: hollow square (color 4)
      - Color 3: corner-missing square (color 1)
      - Color 5: two diagonal 2x2 blocks (color 6)
      - Color 8: diamond (color 7)
    
    Shapes are placed with the marker at the top-left corner.
    Processing order: bottom-right to top-left.
    A shape is skipped if it goes out of bounds or overlaps an already-placed shape.
    """
    rows, cols = grid.shape
    output = np.zeros_like(grid)
    
    # Define the 4x4 shapes for each marker color
    shapes = {
        2: np.array([[4,4,4,4],[4,0,0,4],[4,0,0,4],[4,4,4,4]]),
        3: np.array([[0,1,1,0],[1,0,0,1],[1,0,0,1],[0,1,1,0]]),
        5: np.array([[6,6,0,0],[6,6,0,0],[0,0,6,6],[0,0,6,6]]),
        8: np.array([[7,0,0,7],[0,7,7,0],[0,7,7,0],[7,0,0,7]]),
    }
    
    # Find all single-pixel connected components (markers)
    visited = np.zeros_like(grid, dtype=bool)
    markers = []  # (row, col, color)
    
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0 and not visited[r, c]:
                color = grid[r, c]
                component = []
                queue = [(r, c)]
                visited[r, c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    component.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and grid[nr, nc] == color:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                if len(component) == 1:
                    markers.append((r, c, color))
    
    # Sort from bottom-right to top-left
    markers.sort(key=lambda m: (m[0], m[1]), reverse=True)
    
    # Place shapes, tracking bounding boxes for overlap detection
    placed_rects = []
    
    for r, c, color in markers:
        if color not in shapes:
            continue
        
        shape = shapes[color]
        sh, sc = shape.shape
        
        # Check bounds
        if r + sh > rows or c + sc > cols:
            continue
        
        # Check overlap with already-placed shapes
        r1, c1 = r, c
        r2, c2 = r + sh - 1, c + sc - 1
        overlap = False
        for pr1, pc1, pr2, pc2 in placed_rects:
            if r1 <= pr2 and r2 >= pr1 and c1 <= pc2 and c2 >= pc1:
                overlap = True
                break
        
        if overlap:
            continue
        
        # Place the shape
        output[r:r+sh, c:c+sc] = shape
        placed_rects.append((r1, c1, r2, c2))
    
    return output