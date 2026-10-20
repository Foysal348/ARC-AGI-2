import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by painting 7s around 1-structures.
    
    Rules:
    1. Keep all 1s and 6s.
    2. Find 4-connected regions of 8s.
    3. For enclosed regions (not touching boundary): fill all with 7.
    4. For boundary-touching regions: paint cells within Chebyshev dist 1 of any 1.
    """
    rows, cols = grid.shape
    result = grid.copy()
    
    # Create mask of 8-cells
    is_eight = (grid == 8)
    
    # Find 4-connected components of 8s
    visited = np.zeros((rows, cols), dtype=bool)
    regions = []  # list of (cells, touches_boundary)
    
    for r in range(rows):
        for c in range(cols):
            if is_eight[r, c] and not visited[r, c]:
                # BFS to find the component
                cells = []
                queue = deque([(r, c)])
                visited[r, c] = True
                touches_boundary = False
                while queue:
                    cr, cc = queue.popleft()
                    cells.append((cr, cc))
                    if cr == 0 or cr == rows-1 or cc == 0 or cc == cols-1:
                        touches_boundary = True
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if is_eight[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                regions.append((cells, touches_boundary))
    
    # Get all 1-cell positions
    one_positions = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1:
                one_positions.add((r, c))
    
    # Apply rules to each region
    for cells, touches_boundary in regions:
        if not touches_boundary:
            # Enclosed region: fill all with 7
            for (r, c) in cells:
                result[r, c] = 7
        else:
            # Boundary-touching: paint cells within Chebyshev dist 1 of any 1
            for (r, c) in cells:
                # Check 8-adjacency to any 1
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if (r+dr, c+dc) in one_positions:
                            result[r, c] = 7
                            break
                    if result[r, c] == 7:
                        break
    
    return result