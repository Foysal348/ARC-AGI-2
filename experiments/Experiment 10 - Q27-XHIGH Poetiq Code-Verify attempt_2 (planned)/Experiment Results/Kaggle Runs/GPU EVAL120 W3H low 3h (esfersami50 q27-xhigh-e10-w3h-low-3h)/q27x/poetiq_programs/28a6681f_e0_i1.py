import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill U-shaped container interiors with 1s and remove 1s not connected to containers.
    
    A cell is 'inside' a container of a given color if:
    - There exists a cell of that color below in the same column
    - There exists a cell of that color to the left in the same row
    - There exists a cell of that color to the right in the same row or the row below
    """
    grid = grid.copy()
    rows, cols = grid.shape
    
    # Find all non-zero, non-1 colors
    colors = set(grid[grid != 0].flatten().tolist()) - {1}
    
    # Find all cells that are inside container interiors
    interior = np.zeros((rows, cols), dtype=bool)
    
    for color in colors:
        mask = (grid == color)
        for r in range(rows):
            for c in range(cols):
                if mask[r, c]:
                    continue
                # Check: color cell below in same column
                below = mask[r+1:, c].any() if r + 1 < rows else False
                if not below:
                    continue
                # Check: color cell to the left in same row
                left = mask[r, :c].any() if c > 0 else False
                if not left:
                    continue
                # Check: color cell to the right in same row or row below
                right = False
                if c < cols and mask[r, c:].any():
                    right = True
                if not right and r + 1 < rows and c < cols and mask[r+1, c:].any():
                    right = True
                if right:
                    interior[r, c] = True
    
    # Find all input 1-cells
    ones_mask = (grid == 1)
    
    # Find connected component of 1s and interior cells
    # The output 1s = interior cells + input 1-cells that are 4-connected to interior
    visited = np.zeros((rows, cols), dtype=bool)
    output_ones = np.zeros((rows, cols), dtype=bool)
    
    # BFS from all interior cells, expanding through cells that are interior OR input 1s
    queue = deque()
    for r in range(rows):
        for c in range(cols):
            if interior[r, c]:
                queue.append((r, c))
                visited[r, c] = True
                output_ones[r, c] = True
    
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                if interior[nr, nc] or ones_mask[nr, nc]:
                    visited[nr, nc] = True
                    output_ones[nr, nc] = True
                    queue.append((nr, nc))
    
    # Build output: same as input but 1s replaced by output_ones
    result = grid.copy()
    # Remove all 1s first
    result[ones_mask] = 0
    # Set output 1s
    result[output_ones] = 1
    
    return result