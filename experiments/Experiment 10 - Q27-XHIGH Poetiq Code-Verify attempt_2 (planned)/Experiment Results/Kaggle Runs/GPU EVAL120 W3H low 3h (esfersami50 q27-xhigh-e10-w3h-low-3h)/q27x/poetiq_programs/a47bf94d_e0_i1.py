import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform solid 3x3 blocks into cross patterns.
    
    Steps:
    1. Find all solid 3x3 blocks (uniform non-zero, non-8/9/5 color)
    2. Convert them to cross patterns (keep edge midpoints only)
    3. Trace paths through the 8-maze to place corresponding crosses at the bottom
    """
    result = grid.copy()
    rows, cols = grid.shape
    
    # Step 1: Find and convert solid 3x3 blocks to crosses
    solid_blocks = []  # (top_row, left_col, color)
    
    for r in range(rows - 2):
        for c in range(cols - 2):
            # Check if this is a 3x3 solid block
            block = grid[r:r+3, c:c+3]
            # All cells must be the same non-zero color
            val = block[0, 0]
            if val == 0 or val in (8, 9, 5):
                continue
            if np.all(block == val):
                # Make sure it's not part of a larger solid region
                # Check that surrounding cells aren't the same color (to ensure it's exactly 3x3)
                is_exact = True
                # Check above
                if r > 0 and np.all(grid[r-1, c:c+3] == val):
                    is_exact = False
                # Check below
                if r+3 < rows and np.all(grid[r+3, c:c+3] == val):
                    is_exact = False
                # Check left
                if c > 0 and np.all(grid[r:r+3, c-1] == val):
                    is_exact = False
                # Check right
                if c+3 < cols and np.all(grid[r:r+3, c+3] == val):
                    is_exact = False
                
                if is_exact:
                    solid_blocks.append((r, c, val))
    
    # Convert solid blocks to crosses
    for (r, c, color) in solid_blocks:
        # Cross pattern: keep (0,1), (1,0), (1,2), (2,1) relative to block top-left
        result[r+1, c] = 0      # top-left corner
        result[r+1, c+1] = 0    # center
        result[r+1, c+2] = 0    # top-right corner
        result[r, c] = 0        # top-left
        result[r+2, c] = 0      # bottom-left
        result[r, c+2] = 0      # top-right
        result[r+2, c+2] = 0    # bottom-right
        # Keep: (r, c+1), (r+1, c), (r+1, c+2), (r+2, c+1) - but we already zeroed (r+1,c)
        # Let me redo this properly
        result[r, c] = 0
        result[r, c+1] = color   # top-center
        result[r, c+2] = 0
        result[r+1, c] = color   # middle-left
        result[r+1, c+1] = 0    # center
        result[r+1, c+2] = color # middle-right
        result[r+2, c] = 0
        result[r+2, c+1] = color # bottom-center
        result[r+2, c+2] = 0
    
    # Step 2: Find the 8-maze boundaries
    mask_8 = (grid == 8)
    if not np.any(mask_8):
        return result
    
    maze_rows = np.where(mask_8.any(axis=1))[0]
    maze_cols = np.where(mask_8.any(axis=0))[0]
    maze_top = maze_rows[0]
    maze_bottom = maze_rows[-1]
    
    # Step 3: Identify top entry points (8s at the top row of maze, aligned with solid blocks)
    top_entries = []  # (row, col, color)
    for (r, c, color) in solid_blocks:
        # The entry is at the maze top row, at the center column of the block
        center_col = c + 1
        entry_row = maze_top
        # Find the nearest 8 in the entry row
        if grid[entry_row, center_col] == 8:
            top_entries.append((entry_row, center_col, color))
        else:
            # Look for nearest 8 in the same column
            for dc in range(-1, 2):
                if 0 <= center_col + dc < cols and grid[entry_row, center_col + dc] == 8:
                    top_entries.append((entry_row, center_col + dc, color))
                    break
    
    # Step 4: Identify bottom exit points (8s at the bottom row of maze)
    bottom_row = maze_bottom
    bottom_exits = []
    for c in range(cols):
        if grid[bottom_row, c] == 8:
            bottom_exits.append(c)
    
    # Step 5: Trace paths from top entries to bottom exits
    # Passable: 0, 5, 8 (the path follows 8s and can go through 0s and 5s)
    # NOT passable: 9 and other colors
    
    def is_passable(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return False
        val = grid[r, c]
        return val in (0, 5, 8)
    
    def find_path(start_r, start_c):
        """BFS from start to find which bottom row 8 we can reach."""
        visited = set()
        queue = deque()
        queue.append((start_r, start_c))
        visited.add((start_r, start_c))
        
        while queue:
            r, c = queue.popleft()
            # Check if we've reached the bottom
            if r == bottom_row and grid[r, c] == 8:
                return c
            # Try all 4 directions
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited and is_passable(nr, nc):
                    # Don't go above the maze top
                    if nr < maze_top:
                        continue
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return None
    
    # For each top entry, find which bottom exit it connects to
    bottom_crosses = {}  # col -> color
    for (er, ec, color) in top_entries:
        exit_col = find_path(er, ec)
        if exit_col is not None:
            bottom_crosses[exit_col] = color
    
    # Step 6: Place crosses at bottom exits
    # The cross is centered at the bottom exit column, 3 rows up from maze bottom
    for exit_col, color in bottom_crosses.items():
        # Place cross at rows (bottom_row-2) to (bottom_row), centered at exit_col
        cr = maze_bottom  # center row of cross
        cc = exit_col     # center col of cross
        # Only place if the area is empty (all zeros)
        if cr >= 2 and cr + 1 < rows and cc >= 1 and cc + 1 < cols:
            # Check if area is empty
            area = result[cr-1:cr+2, cc-1:cc+2]
            if np.all(area == 0):
                # Place cross pattern
                result[cr-1, cc] = color   # top-center
                result[cr, cc-1] = color   # middle-left
                result[cr, cc+1] = color   # middle-right
                result[cr+1, cc] = color   # bottom-center
    
    return result