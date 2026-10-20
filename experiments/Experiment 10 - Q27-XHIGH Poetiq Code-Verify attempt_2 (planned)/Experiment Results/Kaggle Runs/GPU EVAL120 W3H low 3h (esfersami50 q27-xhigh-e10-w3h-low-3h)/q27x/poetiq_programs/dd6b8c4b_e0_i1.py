import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Move 9s that can reach the 3x3 center block (containing a 2) through
    non-6, non-9 cells into the block, replacing 3s and the 2 in row-major order.
    6s act as walls that block connectivity.
    """
    grid = grid.copy()
    rows, cols = grid.shape
    
    # Find the 3x3 block center (the cell with value 2)
    center_pos = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                center_pos = (r, c)
                break
        if center_pos:
            break
    
    if center_pos is None:
        return grid
    
    cr, cc = center_pos
    
    # The 3x3 block positions (row-major order, including center)
    block_positions = []
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            block_positions.append((cr + dr, cc + dc))
    
    # Identify all 9 positions outside the 3x3 block
    block_set = set(block_positions)
    nine_positions = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 9 and (r, c) not in block_set:
                nine_positions.append((r, c))
    
    # For each 9, check if it can reach the 3-block through non-6, non-9 cells
    def can_reach_block(start_r, start_c):
        """BFS from a 9's position to see if it can reach any block cell
        through cells with value 7, 3, or 2 (non-6, non-9)."""
        visited = set()
        queue = deque()
        queue.append((start_r, start_c))
        visited.add((start_r, start_c))
        
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    if grid[nr][nc] in (7, 3, 2):
                        if (nr, nc) in block_set:
                            return True
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return False
    
    # Find which 9s can reach the block
    movable_nines = []
    for pos in nine_positions:
        if can_reach_block(pos[0], pos[1]):
            movable_nines.append(pos)
    
    # Sort movable nines (for deterministic behavior)
    movable_nines.sort()
    
    n_to_move = len(movable_nines)
    
    # Remove the 9s from their positions (set to 7)
    for r, c in movable_nines:
        grid[r][c] = 7
    
    # Place 9s into the block in row-major order
    for i, (r, c) in enumerate(block_positions):
        if i < n_to_move:
            grid[r][c] = 9
    
    return grid