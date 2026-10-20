import numpy as np
from typing import List, Tuple

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform grid by repositioning 5x5 boxes to fixed column slots.
    - Border=8 boxes move to left side (col 0 or 5)
    - Border=2 boxes move to right side (col 12 or 17)
    - Row positions are preserved
    - Column slot chosen via greedy conflict avoidance
    """
    rows, cols = grid.shape
    bg_color = grid[0, 0]
    
    # Detect all 5x5 boxes
    boxes = []
    for r in range(rows - 4):
        for c in range(cols - 4):
            # Get border color from top-left corner
            border_color = grid[r, c]
            if border_color == bg_color:
                continue
            # Verify it's a valid box: all border pixels match, fill is uniform and different
            # Check border
            valid = True
            for i in range(5):
                if grid[r, c+i] != border_color or grid[r+4, c+i] != border_color:
                    valid = False
                    break
                if grid[r+i, c] != border_color or grid[r+i, c+4] != border_color:
                    valid = False
                    break
            if not valid:
                continue
            # Check fill
            fill_color = grid[r+1, c+1]
            if fill_color == border_color:
                continue
            for i in range(1, 4):
                for j in range(1, 4):
                    if grid[r+i, c+j] != fill_color:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                boxes.append((r, c, border_color, fill_color))
    
    # Sort boxes by (row, col)
    boxes.sort(key=lambda b: (b[0], b[1]))
    
    # Define column slots
    left_slots = [0, 5]
    right_slots = [17, 12]
    
    # Assign columns with greedy conflict avoidance
    assignments = []  # (row, new_col, border_color, fill_color)
    placed = []  # list of (row, col) for placed boxes
    
    def conflicts(r, c, placed_boxes):
        """Check if placing a 5x5 box at (r,c) conflicts with any placed box."""
        for (pr, pc) in placed_boxes:
            # Check overlap: boxes overlap if their row ranges and col ranges both intersect
            row_overlap = (r < pr + 5) and (pr < r + 5)
            col_overlap = (c < pc + 5) and (pc < c + 5)
            if row_overlap and col_overlap:
                return True
        return False
    
    for (r, c, border_color, fill_color) in boxes:
        if border_color == 8:
            slots = left_slots  # Try col 0 first, then col 5
        else:
            slots = right_slots  # Try col 17 first, then col 12
        
        assigned = False
        for slot in slots:
            if not conflicts(r, slot, placed):
                assignments.append((r, slot, border_color, fill_color))
                placed.append((r, slot))
                assigned = True
                break
        if not assigned:
            # Fallback: use first slot
            assignments.append((r, slots[0], border_color, fill_color))
            placed.append((r, slots[0]))
    
    # Create output grid
    new_grid = np.full((rows, cols), bg_color, dtype=grid.dtype)
    
    # Place boxes
    for (r, c, border_color, fill_color) in assignments:
        for i in range(5):
            for j in range(5):
                if i == 0 or i == 4 or j == 0 or j == 4:
                    new_grid[r+i, c+j] = border_color
                else:
                    new_grid[r+i, c+j] = fill_color
    
    return new_grid