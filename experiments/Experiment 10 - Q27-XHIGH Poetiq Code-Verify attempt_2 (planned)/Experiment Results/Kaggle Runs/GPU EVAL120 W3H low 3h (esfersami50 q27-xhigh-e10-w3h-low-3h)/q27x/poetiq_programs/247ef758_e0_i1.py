import numpy as np
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transforms the grid by:
    1. Finding the vertical dividing line (column with constant non-zero value)
    2. Identifying the right frame and interior
    3. Extracting objects from the left area
    4. Placing symmetric objects (with both H and V symmetry) into the interior
       at positions determined by frame markers
    5. Non-symmetric objects remain in the left area
    """
    grid = grid.copy()
    rows, cols = grid.shape
    
    # Find the dividing line: column with constant non-zero value
    div_col = None
    for c in range(cols):
        col_vals = grid[:, c]
        if col_vals[0] != 0 and np.all(col_vals == col_vals[0]):
            div_col = c
            break
    if div_col is None:
        return grid
    
    div_val = grid[0, div_col]
    
    # Find the frame boundaries on the right side
    # Top frame: first row where cells to the right of div_col are non-zero
    top_row = None
    for r in range(rows):
        if np.any(grid[r, div_col+1:] != 0):
            top_row = r
            break
    
    # Bottom frame: last row where cells to the right of div_col are non-zero
    bot_row = None
    for r in range(rows - 1, -1, -1):
        if np.any(grid[r, div_col+1:] != 0):
            bot_row = r
            break
    
    # Left frame: first col after div_col that has non-zero in top and bottom rows
    left_frame_col = None
    for c in range(div_col + 1, cols):
        if grid[top_row, c] != 0 and grid[bot_row, c] != 0:
            left_frame_col = c
            break
    
    # Right frame: last col before end that has non-zero in top and bottom rows
    right_frame_col = None
    for c in range(cols - 1, div_col, -1):
        if grid[top_row, c] != 0 and grid[bot_row, c] != 0:
            right_frame_col = c
            break
    
    # Interior bounds
    int_top = top_row + 1
    int_bot = bot_row  # exclusive
    int_left = left_frame_col + 1
    int_right = right_frame_col  # exclusive
    
    # Read frame marker positions
    # Left frame: which row has which color
    left_frame_markers = {}  # color -> row
    for r in range(top_row, bot_row + 1):
        v = grid[r, left_frame_col]
        if v != 0 and v != div_val:
            left_frame_markers[v] = r
    
    # Top frame: which col has which color
    top_frame_markers = {}  # color -> col
    for c in range(left_frame_col, right_frame_col + 1):
        v = grid[top_row, c]
        if v != 0 and v != div_val:
            top_frame_markers[v] = c
    
    # Left area: extract objects
    left_area = grid[:, :div_col].copy()
    
    # Find objects in left area using BFS
    def find_objects(area):
        visited = np.zeros(area.shape, dtype=bool)
        objects = []
        for r in range(area.shape[0]):
            for c in range(area.shape[1]):
                if area[r, c] != 0 and not visited[r, c]:
                    # BFS
                    color = area[r, c]
                    cells = []
                    queue = [(r, c)]
                    visited[r, c] = True
                    while queue:
                        cr, cc = queue.pop(0)
                        cells.append((cr, cc))
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = cr+dr, cc+dc
                            if 0 <= nr < area.shape[0] and 0 <= nc < area.shape[1]:
                                if not visited[nr, nc] and area[nr, nc] == color:
                                    visited[nr, nc] = True
                                    queue.append((nr, nc))
                    objects.append((color, cells))
        return objects
    
    objects = find_objects(left_area)
    
    # Determine symmetry of each object
    def is_symmetric(cells):
        if len(cells) < 2:
            return False
        cell_set = set(cells)
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        h_axis = (min_r + max_r) / 2
        v_axis = (min_c + max_c) / 2
        
        # Check vertical symmetry (reflect across vertical axis)
        for r, c in cells:
            if (r, 2*v_axis - c) not in cell_set:
                return False
        # Check horizontal symmetry (reflect across horizontal axis)
        for r, c in cells:
            if (2*h_axis - r, c) not in cell_set:
                return False
        return True
    
    # Create output
    output = grid.copy()
    
    # Clear interior
    output[int_top:int_bot, int_left:int_right] = 0
    
    # Determine which objects stay and which move
    for color, cells in objects:
        if is_symmetric(cells):
            # Clear from left area
            for r, c in cells:
                output[r, c] = 0
            # Place in interior if color has frame markers
            if color in left_frame_markers and color in top_frame_markers:
                # Center position in interior
                center_r = left_frame_markers[color] - top_row - 1
                center_c = top_frame_markers[color] - left_frame_col - 1
                
                # If center is out of bounds, try alternate offset
                if center_r < 0 or center_r >= int_bot - int_top:
                    center_r = left_frame_markers[color] - top_row
                if center_c < 0 or center_c >= int_right - int_left:
                    center_c = top_frame_markers[color] - left_frame_col
                
                # Place object preserving relative positions
                rs = [r for r, c in cells]
                cs = [c for r, c in cells]
                obj_min_r, obj_max_r = min(rs), max(rs)
                obj_min_c, obj_max_c = min(cs), max(cs)
                obj_center_r = (obj_min_r + obj_max_r) / 2
                obj_center_c = (obj_min_c + obj_max_c) / 2
                
                for r, c in cells:
                    new_r = int_top + center_r + int(r - obj_center_r)
                    new_c = int_left + center_c + int(c - obj_center_c)
                    if 0 <= new_r < rows and 0 <= new_c < cols:
                        output[new_r, new_c] = color
        # Non-symmetric objects stay in place (already in output)
    
    return output