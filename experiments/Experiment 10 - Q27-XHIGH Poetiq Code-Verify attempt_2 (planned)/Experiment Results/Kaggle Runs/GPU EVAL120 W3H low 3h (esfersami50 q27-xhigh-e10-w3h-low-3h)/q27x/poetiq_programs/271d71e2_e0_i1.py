import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform: Units (9-block + adjacent 0-bordered box) move toward the 9-block.
    Content 5s convert to 7s based on movement distance.
    """
    grid = grid.copy()
    rows, cols = grid.shape
    result = grid.copy()  # Start with original, we'll modify unit areas
    
    # Step 1: Find all 9-blocks (connected components of 9)
    visited = np.zeros((rows, cols), dtype=bool)
    nine_blocks = []
    
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 9 and not visited[r, c]:
                block = []
                queue = deque([(r, c)])
                visited[r, c] = True
                while queue:
                    cr, cc = queue.popleft()
                    block.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr,nc] == 9 and not visited[nr,nc]:
                            visited[nr,nc] = True
                            queue.append((nr, nc))
                nine_blocks.append(block)
    
    # Step 2: For each 9-block, find adjacent box and determine unit
    units = []
    for block in nine_blocks:
        min_r = min(r for r,c in block)
        max_r = max(r for r,c in block)
        min_c = min(c for r,c in block)
        max_c = max(c for r,c in block)
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        
        # Must be a solid rectangle
        if len(block) != h * w:
            continue
        
        # Try each direction
        for direction, check in get_direction_checks(grid, min_r, max_r, min_c, max_c, h, w):
            box = check()
            if box:
                units.append((min_r, min_c, max_r, max_c, direction, box))
                break
    
    # Step 3: Process each unit
    for (nmin_r, nmin_c, nmax_r, nmax_c, direction, box) in units:
        bh = nmax_r - nmin_r + 1
        bw = nmax_c - nmin_c + 1
        bmin_r, bmin_c, bmax_r, bmax_c = box
        
        # Get content
        cr1, cc1 = bmin_r + 1, bmin_c + 1
        cr2, cc2 = bmax_r - 1, bmax_c - 1
        if cr2 < cr1 or cc2 < cc1:
            continue
        content = grid[cr1:cr2+1, cc1:cc2+1].copy()
        ch, cw = content.shape
        
        # Movement distance = dimension of 9-block perpendicular to movement
        if direction in ['LEFT', 'RIGHT']:
            move = bh
        else:
            move = bw
        
        # Cap by available space
        move = cap_movement(grid, direction, nmin_r, nmax_r, nmin_c, nmax_c, bmin_r, bmax_r, bmin_c, bmax_c, rows, cols)
        
        # Convert 5s to 7s
        num_convert = min(move, ch * cw)
        content = convert_content(content, direction, num_convert)
        
        # Clear the original unit area in result
        clear_unit(result, nmin_r, nmin_c, nmax_r, nmax_c, bmin_r, bmin_c, bmax_r, bmax_c, direction)
        
        # Place new unit
        place_unit(result, grid, nmin_r, nmin_c, nmax_r, nmax_c, bmin_r, bmin_c, bmax_r, bmax_c, direction, move, content)
    
    return result

def get_direction_checks(grid, min_r, max_r, min_c, max_c, h, w):
    """Generate direction checks for a 9-block."""
    checks = []
    rows, cols = grid.shape
    
    # Box below (9s on top) - direction DOWN means 9s pull box down? No.
    # Convention: direction = where the 9s are relative to box
    # 'TOP' = 9s on top of box (box is below 9s)
    # Box is below: check row min_r+h for 0s
    if min_r + h < rows:
        if all(grid[min_r+h, c] == 0 for c in range(min_c, max_c+1)):
            box = _find_box(grid, min_r+h, min_c, max_c, 'below')
            if box:
                checks.append(('TOP', lambda b=box: b))
    
    # Box above (9s on bottom)
    if min_r > 0:
        if all(grid[min_r-1, c] == 0 for c in range(min_c, max_c+1)):
            box = _find_box(grid, min_r-1, min_c, max_c, 'above')
            if box:
                checks.append(('BOTTOM', lambda b=box: b))
    
    # Box to the right (9s on left)
    if max_c + w < cols:
        if all(grid[r, max_c+w] == 0 for r in range(min_r, max_r+1)):
            box = _find_box(grid, min_r, max_r, max_c+w, 'right')
            if box:
                checks.append(('LEFT', lambda b=box: b))
    
    # Box to the left (9s on right)
    if min_c > 0:
        if all(grid[r, min_c-1] == 0 for r in range(min_r, max_r+1)):
            box = _find_box(grid, min_r, max_r, min_c-1, 'left')
            if box:
                checks.append(('RIGHT', lambda b=box: b))
    
    return checks

def _find_box(grid, start, c1, c2, side):
    """Find a 0-bordered box starting from a border line."""
    rows, cols = grid.shape
    if side == 'below':
        # Top border at 'start', find bottom border
        for r in range(start+1, rows):
            if all(grid[r, c] == 0 for c in range(c1, c2+1)):
                if all(grid[rr, c1] == 0 and grid[rr, c2] == 0 for rr in range(start, r+1)):
                    return (start, c1, r, c2)
        return None
    elif side == 'above':
        for r in range(start-1, -1, -1):
            if all(grid[r, c] == 0 for c in range(c1, c2+1)):
                if all(grid[rr, c1] == 0 and grid[rr, c2] == 0 for rr in range(r, start+1)):
                    return (r, c1, start, c2)
        return None
    elif side == 'right':
        # Left border at 'start' (col), find right border
        for c in range(start+1, cols):
            if all(grid[r, c] == 0 for r in range(c1, c2+1)):
                if all(grid[c1, cc] == 0 and grid[c2, cc] == 0 for cc in range(start, c+1)):
                    return (c1, start, c2, c)
        return None
    elif side == 'left':
        for c in range(start-1, -1, -1):
            if all(grid[r, c] == 0 for r in range(c1, c2+1)):
                if all(grid[c1, cc] == 0 and grid[c2, cc] == 0 for cc in range(c, start+1)):
                    return (c1, c, c2, start)
        return None

def cap_movement(grid, direction, nmin_r, nmax_r, nmin_c, nmax_c, bmin_r, bmax_r, bmin_c, bmax_c, rows, cols):
    """Cap movement by available space."""
    # Determine the extent of the whole unit
    if direction == 'TOP':
        unit_min_r, unit_max_r = nmin_r, bmax_r
        unit_min_c, unit_max_c = min(nmin_c, bmin_c), max(nmax_c, bmax_c)
        # Moving up: available = unit_min_r (distance to top edge)
        return unit_min_r
    elif direction == 'BOTTOM':
        unit_min_r, unit_max_r = bmin_r, nmax_r
        # Moving down: available = rows - 1 - unit_max_r
        return rows - 1 - unit_max_r
    elif direction == 'LEFT':
        unit_min_c, unit_max_c = min(nmin_c, bmin_c), max(nmax_c, bmax_c)
        unit_min_r, unit_max_r = min(nmin_r, bmin_r), max(nmax_r, bmax_r)
        # Moving left: check for obstacles
        available = unit_min_c
        # Check for other 9s to the left
        for r in range(unit_min_r, unit_max_r+1):
            for c in range(unit_min_c-1, -1, -1):
                if grid[r, c] == 9:
                    available = min(available, unit_min_c - c - 2)  # leave gap
                    break
                elif grid[r, c] == 6:
                    break
        return max(0, available)
    elif direction == 'RIGHT':
        unit_min_c, unit_max_c = min(nmin_c, bmin_c), max(nmax_c, bmax_c)
        unit_min_r, unit_max_r = min(nmin_r, bmin_r), max(nmax_r, bmax_r)
        available = cols - 1 - unit_max_c
        for r in range(unit_min_r, unit_max_r+1):
            for c in range(unit_max_c+1, cols):
                if grid[r, c] == 9:
                    available = min(available, c - unit_max_c - 2)
                    break
                elif grid[r, c] == 6:
                    break
        return max(0, available)
    return 0

def convert_content(content, direction, num_convert):
    """Convert 5s to 7s based on direction and count."""
    content = content.copy()
    ch, cw = content.shape
    count = 0
    
    if direction == 'LEFT':
        for c in range(cw):
            for r in range(ch-1, -1, -1):
                if count >= num_convert:
                    return content
                if content[r, c] == 5:
                    content[r, c] = 7
                    count += 1
    elif direction == 'RIGHT':
        for c in range(cw-1, -1, -1):
            for r in range(ch):
                if count >= num_convert:
                    return content
                if content[r, c] == 5:
                    content[r, c] = 7
                    count += 1
    elif direction == 'TOP':
        for r in range(ch):
            for c in range(cw):
                if count >= num_convert:
                    return content
                if content[r, c] == 5:
                    content[r, c] = 7
                    count += 1
    elif direction == 'BOTTOM':
        for r in range(ch-1, -1, -1):
            for c in range(cw-1, -1, -1):
                if count >= num_convert:
                    return content
                if content[r, c] == 5:
                    content[r, c] = 7
                    count += 1
    return content

def clear_unit(result, nmin_r, nmin_c, nmax_r, nmax_c, bmin_r, bmin_c, bmax_r, bmax_c, direction):
    """Clear the unit area in result (set to 6)."""
    # Clear 9-block area
    result[nmin_r:nmax_r+1, nmin_c:nmax_c+1] = 6
    # Clear box area
    result[bmin_r:bmax_r+1, bmin_c:bmax_c+1] = 6

def place_unit(result, grid, nmin_r, nmin_c, nmax_r, nmax_c, bmin_r, bmin_c, bmax_r, bmax_c, direction, move, content):
    """Place the unit at its new position."""
    if direction == 'TOP':
        # 9s on top, box below. Unit moves up by 'move'
        new_nmin_r = nmin_r - move
        # Place 9s
        result[new_nmin_r:new_nmin_r+(nmax_r-nmin_r+1), nmin_c:nmax_c+1] = 9
        # Place box (relative position to 9s stays same)
        offset = bmin_r - nmax_r  # gap between 9 bottom and box top (usually 0)
        new_bmin_r = new_nmin_r + (nmax_r-nmin_r+1) + offset
        result[new_bmin_r:new_bmin_r+(bmax_r-bmin_r+1), bmin_c:bmax_c+1] = 0
        cr1, cc1 = new_bmin_r+1, bmin_c+1
        cr2, cc2 = new_bmin_r+(bmax_r-bmin_r+1)-1, bmin_c+(bmax_c-bmin_c+1)-1
        result[cr1:cr2+1, cc1:cc2+1] = content
    elif direction == 'BOTTOM':
        new_nmin_r = nmin_r + move
        result[new_nmin_r:new_nmin_r+(nmax_r-nmin_r+1), nmin_c:nmax_c+1] = 9
        offset = nmin_r - bmax_r
        new_bmax_r = new_nmin_r - offset
        new_bmin_r = new_bmax_r - (bmax_r-bmin_r)
        result[new_bmin_r:new_bmax_r+1, bmin_c:bmax_c+1] = 0
        cr1, cc1 = new_bmin_r+1, bmin_c+1
        cr2, cc2 = new_bmax_r, bmin_c+(bmax_c-bmin_c+1)-1
        result[cr1:cr2+1, cc1:cc2+1] = content
    elif direction == 'LEFT':
        # 9s on right, box on left. Unit moves left by 'move'
        new_nmin_c = nmin_c - move
        result[nmin_r:nmax_r+1, new_nmin_c:new_nmin_c+(nmax_c-nmin_c+1)] = 9
        offset = nmin_c - bmax_c
        new_bmax_c = new_nmin_c - offset
        new_bmin_c = new_bmax_c - (bmax_c-bmin_c)
        result[bmin_r:bmax_r+1, new_bmin_c:new_bmax_c+1] = 0
        cr1, cc1 = bmin_r+1, new_bmin_c+1
        cr2, cc2 = bmax_r-1, new_bmax_c-1
        result[cr1:cr2+1, cc1:cc2+1] = content
    elif direction == 'RIGHT':
        # 9s on left, box on right. Unit moves right by 'move'
        new_nmin_c = nmin_c + move
        result[nmin_r:nmax_r+1, new_nmin_c:new_nmin_c+(nmax_c-nmin_c+1)] = 9
        offset = bmin_c - nmax_c
        new_bmin_c = new_nmin_c + (nmax_c-nmin_c+1) + offset
        result[bmin_r:bmax_r+1, new_bmin_c:new_bmin_c+(bmax_c-bmin_c+1)] = 0
        cr1, cc1 = bmin_r+1, new_bmin_c+1
        cr2, cc2 = bmax_r-1, new_bmin_c+(bmax_c-bmin_c+1)-1
        result[cr1:cr2+1, cc1:cc2+1] = content