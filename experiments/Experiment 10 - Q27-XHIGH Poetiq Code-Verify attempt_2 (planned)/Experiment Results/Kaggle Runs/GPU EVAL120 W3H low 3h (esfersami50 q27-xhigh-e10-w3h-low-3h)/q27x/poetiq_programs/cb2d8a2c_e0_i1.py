import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the ARC grid by:
    1. Replacing all 1s in line structures with 2s
    2. Drawing a serpentine path of 3s from the source '3' to the opposite edge,
       using two cross-direction lanes (positions 1 and 8) to navigate around lines.
    
    The path alternates between two lanes, turning at midpoints of gaps between lines.
    """
    result = grid.copy().astype(int)
    rows, cols = result.shape
    
    # Find source (the '3')
    sr, sc = -1, -1
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 3:
                sr, sc = r, c
                break
        if sr != -1:
            break
    if sr == -1:
        return result
    
    # Find horizontal lines (runs of 1/2 with length >= 2)
    h_lines = []
    for r in range(rows):
        c = 0
        while c < cols:
            if grid[r, c] in (1, 2):
                start = c
                while c < cols and grid[r, c] in (1, 2):
                    c += 1
                if c - start >= 2:
                    h_lines.append((r, start, c - 1))
            else:
                c += 1
    
    # Find vertical lines
    v_lines = []
    for c in range(cols):
        r = 0
        while r < rows:
            if grid[r, c] in (1, 2):
                start = r
                while r < rows and grid[r, c] in (1, 2):
                    r += 1
                if r - start >= 2:
                    v_lines.append((c, start, r - 1))
            else:
                r += 1
    
    # Replace 1s with 2s in line cells
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1:
                in_line = False
                for (lr, lc1, lc2) in h_lines:
                    if lr == r and lc1 <= c <= lc2:
                        in_line = True
                for (lc, lr1, lr2) in v_lines:
                    if lc == c and lr1 <= r <= lr2:
                        in_line = True
                if in_line:
                    result[r, c] = 2
    
    # Determine primary direction
    if sr == 0:
        direction = 'down'
    elif sr == rows - 1:
        direction = 'up'
    elif sc == 0:
        direction = 'right'
    else:
        direction = 'left'
    
    path_cells = [(sr, sc)]
    
    if direction == 'down':
        lane_a, lane_b = 1, 8
        line_positions = sorted(set(lr for lr, _, _ in h_lines))
        boundaries = [sr] + line_positions + [rows - 1]
        
        # Compute turn rows at midpoints of gaps
        turn_rows = []
        for i in range(len(boundaries) - 1):
            if i == 0:
                a, b = boundaries[i], boundaries[i+1] - 1
            elif i == len(boundaries) - 2:
                a, b = boundaries[i] + 1, boundaries[i+1]
            else:
                a, b = boundaries[i] + 1, boundaries[i+1] - 1
            if b >= a:
                turn_rows.append((a + b) // 2)
        
        seg_rows = [sr] + turn_rows + [rows - 1]
        
        # Assign lanes: alternate starting based on source position
        if sc >= (lane_a + lane_b) // 2:
            lane_seq = [sc] + [lane_a if i % 2 == 0 else lane_b for i in range(len(turn_rows))] + [lane_b]
        else:
            lane_seq = [sc] + [lane_b if i % 2 == 0 else lane_a for i in range(len(turn_rows))] + [lane_b]
        
        for i in range(len(seg_rows) - 1):
            r1, r2 = seg_rows[i], seg_rows[i+1]
            c = lane_seq[i]
            for r in range(r1, r2 + 1):
                path_cells.append((r, c))
            if i < len(seg_rows) - 2:
                nc = lane_seq[i+1]
                for cc in range(min(c, nc), max(c, nc) + 1):
                    path_cells.append((r2, cc))
    
    elif direction == 'right':
        lane_a, lane_b = sr, 8
        line_positions = sorted(set(lc for lc, _, _ in v_lines))
        boundaries = [sc] + line_positions + [cols - 1]
        
        turn_cols = []
        for i in range(len(boundaries) - 1):
            if i == 0:
                a, b = boundaries[i], boundaries[i+1] - 1
            elif i == len(boundaries) - 2:
                a, b = boundaries[i] + 1, boundaries[i+1]
            else:
                a, b = boundaries[i] + 1, boundaries[i+1] - 1
            if b >= a:
                turn_cols.append((a + b) // 2)
        
        seg_cols = [sc] + turn_cols + [cols - 1]
        lane_seq = [sr] + [lane_b if i % 2 == 0 else lane_a for i in range(len(turn_cols))] + [lane_a]
        
        for i in range(len(seg_cols) - 1):
            c1, c2 = seg_cols[i], seg_cols[i+1]
            r = lane_seq[i]
            for c in range(c1, c2 + 1):
                path_cells.append((r, c))
            if i < len(seg_cols) - 2:
                nr = lane_seq[i+1]
                for rr in range(min(r, nr), max(r, nr) + 1):
                    path_cells.append((rr, c2))
    
    # Draw path
    for r, c in path_cells:
        if 0 <= r < rows and 0 <= c < cols:
            result[r, c] = 3
    
    return result