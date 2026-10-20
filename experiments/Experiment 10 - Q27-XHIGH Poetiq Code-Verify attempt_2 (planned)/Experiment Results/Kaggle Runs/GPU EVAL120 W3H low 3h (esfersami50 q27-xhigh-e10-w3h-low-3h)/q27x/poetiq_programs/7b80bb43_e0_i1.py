import numpy as np
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by completing line structures:
    1. Fill gaps between anchored line segments
    2. Extend anchored lines to cover all pixels in their row/column
    3. Snap nearby stray pixels to lines
    4. Remove remaining strays
    """
    grid = grid.copy()
    rows, cols = grid.shape
    
    # Find background (most common) and object colors
    colors, counts = np.unique(grid, return_counts=True)
    bg_color = colors[np.argmax(counts)]
    object_colors = [c for c in colors if c != bg_color]
    if not object_colors:
        return grid
    obj_color = object_colors[0]
    
    # Helper: get all object pixel positions
    obj_pixels = set(zip(*np.where(grid == obj_color)))
    
    # Find horizontal runs (maximal consecutive in a row)
    def find_h_runs():
        runs = []
        for r in range(rows):
            c = 0
            while c < cols:
                if grid[r, c] == obj_color:
                    start = c
                    while c < cols and grid[r, c] == obj_color:
                        c += 1
                    runs.append((r, start, c - 1))  # (row, col_start, col_end)
                else:
                    c += 1
        return runs
    
    # Find vertical runs (maximal consecutive in a column)
    def find_v_runs():
        runs = []
        for c in range(cols):
            r = 0
            while r < rows:
                if grid[r, c] == obj_color:
                    start = r
                    while r < rows and grid[r, c] == obj_color:
                        r += 1
                    runs.append((c, start, r - 1))  # (col, row_start, row_end)
                else:
                    r += 1
        return runs
    
    h_runs = find_h_runs()
    v_runs = find_v_runs()
    
    # Determine which runs are anchored
    # A horizontal run is anchored if any of its pixels is on a vertical run of length >= 2
    v_run_pixels = set()
    for (c, r1, r2) in v_runs:
        if r2 - r1 + 1 >= 2:
            for r in range(r1, r2 + 1):
                v_run_pixels.add((r, c))
    
    h_run_pixels = set()
    for (r, c1, c2) in h_runs:
        if c2 - c1 + 1 >= 2:
            for c in range(c1, c2 + 1):
                h_run_pixels.add((r, c))
    
    def is_h_anchored(run):
        r, c1, c2 = run
        for c in range(c1, c2 + 1):
            if (r, c) in v_run_pixels:
                return True
        return False
    
    def is_v_anchored(run):
        c, r1, r2 = run
        for r in range(r1, r2 + 1):
            if (r, c) in h_run_pixels:
                return True
        return False
    
    anchored_h_runs = [r for r in h_runs if is_h_anchored(r)]
    anchored_v_runs = [r for r in v_runs if is_v_anchored(r)]
    
    # Step 1: Fill gaps between anchored runs in the same row
    # Group anchored h_runs by row
    h_runs_by_row = defaultdict(list)
    for r in anchored_h_runs:
        h_runs_by_row[r[0]].append(r)
    
    for row in h_runs_by_row:
        runs = sorted(h_runs_by_row[row], key=lambda x: x[1])
        for i in range(len(runs) - 1):
            r1, c1_start, c1_end = runs[i]
            r2, c2_start, c2_end = runs[i + 1]
            gap = c2_start - c1_end - 1
            len1 = c1_end - c1_start + 1
            len2 = c2_end - c2_start + 1
            if gap >= 1 and gap <= min(len1, len2):
                for c in range(c1_end + 1, c2_start):
                    grid[row, c] = obj_color
    
    # Step 2: Fill gaps between anchored runs in the same column
    v_runs_by_col = defaultdict(list)
    for r in anchored_v_runs:
        v_runs_by_col[r[0]].append(r)
    
    for col in v_runs_by_col:
        runs = sorted(v_runs_by_col[col], key=lambda x: x[1])
        for i in range(len(runs) - 1):
            c1, r1_start, r1_end = runs[i]
            c2, r2_start, r2_end = runs[i + 1]
            gap = r2_start - r1_end - 1
            len1 = r1_end - r1_start + 1
            len2 = r2_end - r2_start + 1
            if gap >= 1 and gap <= min(len1, len2):
                for r in range(r1_end + 1, r2_start):
                    grid[r, col] = obj_color
    
    # Step 3: For each anchored vertical run, extend to cover all object pixels in that column
    # between the topmost and bottommost object pixel in the column
    new_obj_pixels = set(zip(*np.where(grid == obj_color)))
    
    for (c, r1, r2) in anchored_v_runs:
        # Find all object pixels in this column
        col_pixels = sorted([r for (r, cc) in new_obj_pixels if cc == c])
        if col_pixels:
            top, bottom = col_pixels[0], col_pixels[-1]
            for r in range(top, bottom + 1):
                grid[r, c] = obj_color
    
    # Step 4: For each anchored horizontal run, extend to cover all object pixels in that row
    new_obj_pixels = set(zip(*np.where(grid == obj_color)))
    for (r, c1, c2) in anchored_h_runs:
        row_pixels = sorted([c for (rr, c) in new_obj_pixels if rr == r])
        if row_pixels:
            left, right = row_pixels[0], row_pixels[-1]
            for c in range(left, right + 1):
                grid[r, c] = obj_color
    
    # Step 5: Snap stray pixels near lines and remove the rest
    # Re-find runs after modifications
    h_runs = find_h_runs()
    v_runs = find_v_runs()
    
    # Build set of all pixels on runs of length >= 2
    line_pixels = set()
    for (r, c1, c2) in h_runs:
        if c2 - c1 + 1 >= 2:
            for c in range(c1, c2 + 1):
                line_pixels.add((r, c))
    for (c, r1, r2) in v_runs:
        if r2 - r1 + 1 >= 2:
            for r in range(r1, r2 + 1):
                line_pixels.add((r, c))
    
    # Find stray pixels (object pixels not on any line)
    stray_pixels = []
    for (r, c) in set(zip(*np.where(grid == obj_color))):
        if (r, c) not in line_pixels:
            stray_pixels.append((r, c))
    
    # For each stray, try to snap to nearest line
    to_remove = set()
    to_add = set()
    
    for (r, c) in stray_pixels:
        # Check horizontal: find nearest h_run in same row
        best_dist = 3  # max distance to snap
        best_pos = None
        for (rr, c1, c2) in h_runs:
            if rr == r and c2 - c1 + 1 >= 2:
                if c < c1:
                    d = c1 - c
                    if d < best_dist:
                        best_dist = d
                        best_pos = (r, c1)
                elif c > c2:
                    d = c - c2
                    if d < best_dist:
                        best_dist = d
                        best_pos = (r, c2)
        
        # Check vertical: find nearest v_run in same column
        for (cc, r1, r2) in v_runs:
            if cc == c and r2 - r1 + 1 >= 2:
                if r < r1:
                    d = r1 - r
                    if d < best_dist:
                        best_dist = d
                        best_pos = (r1, c)
                elif r > r2:
                    d = r - r2
                    if d < best_dist:
                        best_dist = d
                        best_pos = (r2, c)
        
        if best_pos is not None and grid[best_pos] != obj_color:
            to_add.add(best_pos)
            to_remove.add((r, c))
        else:
            to_remove.add((r, c))
    
    for (r, c) in to_remove:
        grid[r, c] = bg_color
    for (r, c) in to_add:
        grid[r, c] = obj_color
    
    return grid