import numpy as np
from scipy import ndimage

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform by identifying frame structures and drawing colored lines on their open sides.
    """
    result = grid.copy()
    rows, cols = grid.shape
    non8 = (grid != 8)
    
    # Find connected components
    labeled, num_features = ndimage.label(non8)
    
    frames = []
    for i in range(1, num_features + 1):
        mask = (labeled == i)
        positions = np.argwhere(mask)
        if len(positions) < 4:
            continue
        rmin, cmin = positions.min(axis=0)
        rmax, cmax = positions.max(axis=0)
        h = rmax - rmin + 1
        w = cmax - cmin + 1
        
        if w == 2 and h >= 4:
            frame_data = grid[rmin:rmax+1, cmin:cmax+1]
            left_col = frame_data[:, 0]
            right_col = frame_data[:, 1]
            left_4s = np.sum(left_col == 4)
            right_4s = np.sum(right_col == 4)
            
            if left_4s > right_4s:
                direction = 'right'
                heavy_col = left_col
            else:
                direction = 'left'
                heavy_col = right_col
            
            four_positions = np.where(heavy_col == 4)[0]
            if len(four_positions) >= 3:
                gap_start = four_positions[1] + 1
                gap_end = four_positions[-1] - 1
                gap_len = gap_end - gap_start + 1
                if gap_len >= 2:
                    line1 = 1
                    if gap_len >= 5:
                        line2 = gap_start + 1
                        line3 = gap_start + 3
                    else:
                        line2 = gap_start + 2
                        line3 = gap_start + 3
                    frames.append(('v', rmin, rmax, cmin, cmax, direction, [line1, line2, line3]))
        
        elif h == 2 and w >= 4:
            frame_data = grid[rmin:rmax+1, cmin:cmax+1]
            top_row = frame_data[0, :]
            bottom_row = frame_data[1, :]
            top_4s = np.sum(top_row == 4)
            bottom_4s = np.sum(bottom_row == 4)
            
            if top_4s > bottom_4s:
                direction = 'down'
                heavy_row = top_row
            else:
                direction = 'up'
                heavy_row = bottom_row
            
            four_positions = np.where(heavy_row == 4)[0]
            non4_positions = np.where(heavy_row != 4)[0]
            if len(four_positions) >= 2 and len(non4_positions) >= 2:
                non4_start = non4_positions[0]
                line1 = non4_start
                line2 = non4_start + 1
                four_after = four_positions[four_positions > non4_positions[-1]]
                line3 = four_after[0] if len(four_after) > 0 else non4_positions[-1]
                frames.append(('h', rmin, rmax, cmin, cmax, direction, [line1, line2, line3]))
    
    # Identify seeds
    frame_mask = np.zeros((rows, cols), dtype=bool)
    for f in frames:
        frame_mask[f[1]:f[2]+1, f[3]:f[4]+1] = True
    
    seeds = []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 8 and not frame_mask[r, c]:
                seeds.append((r, c, int(grid[r, c])))
    
    seed_colors = sorted(set(s[2] for s in seeds))
    
    # Determine global color order for horizontal lines
    if seeds:
        seed_rows = [s[0] for s in seeds]
        use_descending_h = np.mean(seed_rows) > rows / 2
    else:
        use_descending_h = False
    
    # Draw lines
    for f in frames:
        ftype, rmin, rmax, cmin, cmax, direction, offsets = f
        colors = seed_colors[:]
        assigned = {}
        
        if ftype == 'v':
            line_rows = [rmin + off for off in offsets if 0 <= rmin + off < rows]
            
            if direction == 'left':
                c_start, c_end = 0, cmin - 1
            else:
                c_start, c_end = cmax + 1, cols - 1
            
            if c_start > c_end:
                continue
            
            # Check seeds on lines
            for lr in line_rows:
                for s in seeds:
                    if s[0] == lr and c_start <= s[1] <= c_end:
                        assigned[lr] = s[2]
                        if s[2] in colors:
                            colors.remove(s[2])
                        break
            
            remaining = [lr for lr in line_rows if lr not in assigned]
            rem_colors = sorted(colors, reverse=True) if use_descending_h else sorted(colors)
            for i, lr in enumerate(remaining):
                if i < len(rem_colors):
                    assigned[lr] = rem_colors[i]
            
            for lr in line_rows:
                if lr in assigned:
                    for c in range(c_start, c_end + 1):
                        if result[lr, c] == 8:
                            result[lr, c] = assigned[lr]
        
        else:  # horizontal frame
            line_cols = [cmin + off for off in offsets if 0 <= cmin + off < cols]
            
            if direction == 'down':
                r_start, r_end = rmax + 1, rows - 1
            else:
                r_start, r_end = 0, rmin - 1
            
            if r_start > r_end:
                continue
            
            for lc in line_cols:
                for s in seeds:
                    if s[1] == lc and r_start <= s[0] <= r_end:
                        assigned[lc] = s[2]
                        if s[2] in colors:
                            colors.remove(s[2])
                        break
            
            remaining = [lc for lc in line_cols if lc not in assigned]
            rem_colors = sorted(colors)
            for i, lc in enumerate(remaining):
                if i < len(rem_colors):
                    assigned[lc] = rem_colors[i]
            
            for lc in line_cols:
                if lc in assigned:
                    for r in range(r_start, r_end + 1):
                        if result[r, lc] == 8:
                            result[r, lc] = assigned[lc]
    
    return result