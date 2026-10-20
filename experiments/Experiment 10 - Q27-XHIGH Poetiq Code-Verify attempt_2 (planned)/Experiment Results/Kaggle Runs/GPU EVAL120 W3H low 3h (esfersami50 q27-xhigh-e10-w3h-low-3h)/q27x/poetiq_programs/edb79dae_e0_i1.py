import numpy as np
from collections import Counter
from scipy import ndimage

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform grid by:
    1. Finding the 5-border rectangle
    2. Identifying N×N solid blocks inside (separated by a background color)
    3. Finding N×N pattern shapes outside the border
    4. Finding 2-color mapping shapes outside the border  
    5. Output: same layout but with patterns from step 3, colors from step 4
    """
    grid = grid.copy()
    rows, cols = grid.shape
    
    # Step 1: Find the 5-border rectangle
    five_mask = (grid == 5)
    five_rows = np.where(five_mask.any(axis=1))[0]
    five_cols = np.where(five_mask.any(axis=0))[0]
    
    if len(five_rows) == 0 or len(five_cols) == 0:
        return grid
    
    top, bottom = five_rows[0], five_rows[-1]
    left, right = five_cols[0], five_cols[-1]
    
    # Step 2: Extract inside and find blocks
    inside = grid[top+1:bottom, left+1:right].copy()
    ir, ic = inside.shape
    
    # Separator color = most common color in inside
    color_counts = Counter(inside.flatten().tolist())
    sep_color = color_counts.most_common(1)[0][0]
    
    # Find solid N×N blocks
    non_sep = (inside != sep_color)
    labeled, num_features = ndimage.label(non_sep)
    
    blocks = []
    for i in range(1, num_features + 1):
        component = (labeled == i)
        comp_pixels = np.where(component)
        r_min, r_max = int(comp_pixels[0].min()), int(comp_pixels[0].max())
        c_min, c_max = int(comp_pixels[1].min()), int(comp_pixels[1].max())
        h = r_max - r_min + 1
        w = c_max - c_min + 1
        
        region = inside[r_min:r_max+1, c_min:c_max+1]
        if len(np.unique(region)) == 1 and component.sum() == h * w:
            blocks.append({'row': r_min, 'col': c_min, 'h': h, 'w': w, 'color': int(region[0,0])})
    
    if not blocks:
        return grid
    
    N = blocks[0]['h']
    
    # Grid layout
    block_row_positions = sorted(set(b['row'] for b in blocks))
    block_col_positions = sorted(set(b['col'] for b in blocks))
    n_br = len(block_row_positions)
    n_bc = len(block_col_positions)
    
    block_grid = {}
    for b in blocks:
        ri = block_row_positions.index(b['row'])
        ci = block_col_positions.index(b['col'])
        block_grid[(ri, ci)] = b['color']
    
    # Step 3: Find regions outside the border
    outside_mask = np.ones((rows, cols), dtype=bool)
    outside_mask[top:bottom+1, left:right+1] = False
    
    outside_pixels = grid[outside_mask]
    outside_counts = Counter(outside_pixels.tolist())
    outside_bg = outside_counts.most_common(1)[0][0]
    
    outside_non_bg = outside_mask & (grid != outside_bg)
    labeled_out, num_out = ndimage.label(outside_non_bg)
    
    pattern_shapes = []
    mapping_shapes = []
    
    for i in range(1, num_out + 1):
        component = (labeled_out == i)
        comp_pixels = np.where(component)
        r_min, r_max = int(comp_pixels[0].min()), int(comp_pixels[0].max())
        c_min, c_max = int(comp_pixels[1].min()), int(comp_pixels[1].max())
        h = r_max - r_min + 1
        w = c_max - c_min + 1
        
        region = grid[r_min:r_max+1, c_min:c_max+1].copy()
        
        # Get non-bg colors in bounding box
        unique_in_box = set(np.unique(region).tolist())
        if outside_bg in unique_in_box:
            unique_in_box.discard(outside_bg)
        
        if len(unique_in_box) != 2:
            continue
        
        if h == N and w == N:
            pattern_shapes.append({'region': region, 'bg': outside_bg})
        else:
            mapping_shapes.append({'colors': sorted(unique_in_box)})
    
    # Step 4: Build color mapping
    pattern_fg_colors = set()
    for ps in pattern_shapes:
        colors_in_pattern = set(np.unique(ps['region']).tolist()) - {ps['bg']}
        pattern_fg_colors.update(colors_in_pattern)
    
    color_map = {}
    for ms in mapping_shapes:
        c1, c2 = ms['colors']
        if c1 in pattern_fg_colors and c2 not in pattern_fg_colors:
            color_map[c1] = c2
        elif c2 in pattern_fg_colors and c1 not in pattern_fg_colors:
            color_map[c2] = c1
    
    for pc in pattern_fg_colors:
        if pc not in color_map:
            color_map[pc] = pc
    
    # Step 5: Build output
    out_rows = ir + 2
    out_cols = ic + 2
    output = np.full((out_rows, out_cols), 5, dtype=int)
    output[1:-1, 1:-1] = sep_color
    
    # Find pattern for each color
    color_to_pattern = {}
    for ps in pattern_shapes:
        colors_in_pattern = set(np.unique(ps['region']).tolist()) - {ps['bg']}
        for c in colors_in_pattern:
            color_to_pattern[c] = ps
    
    for ri in range(n_br):
        for ci in range(n_bc):
            if (ri, ci) not in block_grid:
                continue
            
            bc = block_grid[(ri, ci)]
            out_c = color_map.get(bc, bc)
            
            r0 = 1 + block_row_positions[ri]
            c0 = 1 + block_col_positions[ci]
            
            ps = color_to_pattern.get(bc)
            if ps:
                region = ps['region']
                bg = ps['bg']
                for dr in range(N):
                    for dc in range(N):
                        output[r0+dr, c0+dc] = sep_color if region[dr,dc] == bg else out_c
            else:
                output[r0:r0+N, c0:c0+N] = out_c
    
    return output