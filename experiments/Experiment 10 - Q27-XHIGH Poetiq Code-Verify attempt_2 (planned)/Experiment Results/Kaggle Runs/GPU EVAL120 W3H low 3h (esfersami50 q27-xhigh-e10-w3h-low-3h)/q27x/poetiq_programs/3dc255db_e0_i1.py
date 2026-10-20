import numpy as np
from scipy.ndimage import binary_fill_holes

def transform(grid: np.ndarray) -> np.ndarray:
    """
    For each container (ring) shape with inner pixels inside it,
    move the inner pixels to a line above the container's topmost point.
    If inner pixels are outside the container's bounding box on one side,
    move the inside ones to the opposite side.
    """
    result = np.array(grid, dtype=int)
    rows, cols = grid.shape
    colors = set(grid[grid != 0].flatten().tolist())
    
    color_positions = {}
    for c in colors:
        color_positions[c] = np.argwhere(grid == c)
    
    def get_bbox(positions):
        return int(positions[:,0].min()), int(positions[:,1].min()), int(positions[:,0].max()), int(positions[:,1].max())
    
    def is_ring(mask):
        filled = binary_fill_holes(mask)
        return bool(np.any(filled & ~mask))
    
    # Find container-inner pairs
    pairs = []
    for c1 in colors:
        mask1 = (grid == c1)
        if is_ring(mask1):
            bb1 = get_bbox(color_positions[c1])
            for c2 in colors:
                if c2 == c1:
                    continue
                bb2 = get_bbox(color_positions[c2])
                if bb1[0] <= bb2[0] and bb1[1] <= bb2[1] and bb1[2] >= bb2[2] and bb1[3] >= bb2[3]:
                    pairs.append((c1, c2))
    
    # Fallback: use size-based heuristics
    if not pairs:
        sorted_colors = sorted(colors, key=lambda c: -len(color_positions[c]))
        for c1 in sorted_colors:
            bb1 = get_bbox(color_positions[c1])
            for c2 in sorted_colors:
                if c2 == c1:
                    continue
                bb2 = get_bbox(color_positions[c2])
                if bb1[0] <= bb2[0] and bb1[1] <= bb2[1] and bb1[2] >= bb2[2] and bb1[3] >= bb2[3]:
                    if (c1, c2) not in pairs:
                        pairs.append((c1, c2))
    
    for container_color, inner_color in pairs:
        inner_pos = color_positions[inner_color]
        cont_mask = (grid == container_color)
        cmin_r, cmin_c, cmax_r, cmax_c = get_bbox(color_positions[container_color])
        
        # Get per-row extents of container
        cont_rows = {}
        for r in range(cmin_r, cmax_r + 1):
            rc = np.where(cont_mask[r])[0]
            if len(rc) > 0:
                cont_rows[r] = (int(rc.min()), int(rc.max()))
        
        # Centroids
        ic_r, ic_c = inner_pos[:,0].mean(), inner_pos[:,1].mean()
        cc_r, cc_c = color_positions[container_color][:,0].mean(), color_positions[container_color][:,1].mean()
        dr, dc = ic_r - cc_r, ic_c - cc_c
        
        # Separate inside and outside pixels
        inside = [(int(r), int(c)) for r, c in inner_pos if cmin_c <= c <= cmax_c and cmin_r <= r <= cmax_r]
        outside = [(int(r), int(c)) for r, c in inner_pos if not (cmin_c <= c <= cmax_c and cmin_r <= r <= cmax_r)]
        
        if not inside:
            continue
        
        if abs(dc) > abs(dr) and abs(dc) > 0.5:
            # Horizontal direction
            if dc < 0:
                # Inner is left → push to right
                best_row = max(cont_rows.keys(), key=lambda r: cont_rows[r][1])
                start = cont_rows[best_row][1] + 1
                new_pos = [(best_row, start + i) for i in range(len(inside))]
            else:
                # Inner is right → push to left
                best_row = min(cont_rows.keys(), key=lambda r: cont_rows[r][0])
                start = cont_rows[best_row][0] - 1
                new_pos = [(best_row, start - i) for i in range(len(inside))]
        else:
            # Push up
            top_row = cmin_r
            top_cols = np.where(cont_mask[top_row])[0]
            center_col = int(round(top_cols.mean()))
            new_pos = [(top_row - 1 - i, center_col) for i in range(len(inside))]
            new_pos = [p for p in new_pos if p[0] >= 0]
        
        # Apply: clear all inner pixels, restore outside, place new
        for r, c in inner_pos:
            result[r, c] = 0
        for r, c in outside:
            result[r, c] = inner_color
        for r, c in new_pos:
            if 0 <= r < rows and 0 <= c < cols:
                result[r, c] = inner_color
    
    return result