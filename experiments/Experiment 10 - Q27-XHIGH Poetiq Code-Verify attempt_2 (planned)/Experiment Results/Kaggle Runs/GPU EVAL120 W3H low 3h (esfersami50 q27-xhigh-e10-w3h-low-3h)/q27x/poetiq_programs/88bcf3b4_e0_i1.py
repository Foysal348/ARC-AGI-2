import numpy as np
from collections import Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """Rearrange mobile (non-line) objects into V-shaped/line paths between anchor lines."""
    grid = grid.copy()
    h, w = grid.shape
    
    # Find background color
    flat = grid.flatten()
    bg = Counter(flat).most_common(1)[0][0]
    
    # Get all non-background colors and their cells
    non_bg_colors = [c for c in set(flat) if c != bg]
    color_cells = {}
    for c in non_bg_colors:
        color_cells[c] = np.argwhere(grid == c)
    
    # Classify as line (anchor) or non-line (mobile)
    def is_straight_line(cells):
        if len(cells) <= 1:
            return True
        rows, cols = cells[:, 0], cells[:, 1]
        if len(set(cols.tolist())) == 1:
            if rows.max() - rows.min() + 1 == len(cells):
                return True
        if len(set(rows.tolist())) == 1:
            if cols.max() - cols.min() + 1 == len(cells):
                return True
        return False
    
    anchors = {}
    mobiles = {}
    for c in non_bg_colors:
        if is_straight_line(color_cells[c]):
            anchors[c] = color_cells[c]
        else:
            mobiles[c] = color_cells[c]
    
    # Build result: start with background, place anchors
    result = np.full((h, w), bg, dtype=grid.dtype)
    for c, cells in anchors.items():
        for r, cc in cells:
            result[r, cc] = c
    
    # Helper: get anchor center and extent
    def anchor_props(cells):
        r, c = cells[:, 0], cells[:, 1]
        return r.min(), r.max(), c.min(), c.max(), len(cells)
    
    # For each mobile color, create its path
    for mc, mcells in mobiles.items():
        n = len(mcells)
        
        if len(anchors) < 2:
            for r, cc in mcells:
                result[r, cc] = mc
            continue
        
        # Find nearest two anchors
        mr, mc_c = mcells[:, 0].mean(), mcells[:, 1].mean()
        scored = []
        for ac, acells in anchors.items():
            ar, ac2 = acells[:, 0].mean(), acells[:, 1].mean()
            scored.append((abs(ar - mr) + abs(ac2 - mc_c), ac))
        scored.sort()
        src_c, tgt_c = scored[0][1], scored[1][1]
        
        s = anchor_props(anchors[src_c])
        t = anchor_props(anchors[tgt_c])
        # s, t: (row_min, row_max, col_min, col_max, length)
        
        src_above = s[1] < t[0]  # source entirely above target
        src_len, tgt_len = s[4], t[4]
        
        # Determine end_row and available space
        if src_above:
            if src_len <= tgt_len:
                end_row = t[0]
                avail = end_row + 1
            else:
                end_row = t[1] + tgt_len
                avail = end_row - s[1] + 1
        else:
            end_row = max(s[1], t[1])
            avail = end_row + 1
        
        path_len = min(n, avail)
        start_row = end_row - path_len + 1
        
        # Determine column parameters
        src_col = (s[2] + s[3]) // 2
        tgt_col = (t[2] + t[3]) // 2
        
        if src_above and src_len <= tgt_len:
            # V shape, point at source side
            v_col = tgt_col - 1 if src_col < tgt_col else tgt_col + 1
            for row in range(start_row, end_row + 1):
                t_val = min(row - start_row, end_row - row)
                col = v_col + t_val
                col = max(0, min(w - 1, col))
                if result[row, col] == bg:
                    result[row, col] = mc
        elif src_above and src_len > tgt_len:
            # V shape, point at target side (max col in middle)
            min_c = src_col + 1 if src_col < tgt_col else src_col - 1
            max_c = tgt_col + 1 if src_col < tgt_col else tgt_col - 1
            for row in range(start_row, end_row + 1):
                t_val = min(row - start_row, end_row - row)
                col = min(max_c, min_c + t_val)
                col = max(0, min(w - 1, col))
                if result[row, col] == bg:
                    result[row, col] = mc
        else:
            # Line or V
            v_col = tgt_col - 1 if src_col < tgt_col else tgt_col + 1
            for row in range(start_row, end_row + 1):
                t_val = min(row - start_row, end_row - row)
                col = v_col + t_val
                col = max(0, min(w - 1, col))
                if result[row, col] == bg:
                    result[row, col] = mc
    
    return result