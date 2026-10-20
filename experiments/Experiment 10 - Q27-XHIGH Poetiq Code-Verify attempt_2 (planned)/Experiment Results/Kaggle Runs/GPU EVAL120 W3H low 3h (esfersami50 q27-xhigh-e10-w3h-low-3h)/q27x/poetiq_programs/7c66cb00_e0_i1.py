import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform grid by:
    1. Identifying template shapes (small non-background connected regions)
    2. Identifying target blocks (full-width bordered rectangular regions)
    3. Removing templates (replacing with background)
    4. Stamping templates into each target block at same column, bottom-aligned,
       with color mapping (template border -> block border, template fill -> block fill)
    """
    grid = grid.copy()
    H, W = grid.shape
    
    # Find background color (most common on borders)
    border_vals = list(grid[0, :].tolist()) + list(grid[-1, :].tolist()) + \
                  list(grid[:, 0].tolist()) + list(grid[:, -1].tolist())
    from collections import Counter
    bg_color = Counter(border_vals).most_common(1)[0][0]
    
    # Identify target blocks (full-width rows where col0 != bg and col_W-1 != bg)
    target_blocks = []
    i = 0
    while i < H:
        if grid[i, 0] != bg_color and grid[i, W-1] != bg_color:
            start = i
            while i < H and grid[i, 0] != bg_color and grid[i, W-1] != bg_color:
                i += 1
            end = i - 1
            block_height = end - start + 1
            block_width = W
            border_color = grid[start, 0]
            # Determine interior color
            if block_height >= 3 and block_width >= 3:
                interior_color = grid[start + 1, 1]
            else:
                interior_color = grid[start, 1] if block_width > 1 else border_color
            target_blocks.append({
                'top': start, 'bottom': end,
                'left': 0, 'right': W - 1,
                'border_color': border_color,
                'interior_color': interior_color
            })
        else:
            i += 1
    
    # Find template shapes using flood fill on non-bg, non-target cells
    in_target = np.zeros((H, W), dtype=bool)
    for blk in target_blocks:
        in_target[blk['top']:blk['bottom']+1, blk['left']:blk['right']+1] = True
    
    visited = np.zeros((H, W), dtype=bool)
    templates = []
    
    for r in range(H):
        for c in range(W):
            if grid[r, c] != bg_color and not in_target[r, c] and not visited[r, c]:
                # BFS to find connected component
                queue = deque([(r, c)])
                visited[r, c] = True
                cells = [(r, c)]
                while queue:
                    cr, cc = queue.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and grid[nr, nc] != bg_color and not in_target[nr, nc]:
                            visited[nr, nc] = True
                            cells.append((nr, nc))
                            queue.append((nr, nc))
                
                if len(cells) < 4:
                    continue
                
                rows = [x[0] for x in cells]
                cols = [x[1] for x in cells]
                r_min, r_max = min(rows), max(rows)
                c_min, c_max = min(cols), max(cols)
                sh = r_max - r_min + 1
                sw = c_max - c_min + 1
                
                # Must fit within a target block's height
                if sh > max(b['bottom'] - b['top'] + 1 for b in target_blocks) if target_blocks else True:
                    continue
                
                # Extract shape pattern
                pattern = np.zeros((sh, sw), dtype=int)
                for (cr, cc) in cells:
                    pattern[cr - r_min, cc - c_min] = grid[cr, cc]
                
                # Determine border and fill colors of template
                # Border = color on edges of bounding box that appear in shape
                border_px = set()
                for (cr, cc) in cells:
                    pr, pc = cr - r_min, cc - c_min
                    if pr == 0 or pr == sh-1 or pc == 0 or pc == sw-1:
                        border_px.add(grid[cr, cc])
                
                all_colors = set(grid[cr, cc] for (cr, cc) in cells)
                
                if len(border_px) == 1:
                    t_border = list(border_px)[0]
                    t_fill = list(all_colors - border_px)[0] if len(all_colors) > 1 else t_border
                else:
                    # Shape border has multiple colors - use the most common edge color
                    from collections import Counter as C2
                    edge_counts = C2(grid[cr, cc] for (cr, cc) in cells 
                                     if (cr-r_min==0 or cr-r_min==sh-1 or cc-c_min==0 or cc-c_min==sw-1))
                    t_border = edge_counts.most_common(1)[0][0]
                    t_fill = list(all_colors - {t_border})[0] if len(all_colors) > 1 else t_border
                
                templates.append({
                    'pattern': pattern,
                    'col_min': c_min,
                    'col_max': c_max,
                    'sh': sh, 'sw': sw,
                    'border_color': t_border,
                    'fill_color': t_fill
                })
    
    # Create output: start with background
    output = np.full((H, W), bg_color, dtype=int)
    
    # Place target blocks (unmodified)
    for blk in target_blocks:
        for r in range(blk['top'], blk['bottom']+1):
            for c in range(blk['left'], blk['right']+1):
                if r == blk['top'] or r == blk['bottom'] or c == blk['left'] or c == blk['right']:
                    output[r, c] = blk['border_color']
                else:
                    output[r, c] = blk['interior_color']
    
    # Stamp templates into each target block (bottom-aligned, same column)
    for blk in target_blocks:
        block_h = blk['bottom'] - blk['top'] + 1
        for tpl in templates:
            sh, sw = tpl['sh'], tpl['sw']
            c_min = tpl['col_min']
            # Bottom-aligned placement
            stamp_top = blk['bottom'] - sh + 1
            stamp_bottom = blk['bottom']
            
            for dr in range(sh):
                for dc in range(sw):
                    r = stamp_top + dr
                    c = c_min + dc
                    if 0 <= r <= blk['bottom'] and 0 <= c < W:
                        val = tpl['pattern'][dr, dc]
                        if val == tpl['border_color']:
                            output[r, c] = blk['border_color']
                        else:
                            output[r, c] = blk['interior_color']
    
    return output