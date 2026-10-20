import numpy as np
from collections import Counter, deque

def transform(grid: np.ndarray) -> np.ndarray:
    """Transform by removing 2x2 source blocks and tiling their colors
    as a repeating pattern along paths in the target region."""
    grid = grid.copy()
    rows, cols = grid.shape
    result = grid.copy()
    
    # Identify background colors (two most common)
    color_counts = Counter(grid.flatten().tolist())
    sorted_colors = sorted(color_counts.items(), key=lambda x: (-x[1], x[0]))
    bg1, bg2 = sorted_colors[0][0], sorted_colors[1][0]
    
    # Find 2x2 blocks of non-background colors
    blocks = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            v = grid[r, c]
            if v != bg1 and v != bg2:
                if grid[r, c+1] == v and grid[r+1, c] == v and grid[r+1, c+1] == v:
                    blocks.append((r, c, v))
    
    if not blocks:
        return result
    
    # Determine source and target background
    br, bc, _ = blocks[0]
    source_bg = None
    for dr, dc in [(-1,0),(0,-1),(0,1),(1,0)]:
        nr, nc = br+dr, bc+dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if grid[nr, nc] in (bg1, bg2):
                source_bg = grid[nr, nc]
                break
    target_bg = bg2 if source_bg == bg1 else bg1
    
    # Ordered unique source colors (by column position)
    src_colors = []
    for r, c, v in sorted(blocks, key=lambda x: (x[0], x[1])):
        if v not in src_colors:
            src_colors.append(v)
    
    # Remove blocks
    for r, c, v in blocks:
        for dr in range(2):
            for dc in range(2):
                result[r+dr, c+dc] = source_bg
    
    # Find target region (flood fill from target_bg cells away from blocks)
    target_region = set()
    seeds = []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == target_bg:
                near_block = any(abs(r-br) <= 2 and abs(c-bc) <= 2 for br, bc, _ in blocks)
                if not near_block:
                    seeds.append((r, c))
    
    if seeds:
        queue = deque(seeds)
        for s in seeds:
            target_region.add(s)
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr,nc) not in target_region:
                    if grid[nr, nc] == target_bg:
                        target_region.add((nr, nc))
                        queue.append((nr, nc))
    
    if not target_region:
        return result
    
    # Find noise rows (rows with isolated source-color pixels in target region)
    noise_rows = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] in src_colors and (r, c) in target_region:
                noise_rows.add(r)
    
    # Determine path geometry
    top_row = min(r for r, c in target_region)
    
    if noise_rows:
        first_noise = min(noise_rows)
        row_cols = [c for r, c in target_region if r == first_noise]
        vert_col = max(row_cols) if row_cols else 0
    else:
        first_noise = top_row
        row_cols = [c for r, c in target_region if r == top_row]
        vert_col = max(row_cols) if row_cols else 0
    
    # Get pattern colors ordered by column
    ordered = [v for _, _, v in sorted(blocks, key=lambda x: x[1])]
    pat = []
    for v in ordered:
        if v not in pat:
            pat.append(v)
    P = len(pat)
    
    # Draw vertical segment
    for r in range(top_row, first_noise + 1):
        if (r, vert_col) in target_region:
            result[r, vert_col] = pat[r % P]
    
    # Draw horizontal at top
    top_cols = [c for r, c in target_region if r == top_row]
    if top_cols:
        for c in range(vert_col, max(top_cols) + 1):
            if (top_row, c) in target_region:
                result[top_row, c] = pat[c % P]
    
    # Draw horizontals at noise rows
    for nr in noise_rows:
        row_cols = [c for r, c in target_region if r == nr]
        if row_cols:
            for c in range(min(row_cols), vert_col + 1):
                if (nr, c) in target_region:
                    result[nr, c] = pat[c % P]
    
    return result