import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transformation:
    1. Find the container shape (contains 4 special pixels in 2x2 arrangement)
    2. Find all other shapes
    3. Match shapes to special pixel colors by relative position
    4. Create output where each region is the transposed, scaled shape
    """
    H, W = grid.shape
    
    # Find all non-zero colors
    colors = set(grid[grid > 0].flatten().tolist())
    
    color_masks = {}
    for c in colors:
        color_masks[c] = (grid == c)
    
    def get_bounding_box(mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax
    
    # Find container: shape whose bounding box contains pixels of >= 4 other colors
    container_color = None
    special_pixels = {}
    
    for c in colors:
        mask = color_masks[c]
        rmin, rmax, cmin, cmax = get_bounding_box(mask)
        subgrid = grid[rmin:rmax+1, cmin:cmax+1]
        other_colors = set(subgrid[subgrid > 0].flatten().tolist()) - {c}
        if len(other_colors) >= 4:
            container_color = c
            for r in range(rmin, rmax+1):
                for col in range(cmin, cmax+1):
                    if grid[r, col] > 0 and grid[r, col] != c:
                        special_pixels[(r, col)] = grid[r, col]
            break
    
    # Determine special pixel quadrants
    sp_rows = sorted(set(r for r, c in special_pixels.keys()))
    sp_cols = sorted(set(c for r, c in special_pixels.keys()))
    top_r, bot_r = sp_rows[0], sp_rows[-1]
    left_c, right_c = sp_cols[0], sp_cols[-1]
    
    colors_by_quad = {}
    for (r, c), color in special_pixels.items():
        if r == top_r and c == left_c:
            colors_by_quad['tl'] = color
        elif r == top_r and c == right_c:
            colors_by_quad['tr'] = color
        elif r == bot_r and c == left_c:
            colors_by_quad['bl'] = color
        elif r == bot_r and c == right_c:
            colors_by_quad['br'] = color
    
    # Find connected components
    def find_largest_component(mask):
        visited = np.zeros_like(mask, dtype=bool)
        H, W = mask.shape
        largest = []
        for i in range(H):
            for j in range(W):
                if mask[i, j] and not visited[i, j]:
                    queue = deque([(i, j)])
                    visited[i, j] = True
                    comp = []
                    while queue:
                        r, c = queue.popleft()
                        comp.append((r, c))
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                    if len(comp) > len(largest):
                        largest = comp
        return largest
    
    # Get shapes
    container_mask = color_masks[container_color]
    container_pixels = find_largest_component(container_mask)
    container_center_r = sum(r for r,c in container_pixels) / len(container_pixels)
    container_center_c = sum(c for r,c in container_pixels) / len(container_pixels)
    
    # Assign other shapes to quadrants
    other_shapes = {}  # quadrant -> pixels
    for c in colors:
        if c == container_color:
            continue
        pixels = find_largest_component(color_masks[c])
        if not pixels:
            continue
        cr = sum(r for r,col in pixels) / len(pixels)
        cc = sum(col for r,col in pixels) / len(pixels)
        
        if cr <= container_center_r and cc <= container_center_c:
            q = 'tl'
        elif cr <= container_center_r and cc > container_center_c:
            q = 'tr'
        elif cr > container_center_r and cc <= container_center_c:
            q = 'bl'
        else:
            q = 'br'
        
        if q not in other_shapes or len(pixels) > len(other_shapes[q]):
            other_shapes[q] = pixels
    
    # Container maps to 'tl'
    # But wait - in example 2, the container IS in the upper-left, so it maps to 'tl'
    # Let's assign: container -> 'tl', and other shapes to their positional quadrants
    
    # Get shape patterns and transpose
    def get_pattern(pixels):
        if not pixels:
            return np.ones((1,1), dtype=int)
        rmin = min(r for r,c in pixels)
        rmax = max(r for r,c in pixels)
        cmin = min(c for r,c in pixels)
        cmax = max(c for r,c in pixels)
        h, w = rmax-rmin+1, cmax-cmin+1
        pat = np.zeros((h,w), dtype=int)
        for r,c in pixels:
            pat[r-rmin, c-cmin] = 1
        return pat.T  # transposed
    
    # Patterns for each quadrant
    quad_patterns = {}
    quad_patterns['tl'] = get_pattern(container_pixels)
    for q in ['tr', 'bl', 'br']:
        if q in other_shapes:
            quad_patterns[q] = get_pattern(other_shapes[q])
        else:
            quad_patterns[q] = np.ones((1,1), dtype=int)
    
    # Determine output size
    # Scale each pattern to fit in the output; output size = max dimension needed
    # Each pattern is scaled to the full output size
    max_dim = 1
    for pat in quad_patterns.values():
        max_dim = max(max_dim, pat.shape[0], pat.shape[1])
    
    # The output size seems to be determined by the overall bounding box
    all_pixels_list = [container_pixels]
    for q in ['tr','bl','br']:
        if q in other_shapes:
            all_pixels_list.append(other_shapes[q])
    
    all_r = [r for pixels in all_pixels_list for r,c in pixels]
    all_c = [c for pixels in all_pixels_list for r,c in pixels]
    output_size = max(max(all_r)-min(all_r), max(all_c)-min(all_c)) + 1
    
    # Create output by scaling each pattern
    output = np.zeros((output_size, output_size), dtype=int)
    
    # Priority: later quadrants overwrite earlier ones
    # From examples, it seems tl has highest priority in overlap areas
    # Let's try: br first, then bl, then tr, then tl (tl wins overlaps)
    for q in ['br', 'bl', 'tr', 'tl']:
        pat = quad_patterns[q]
        ph, pw = pat.shape
        color = colors_by_quad[q]
        
        for r in range(output_size):
            for c in range(output_size):
                pr = min(int(r * ph / output_size), ph - 1)
                pc = min(int(c * pw / output_size), pw - 1)
                if pat[pr, pc] == 1:
                    output[r, c] = color
    
    return output