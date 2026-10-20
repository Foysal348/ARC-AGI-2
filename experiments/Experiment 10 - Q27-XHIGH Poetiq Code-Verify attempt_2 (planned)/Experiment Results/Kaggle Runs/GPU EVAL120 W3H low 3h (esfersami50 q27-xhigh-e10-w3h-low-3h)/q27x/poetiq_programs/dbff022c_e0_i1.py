import numpy as np
from collections import deque, defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform grid by filling enclosed regions based on a source block mapping.
    
    Algorithm:
    1. Find enclosed regions (0-cells not reachable from grid border).
    2. Identify the source block (rectangular block of non-zero, non-border pixels).
    3. Read the source block (col-by-col if cols>=rows, else row-by-row) and pair elements.
    4. For each region with border color C, if C is the first element of a pair (C,V), fill interior with V.
    """
    result = grid.copy()
    rows, cols = grid.shape
    
    # Step 1: Find holes (0-cells not reachable from border via 0-paths)
    visited = np.zeros((rows, cols), dtype=bool)
    queue = deque()
    
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1) and grid[r, c] == 0:
                queue.append((r, c))
                visited[r, c] = True
    
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and grid[nr, nc] == 0:
                visited[nr, nc] = True
                queue.append((nr, nc))
    
    holes = (grid == 0) & (~visited)
    
    if not holes.any():
        return result
    
    # Step 2: Determine border color for each hole and group into regions
    regions = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            if holes[r, c]:
                colors = set()
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != 0:
                        colors.add(int(grid[nr, nc]))
                if len(colors) == 1:
                    regions[colors.pop()].append((r, c))
                elif colors:
                    regions[list(colors)[0]].append((r, c))
    
    # Step 3: Identify border pixels and source block
    is_border = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and holes[nr, nc]:
                        is_border[r, c] = True
                        break
    
    source_mask = (grid != 0) & (~is_border)
    
    if not source_mask.any():
        return result
    
    # Find bounding box of source block
    source_positions = list(zip(*np.where(source_mask)))
    min_r = min(p[0] for p in source_positions)
    max_r = max(p[0] for p in source_positions)
    min_c = min(p[1] for p in source_positions)
    max_c = max(p[1] for p in source_positions)
    
    block_rows = max_r - min_r + 1
    block_cols = max_c - min_c + 1
    
    # Verify it's a solid rectangle
    block = grid[min_r:max_r + 1, min_c:max_c + 1]
    if (block == 0).any():
        return result  # Not a solid block, skip
    
    # Step 4: Read the block and create pairs
    if block_cols >= block_rows:
        # Column by column (top to bottom, left to right)
        sequence = [int(block[r, c]) for c in range(block_cols) for r in range(block_rows)]
    else:
        # Row by row (left to right, top to bottom)
        sequence = [int(block[r, c]) for r in range(block_rows) for c in range(block_cols)]
    
    # Create pairs
    mapping = {}
    for i in range(0, len(sequence) - 1, 2):
        key = sequence[i]
        value = sequence[i + 1]
        if key not in mapping:
            mapping[key] = value
    
    # Step 5: Fill regions based on mapping
    for border_color, hole_positions in regions.items():
        if border_color in mapping:
            fill_color = mapping[border_color]
            for (r, c) in hole_positions:
                result[r, c] = fill_color
    
    return result