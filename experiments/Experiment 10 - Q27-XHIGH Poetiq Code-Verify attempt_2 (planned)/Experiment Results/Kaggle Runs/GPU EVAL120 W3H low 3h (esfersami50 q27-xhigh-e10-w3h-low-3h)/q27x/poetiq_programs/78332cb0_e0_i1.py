import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a grid of blocks separated by lines of 6s.
    
    Rules:
    - If input is a vertical stack of blocks (Nx1), output is horizontal (1xN) with reversed order.
    - If input is a horizontal row of blocks (1xN), output is vertical (Nx1) with same order.
    - If input is a 2x2 grid of blocks, output is vertical (4x1) in order: TL, BR, TR, BL.
    """
    grid = np.array(grid, dtype=int)
    
    # Find row separators (rows where all values are 6)
    row_seps = [i for i in range(grid.shape[0]) if np.all(grid[i] == 6)]
    # Find column separators (columns where all values are 6)
    col_seps = [j for j in range(grid.shape[1]) if np.all(grid[:, j] == 6)]
    
    def get_segments(separators, total_size):
        """Get the (start, end) segments between separators."""
        segments = []
        prev = 0
        for sep in separators:
            segments.append((prev, sep))
            prev = sep + 1
        segments.append((prev, total_size))
        return segments
    
    row_segments = get_segments(row_seps, grid.shape[0])
    col_segments = get_segments(col_seps, grid.shape[1])
    
    # Extract blocks: blocks[i][j] is block at row-segment i, col-segment j
    blocks = []
    for r_start, r_end in row_segments:
        row_blocks = []
        for c_start, c_end in col_segments:
            row_blocks.append(grid[r_start:r_end, c_start:c_end].copy())
        blocks.append(row_blocks)
    
    n_row_blocks = len(row_segments)
    n_col_blocks = len(col_segments)
    
    # Determine output arrangement based on input shape
    if n_row_blocks == 1 and n_col_blocks > 1:
        # Horizontal (1xN) → Vertical (Nx1), same order
        output_blocks = [blocks[0][j] for j in range(n_col_blocks)]
        output_vertical = True
    elif n_row_blocks > 1 and n_col_blocks == 1:
        # Vertical (Nx1) → Horizontal (1xN), reversed order
        output_blocks = [blocks[i][0] for i in range(n_row_blocks - 1, -1, -1)]
        output_vertical = False
    elif n_row_blocks == 2 and n_col_blocks == 2:
        # 2x2 → Vertical (4x1), order: (0,0), (1,1), (0,1), (1,0)
        output_blocks = [blocks[0][0], blocks[1][1], blocks[0][1], blocks[1][0]]
        output_vertical = True
    else:
        # Fallback: row-major order, vertical output
        output_blocks = [blocks[i][j] for i in range(n_row_blocks) for j in range(n_col_blocks)]
        output_vertical = True
    
    # Build output grid
    if output_vertical:
        parts = []
        for i, block in enumerate(output_blocks):
            parts.append(block)
            if i < len(output_blocks) - 1:
                sep = np.full((1, block.shape[1]), 6, dtype=int)
                parts.append(sep)
        output = np.vstack(parts)
    else:
        parts = []
        for i, block in enumerate(output_blocks):
            parts.append(block)
            if i < len(output_blocks) - 1:
                sep = np.full((block.shape[0], 1), 6, dtype=int)
                parts.append(sep)
        output = np.hstack(parts)
    
    return output