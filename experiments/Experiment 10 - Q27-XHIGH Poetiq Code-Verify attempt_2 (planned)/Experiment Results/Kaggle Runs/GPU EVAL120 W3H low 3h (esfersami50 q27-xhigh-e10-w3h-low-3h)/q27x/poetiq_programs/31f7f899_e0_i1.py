import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Redistributes vertical line heights by sorting them in ascending order
    and reassigning from left to right, centered on the horizontal line row.
    
    Steps:
    1. Find the horizontal line (row with most 6s)
    2. Identify vertical lines (columns with non-8, non-6 values at the horizontal row)
    3. Measure each vertical line's height
    4. Sort heights ascending, assign to columns left-to-right
    5. Reconstruct grid with new heights centered on horizontal line
    """
    grid = grid.copy()
    rows, cols = grid.shape
    
    # Step 1: Find horizontal line (row with most 6s)
    six_counts = np.sum(grid == 6, axis=1)
    hline_row = int(np.argmax(six_counts))
    
    # Step 2: Identify vertical line columns
    vertical_cols = []
    for c in range(cols):
        val = grid[hline_row, c]
        if val != 8 and val != 6:
            vertical_cols.append((c, int(val)))
    
    if not vertical_cols:
        return grid
    
    # Step 3: Measure heights of each vertical line
    heights = []
    for c, color in vertical_cols:
        h = int(np.sum(grid[:, c] == color))
        heights.append(h)
    
    # Step 4: Sort heights and assign left to right
    sorted_heights = sorted(heights)
    
    # Step 5: Build output grid
    output = np.full((rows, cols), 8, dtype=int)
    
    # Place horizontal line
    output[hline_row, :] = grid[hline_row, :]
    
    # Draw vertical lines with new heights, centered on hline_row
    for i, (c, color) in enumerate(vertical_cols):
        new_h = sorted_heights[i]
        top = hline_row - (new_h - 1) // 2
        bottom = hline_row + new_h // 2
        for r in range(max(0, top), min(rows, bottom + 1)):
            output[r, c] = color
    
    return output