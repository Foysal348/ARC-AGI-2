import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a 2D grid into a 1D array where each non-background color
    is repeated as many times as it appears in the grid.
    Colors are ordered by their first appearance from top to bottom.
    The background color is the most frequent color in the grid.
    """
    # Find the background color (most frequent)
    values, counts = np.unique(grid, return_counts=True)
    background_color = values[np.argmax(counts)]
    
    # For each non-background color, count pixels and find first row
    color_info = {}
    for color in values:
        if color == background_color:
            continue
        mask = (grid == color)
        count = np.sum(mask)
        rows = np.where(mask)[0]
        first_row = rows[0]
        color_info[color] = {'count': count, 'first_row': first_row}
    
    # Sort colors by first appearance row (top to bottom)
    sorted_colors = sorted(color_info.keys(), key=lambda c: color_info[c]['first_row'])
    
    # Build the output 1D array
    result = []
    for color in sorted_colors:
        result.extend([color] * color_info[color]['count'])
    
    return np.array(result)