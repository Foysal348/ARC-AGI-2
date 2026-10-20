import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """Transform horizontal stripes into concentric square borders.
    
    The input is a grid where each row is a uniform color. Consecutive rows
    with the same color form 'stripes'. The output is a square grid where
    these stripes become concentric square borders from outside to inside.
    The last stripe becomes the center square.
    """
    # Extract stripes from input (consecutive rows with same color)
    heights = []
    colors = []
    for row in grid:
        color = int(row[0])
        if colors and colors[-1] == color:
            heights[-1] += 1
        else:
            colors.append(color)
            heights.append(1)
    
    n = len(heights)
    # Output size: all stripes except the last appear on both sides,
    # the last stripe appears only in the center
    output_size = 2 * sum(heights[:-1]) + heights[-1]
    
    # Build output by filling rectangles from outside to inside
    # Each inner fill overwrites the center of the previous fill
    output = np.zeros((output_size, output_size), dtype=int)
    
    offset = 0
    for i in range(n):
        thickness = heights[i]
        # Fill the current rectangle (will be partially overwritten by next)
        output[offset:output_size - offset, offset:output_size - offset] = colors[i]
        offset += thickness
    
    return output