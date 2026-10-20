import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """Transform the grid by generating patterns on the empty side of a vertical separator.
    
    The separator is a column where all cells have the same non-zero value.
    For each row, the pattern on the non-object side is generated based on the object side:
    - Primary color: the value closest to the separator on the object side
    - Primary appears at pattern positions where j % count_primary == 0
    - Other colors c appear at positions where j % (count_primary * count_c) == count_c
    - All other positions are 0
    """
    grid = grid.copy()
    rows, cols = grid.shape
    
    # Step 1: Find the separator column
    separator_col = None
    separator_color = None
    for c in range(cols):
        col_values = grid[:, c]
        if np.all(col_values == col_values[0]) and col_values[0] != 0:
            separator_col = c
            separator_color = int(col_values[0])
            break
    
    if separator_col is None:
        return grid
    
    right_start = separator_col + 1
    right_width = cols - right_start
    
    # Step 2: Determine which side has objects
    left_obj_count = 0
    right_obj_count = 0
    for r in range(rows):
        for c in range(separator_col):
            if grid[r, c] != 0 and grid[r, c] != separator_color:
                left_obj_count += 1
        for c in range(right_start, cols):
            if grid[r, c] != 0 and grid[r, c] != separator_color:
                right_obj_count += 1
    
    if left_obj_count > 0 and right_obj_count == 0:
        generate_side = 'right'
    elif right_obj_count > 0 and left_obj_count == 0:
        generate_side = 'left'
    elif left_obj_count > 0:
        generate_side = 'right'
    else:
        return grid
    
    # Step 3: Generate pattern for each row
    for r in range(rows):
        if generate_side == 'right':
            obj = grid[r, :separator_col]
            primary_color = None
            for c in range(separator_col - 1, -1, -1):
                if obj[c] != 0 and obj[c] != separator_color:
                    primary_color = int(obj[c])
                    break
            if primary_color is None:
                grid[r, right_start:] = 0
                continue
            count_primary = int(np.sum(obj == primary_color))
            other_colors = {}
            for c in range(separator_col):
                val = int(obj[c])
                if val != 0 and val != primary_color:
                    other_colors[val] = other_colors.get(val, 0) + 1
            for j in range(right_width):
                value = 0
                if j % count_primary == 0:
                    value = primary_color
                else:
                    for color, count in other_colors.items():
                        if j % (count_primary * count) == count:
                            value = color
                            break
                grid[r, right_start + j] = value
        else:
            obj = grid[r, right_start:]
            primary_color = None
            for c in range(right_width):
                if obj[c] != 0 and obj[c] != separator_color:
                    primary_color = int(obj[c])
                    break
            if primary_color is None:
                grid[r, :separator_col] = 0
                continue
            count_primary = int(np.sum(obj == primary_color))
            other_colors = {}
            for c in range(right_width):
                val = int(obj[c])
                if val != 0 and val != primary_color:
                    other_colors[val] = other_colors.get(val, 0) + 1
            for j in range(separator_col):
                value = 0
                if j % count_primary == 0:
                    value = primary_color
                else:
                    for color, count in other_colors.items():
                        if j % (count_primary * count) == count:
                            value = color
                            break
                grid[r, separator_col - 1 - j] = value
    
    return grid