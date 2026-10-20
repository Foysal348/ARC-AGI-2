import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transformation rule:
    1. Clear the left portion (columns 0-8) to zeros.
    2. In the right box interior (columns 10-18, rows 1-6), apply vertical sorting:
       - Non-0, non-5 values ('numbers') float to the top of each column.
       - 5s fill the space from just below the numbers down to the floor (row 5).
       - 0s occupy the remaining space at the bottom.
    """
    result = grid.copy()
    rows, cols = result.shape
    
    # Step 1: Clear the left side (columns 0 through 8)
    result[:, 0:9] = 0
    
    # Step 2: Process each column in the right box interior
    # Interior: columns 10-18, rows 1-6 (6 rows tall)
    # Floor is at grid row 5, which is index 4 in the 0-based interior array
    
    for col in range(10, 19):
        # Extract interior column (rows 1-6)
        interior = result[1:7, col].astype(int).copy()
        
        # Collect numbers (non-0, non-5) preserving top-to-bottom order
        numbers = [interior[i] for i in range(6) if interior[i] not in (0, 5)]
        num_count = len(numbers)
        
        # Construct new column
        new_col = np.zeros(6, dtype=result.dtype)
        
        # Place numbers at the top
        for i in range(num_count):
            new_col[i] = numbers[i]
        
        # Fill 5s from below the numbers down to the floor level (index 4)
        for i in range(num_count, 5):
            new_col[i] = 5
        
        # Index 5 (grid row 6) remains 0
        
        result[1:7, col] = new_col
    
    return result