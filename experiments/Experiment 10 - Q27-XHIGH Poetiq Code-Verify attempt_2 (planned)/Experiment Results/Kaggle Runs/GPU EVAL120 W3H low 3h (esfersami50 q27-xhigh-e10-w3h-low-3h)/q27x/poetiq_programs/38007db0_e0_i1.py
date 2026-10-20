import numpy as np
from collections import Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    The input is a grid of cells separated by uniform-color borders.
    In each row of cells, exactly one cell is different (unique) from the others.
    The output keeps only that unique cell per row, with borders around it.
    """
    # Find border color from corner
    border_color = grid[0, 0]
    
    # Find horizontal border rows (entirely border color)
    h_borders = [i for i in range(grid.shape[0]) if np.all(grid[i] == border_color)]
    
    # Find vertical border cols (entirely border color)
    v_borders = [j for j in range(grid.shape[1]) if np.all(grid[:, j] == border_color)]
    
    # Cell regions between consecutive borders
    cell_row_ranges = [(h_borders[i] + 1, h_borders[i + 1]) for i in range(len(h_borders) - 1)]
    cell_col_ranges = [(v_borders[j] + 1, v_borders[j + 1]) for j in range(len(v_borders) - 1)]
    
    num_rows = len(cell_row_ranges)
    num_cols = len(cell_col_ranges)
    
    cell_h = cell_row_ranges[0][1] - cell_row_ranges[0][0]
    cell_w = cell_col_ranges[0][1] - cell_col_ranges[0][0]
    
    # Construct output: same height, width = cell_w + 2 (border on each side)
    output = np.full((grid.shape[0], cell_w + 2), border_color, dtype=grid.dtype)
    
    for r in range(num_rows):
        # Extract all cells in this row
        cells = []
        for c in range(num_cols):
            cell = grid[cell_row_ranges[r][0]:cell_row_ranges[r][1],
                        cell_col_ranges[c][0]:cell_col_ranges[c][1]]
            cells.append(cell)
        
        # Find the unique cell (appears only once)
        cell_tuples = [tuple(cell.flatten()) for cell in cells]
        counts = Counter(cell_tuples)
        
        unique_cell = None
        for c in range(num_cols):
            if counts[cell_tuples[c]] == 1:
                unique_cell = cells[c]
                break
        
        # If no unique found (all same), take the first as fallback
        if unique_cell is None:
            unique_cell = cells[0]
        
        output[cell_row_ranges[r][0]:cell_row_ranges[r][1], 1:1 + cell_w] = unique_cell
    
    return output