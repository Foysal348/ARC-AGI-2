import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a grid with 4 panels (separated by border-colored columns) where
    the rightmost panel is empty. Fill the empty panel by overlaying shapes from
    the other panels, with priority given to panels closer to the empty panel.
    
    The rule: for each cell in the empty panel, use the color from the nearest
    panel (rightmost first) that has a non-background pixel at that position.
    """
    grid = grid.copy()
    rows, cols = grid.shape
    
    # Find the border color (most common color in first row)
    border_color = grid[0, 0]
    
    # Find separator columns (columns where all rows have border_color)
    sep_cols = [c for c in range(cols) if np.all(grid[:, c] == border_color)]
    
    # Define panel content regions (between separators)
    panel_regions = []
    for i in range(len(sep_cols) - 1):
        c_start = sep_cols[i] + 1
        c_end = sep_cols[i + 1]  # exclusive
        panel_regions.append((c_start, c_end))
    
    # Find the empty panel (one with only background color)
    empty_panel_idx = -1
    for idx, (c_s, c_e) in enumerate(panel_regions):
        panel = grid[1:rows-1, c_s:c_e]
        unique_colors = set(panel.flatten().tolist())
        if len(unique_colors) == 1:
            empty_panel_idx = idx
    
    if empty_panel_idx == -1:
        return grid
    
    # Determine background color for each panel
    def get_background(panel):
        colors, counts = np.unique(panel, return_counts=True)
        return colors[np.argmax(counts)]
    
    # For each non-empty panel, determine which pixels are "shapes"
    # and build the output for the empty panel
    e_s, e_e = panel_regions[empty_panel_idx]
    empty_panel = grid[1:rows-1, e_s:e_e]
    bg_empty = get_background(empty_panel)
    
    # Process panels from right to left (closest to empty panel first)
    # Priority: panels closer to empty panel have higher priority
    result = np.full_like(empty_panel, bg_empty)
    
    # Get indices of non-empty panels, sorted by proximity to empty panel
    other_panels = [i for i in range(len(panel_regions)) if i != empty_panel_idx]
    # Sort by distance to empty panel (closest first)
    other_panels.sort(key=lambda x: abs(x - empty_panel_idx))
    
    for p_idx in other_panels:
        c_s, c_e = panel_regions[p_idx]
        panel = grid[1:rows-1, c_s:c_e]
        bg = get_background(panel)
        
        # Shape mask: where panel differs from its background
        shape_mask = (panel != bg)
        
        # Only fill positions in result that are still background
        # (i.e., not already filled by a higher-priority panel)
        fill_mask = shape_mask & (result == bg_empty)
        result[fill_mask] = panel[fill_mask]
    
    grid[1:rows-1, e_s:e_e] = result
    return grid