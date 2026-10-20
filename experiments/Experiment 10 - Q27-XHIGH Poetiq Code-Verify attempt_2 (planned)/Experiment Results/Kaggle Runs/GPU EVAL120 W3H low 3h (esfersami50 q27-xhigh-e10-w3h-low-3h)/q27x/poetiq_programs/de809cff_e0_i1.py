import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    For each interior 0 (hole) pixel within a colored region:
    - Set the pixel to 8
    - Set all 8 neighbors that were the region's color to the other region's color
    
    Region membership is determined by the dominant non-zero color among the 8 neighbors.
    """
    result = grid.copy()
    rows, cols = grid.shape
    
    # Find the two main non-zero colors
    non_zero = grid[grid > 0]
    if len(non_zero) == 0:
        return result
    
    unique_colors = [c for c in np.unique(non_zero) if c != 8]
    
    if len(unique_colors) < 2:
        return result
    
    color_counts = [(c, np.sum(grid == c)) for c in unique_colors]
    color_counts.sort(key=lambda x: -x[1])
    
    color_a, color_b = int(color_counts[0][0]), int(color_counts[1][0])
    
    # Process each 0 pixel
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0:
                continue
            
            # Gather neighbor info
            count_a = 0
            count_b = 0
            neighbors = []
            
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        val = int(grid[nr, nc])
                        neighbors.append((nr, nc, val))
                        if val == color_a:
                            count_a += 1
                        elif val == color_b:
                            count_b += 1
            
            total = count_a + count_b
            if total == 0:
                continue
            
            # Determine region - must be majority
            if count_a > count_b:
                region_color = color_a
                other_color = color_b
            elif count_b > count_a:
                region_color = color_b
                other_color = color_a
            else:
                continue  # tie, skip
            
            # Set center to 8
            result[r, c] = 8
            
            # Change neighbors from region_color to other_color
            for nr, nc, val in neighbors:
                if val == region_color:
                    result[nr, nc] = other_color
    
    return result