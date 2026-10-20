import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Combines two frame objects separated by 7-marked dividers.
    
    - If separated vertically (7 rows): remove separator, stack objects
    - If separated horizontally (7 columns): overlay objects (4 wins over 1)
    """
    grid = grid.copy()
    rows, cols = grid.shape
    
    # Find rows/cols containing 7s
    rows_with_7 = set(np.where(grid == 7)[0].tolist())
    cols_with_7 = set(np.where(grid == 7)[1].tolist())
    
    # Determine split direction
    if len(rows_with_7) >= len(cols_with_7) and len(rows_with_7) > 0:
        # Vertical split - find separator band
        sorted_r = sorted(rows_with_7)
        # Find the full separator band (consecutive rows with 7s or all-1s between 7 rows)
        sep_start = sorted_r[0]
        sep_end = sorted_r[-1]
        # Expand to include all-1 rows between the 7 rows
        while sep_start > 0 and not (grid[sep_start-1] == 7).any() and (grid[sep_start-1] == 1).all():
            sep_start -= 1
        while sep_end < rows-1 and not (grid[sep_end+1] == 7).any() and (grid[sep_end+1] == 1).all():
            sep_end += 1
        
        # Two parts: top and bottom
        part1 = grid[:sep_start].copy()
        part2 = grid[sep_end+1:].copy()
        
        # Clean: replace 7s with 1s
        part1[part1 == 7] = 1
        part2[part2 == 7] = 1
        
        # Crop each to bounding box of 4s
        def crop_to_4s(part):
            mask = (part == 4)
            if not mask.any():
                return part
            r_idx = np.where(mask.any(axis=1))[0]
            c_idx = np.where(mask.any(axis=0))[0]
            return part[r_idx[0]:r_idx[-1]+1, c_idx[0]:c_idx[-1]+1]
        
        part1 = crop_to_4s(part1)
        part2 = crop_to_4s(part2)
        
        # Stack vertically (same width expected)
        return np.vstack([part1, part2])
    
    elif len(cols_with_7) > 0:
        # Horizontal split - find separator band
        sorted_c = sorted(cols_with_7)
        sep_start = sorted_c[0]
        sep_end = sorted_c[-1]
        while sep_start > 0 and not (grid[:, sep_start-1] == 7).any() and (grid[:, sep_start-1] == 1).all():
            sep_start -= 1
        while sep_end < cols-1 and not (grid[:, sep_end+1] == 7).any() and (grid[:, sep_end+1] == 1).all():
            sep_end += 1
        
        # Two parts: left and right
        part1 = grid[:, :sep_start].copy()
        part2 = grid[:, sep_end+1:].copy()
        
        # Clean: replace 7s with 1s
        part1[part1 == 7] = 1
        part2[part2 == 7] = 1
        
        # Crop each to bounding box of 4s
        def crop_to_4s(part):
            mask = (part == 4)
            if not mask.any():
                return part
            r_idx = np.where(mask.any(axis=1))[0]
            c_idx = np.where(mask.any(axis=0))[0]
            return part[r_idx[0]:r_idx[-1]+1, c_idx[0]:c_idx[-1]+1]
        
        part1 = crop_to_4s(part1)
        part2 = crop_to_4s(part2)
        
        # Overlay: make same size, 4 wins
        h = max(part1.shape[0], part2.shape[0])
        w = max(part1.shape[1], part2.shape[1])
        
        def pad(part, h, w):
            result = np.ones((h, w), dtype=part.dtype)
            result[:part.shape[0], :part.shape[1]] = part
            return result
        
        p1 = pad(part1, h, w)
        p2 = pad(part2, h, w)
        
        # Overlay: 4 wins over 1
        result = np.where((p1 == 4) | (p2 == 4), 4, 1)
        return result
    else:
        # No 7s found, return as-is with 7s replaced
        grid[grid == 7] = 1
        return grid