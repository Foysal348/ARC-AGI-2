import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform by extracting a shape from the first pattern region,
    repeating it N times (determined by second pattern region's pixel count / 2)
    with 1-unit gaps, filling with solid region colors.
    """
    grid = np.array(grid)
    
    # Find separator rows/columns (all 1s)
    vertical_seps = [j for j in range(grid.shape[1]) if np.all(grid[:, j] == 1)]
    horizontal_seps = [i for i in range(grid.shape[0]) if np.all(grid[i, :] == 1)]
    
    if len(vertical_seps) > 0:
        orientation = 'vertical'
        seps = vertical_seps
    elif len(horizontal_seps) > 0:
        orientation = 'horizontal'
        seps = horizontal_seps
    else:
        return grid.copy()
    
    # Split into regions
    regions = []
    if orientation == 'vertical':
        boundaries = [0] + seps + [grid.shape[1]]
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            if end - start > 0:
                regions.append(grid[:, start:end])
    else:
        boundaries = [0] + seps + [grid.shape[0]]
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            if end - start > 0:
                regions.append(grid[start:end, :])
    
    # Classify: pattern (has 0s) vs solid (no 0s)
    pattern_regions = []
    solid_regions = []
    for region in regions:
        if region.size > 0 and np.any(region == 0):
            pattern_regions.append(region)
        elif region.size > 0:
            solid_regions.append(region)
    
    # Extract parameters
    shape1_region = pattern_regions[0]
    shape2_region = pattern_regions[1]
    fill_color = int(solid_regions[0].flat[0])
    gap_color = int(solid_regions[1].flat[0])
    
    # Trim to bounding box
    def trim_to_bbox(region):
        nz = np.nonzero(region)
        if len(nz[0]) == 0:
            return region
        r_min, r_max = nz[0].min(), nz[0].max()
        c_min, c_max = nz[1].min(), nz[1].max()
        return region[r_min:r_max+1, c_min:c_max+1]
    
    shape1 = trim_to_bbox(shape1_region)
    shape2 = trim_to_bbox(shape2_region)
    
    # Binary mask of shape1
    mask1 = (shape1 != 0).astype(int)
    
    # Repetition count from shape2
    n = int(np.count_nonzero(shape2)) // 2
    
    h, w = mask1.shape
    
    # Construct output
    if orientation == 'vertical':
        out_h = h
        out_w = n * w + (n - 1)
        output = np.full((out_h, out_w), gap_color, dtype=int)
        for block in range(n):
            col_start = block * (w + 1)
            for r in range(h):
                for c in range(w):
                    if mask1[r, c] == 1:
                        output[r, col_start + c] = fill_color
    else:
        out_w = w
        out_h = n * h + (n - 1)
        output = np.full((out_h, out_w), gap_color, dtype=int)
        for block in range(n):
            row_start = block * (h + 1)
            for r in range(h):
                for c in range(w):
                    if mask1[r, c] == 1:
                        output[row_start + r, c] = fill_color
    
    return output