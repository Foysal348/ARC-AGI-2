import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Paint 7s in a 3x3 ring around target 0-pixels.
    
    A target is a 0-pixel where:
    - All existing orthogonal neighbors are 1, OR
    - It has exactly one 0-orthogonal-neighbor that itself has >=2 orthogonal 1-neighbors
    
    When painting the 3x3 ring, 0-pixels with >=2 orthogonal 1-neighbors are protected (stay 0).
    """
    rows, cols = grid.shape
    output = grid.copy()
    
    def count_1_neighbors(r, c):
        count = 0
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr, nc] == 1:
                    count += 1
        return count
    
    def is_target(r, c):
        if grid[r, c] != 0:
            return False
        n1 = count_1_neighbors(r, c)
        # Count existing neighbors
        existing = 0
        zero_neighbors = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols:
                existing += 1
                if grid[nr, nc] == 0:
                    zero_neighbors.append((nr, nc))
        
        # All existing neighbors are 1
        if n1 == existing:
            return True
        
        # Exactly one 0-neighbor, and that neighbor has >=2 orthogonal 1-neighbors
        if len(zero_neighbors) == 1:
            zr, zc = zero_neighbors[0]
            if count_1_neighbors(zr, zc) >= 2:
                return True
        
        return False
    
    # Find all targets
    targets = []
    for r in range(rows):
        for c in range(cols):
            if is_target(r, c):
                targets.append((r, c))
    
    # Paint 3x3 ring around each target
    for r, c in targets:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if grid[nr, nc] == 1:
                        output[nr, nc] = 7
                    elif grid[nr, nc] == 0:
                        # Protected if >=2 orthogonal neighbors are 1
                        if count_1_neighbors(nr, nc) < 2:
                            output[nr, nc] = 7
    
    return output