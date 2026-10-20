import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a grid containing a cross/plus shape into a fractal pattern.
    
    Uses iterative XOR-based 2D cellular automaton with a cross-shaped 
    neighborhood (Moore cross: center + 4 neighbors) to generate the 
    self-similar fractal output pattern.
    """
    H, W = grid.shape
    
    # Initialize with the input pattern
    state = (grid == 8).astype(np.int32)
    
    # Determine number of iterations needed to fill the grid
    positions = np.argwhere(grid == 8)
    center_r = int(positions[:, 0].mean())
    center_c = int(positions[:, 1].mean())
    max_dist = max(center_r, H - 1 - center_r, center_c, W - 1 - center_c)
    
    # Iteratively apply XOR rule: new[r,c] = XOR of {center, up, down, left, right}
    for _ in range(max_dist):
        new_state = state.copy()  # center
        new_state[1:, :] ^= state[:-1, :]   # XOR with up neighbor
        new_state[:-1, :] ^= state[1:, :]   # XOR with down neighbor
        new_state[:, 1:] ^= state[:, :-1]   # XOR with left neighbor
        new_state[:, :-1] ^= state[:, 1:]   # XOR with right neighbor
        state = new_state
    
    return state * 8