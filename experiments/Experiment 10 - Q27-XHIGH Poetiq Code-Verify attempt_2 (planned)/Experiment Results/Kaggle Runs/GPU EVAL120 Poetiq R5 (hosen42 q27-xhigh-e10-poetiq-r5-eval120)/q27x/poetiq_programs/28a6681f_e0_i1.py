import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by simulating all 1 pixels as sand grains under gravity.

    Rules:
      - Pixels with values other than 0 and 1 are static walls.
      - 1 pixels fall downward when possible.
      - If a 1 pixel cannot fall, it may slide left (preferred) or right,
        but only if that slide leads to a drop in the same row before
        hitting another wall or grain.
      - This preserves the number of 1 pixels and fills lower cavities,
        matching the examples.
    """
    g = np.array(grid, dtype=int, copy=True)
    H, W = g.shape

    if H == 0 or W == 0:
        return g

    def has_left_fall(r, c):
        """
        Return True if, from empty cell (r, c), moving left along row r
        reaches a cell with an empty cell below it.
        """
        x = c
        while 0 <= x and g[r, x] == 0:
            if r + 1 < H and g[r + 1, x] == 0:
                return True
            if x == 0 or g[r, x - 1] != 0:
                break
            x -= 1
        return False

    def has_right_fall(r, c):
        """
        Return True if, from empty cell (r, c), moving right along row r
        reaches a cell with an empty cell below it.
        """
        x = c
        while x < W and g[r, x] == 0:
            if r + 1 < H and g[r + 1, x] == 0:
                return True
            if x == W - 1 or g[r, x + 1] != 0:
                break
            x += 1
        return False

    # Enough steps for ARC-sized grids; prevents accidental infinite loops.
    max_steps = H * W * max(H, W) * 20 + 1000
    steps = 0

    while steps < max_steps:
        move = None

        # 1) Prioritize downward movement.
        for r in range(H - 1, -1, -1):
            for c in range(W):
                if g[r, c] == 1 and r + 1 < H and g[r + 1, c] == 0:
                    move = (r, c, r + 1, c)
                    break
            if move is not None:
                break

        # 2) If no grain can fall, let a grain slide left/right toward a drop.
        if move is None:
            for r in range(H - 1, -1, -1):
                for c in range(W):
                    if g[r, c] != 1:
                        continue

                    # Prefer left.
                    if (
                        c > 0
                        and g[r, c - 1] == 0
                        and has_left_fall(r, c - 1)
                    ):
                        move = (r, c, r, c - 1)
                        break

                    # Then right.
                    if (
                        c + 1 < W
                        and g[r, c + 1] == 0
                        and has_right_fall(r, c + 1)
                    ):
                        move = (r, c, r, c + 1)
                        break

                if move is not None:
                    break

        if move is None:
            break

        r, c, nr, nc = move
        g[r, c] = 0
        g[nr, nc] = 1
        steps += 1

    return g