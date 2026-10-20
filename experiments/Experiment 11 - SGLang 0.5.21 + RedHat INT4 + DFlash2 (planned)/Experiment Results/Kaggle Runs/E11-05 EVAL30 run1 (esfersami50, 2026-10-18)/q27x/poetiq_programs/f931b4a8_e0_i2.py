import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a 2x2 composite ARC grid.

    The input is divided into four quadrants:
      top-left (TL), top-right (TR), bottom-left (BL), bottom-right (BR).

    - TL specifies the output height by its number of non-zero cells.
    - TR specifies the output width by its number of non-zero cells.
    - If one dimension indicator is empty, use the other dimension; if both
      are empty, use 1x1 as a safe fallback.
    - BL is stretched to the target size using nearest-neighbor sampling.
    - BR is tiled without scaling over the stretched background.  Zero cells
      in BR are transparent; non-zero cells overwrite the background.
    """
    grid = np.asarray(grid)

    if grid.ndim != 2 or grid.size == 0:
        return grid.copy()

    h, w = grid.shape
    mid_h = h // 2
    mid_w = w // 2

    # Need at least a 2x2 grid to form four quadrants.
    if mid_h == 0 or mid_w == 0:
        return grid.copy()

    tl = grid[:mid_h, :mid_w]
    tr = grid[:mid_h, mid_w:]
    bl = grid[mid_h:, :mid_w]
    br = grid[mid_h:, mid_w:]

    # Dimension indicators.
    out_h = int(np.count_nonzero(tl))
    out_w = int(np.count_nonzero(tr))

    # Fallbacks for empty dimension templates.
    if out_h <= 0:
        out_h = out_w if out_w > 0 else 1
    if out_w <= 0:
        out_w = out_h

    def nearest_resize(src: np.ndarray, rows: int, cols: int) -> np.ndarray:
        """Stretch src to rows x cols using nearest-neighbor sampling."""
        if rows == 0 or cols == 0:
            return np.empty((rows, cols), dtype=src.dtype)

        sh, sw = src.shape
        if sh == 0 or sw == 0:
            return np.zeros((rows, cols), dtype=src.dtype)

        r_idx = (np.arange(rows, dtype=np.int64) * sh) // rows
        c_idx = (np.arange(cols, dtype=np.int64) * sw) // cols
        return src[r_idx[:, None], c_idx[None, :]]

    def tile_crop(src: np.ndarray, rows: int, cols: int) -> np.ndarray:
        """Tile src to rows x cols and crop to the requested size."""
        if rows == 0 or cols == 0:
            return np.empty((rows, cols), dtype=src.dtype)

        sh, sw = src.shape
        if sh == 0 or sw == 0:
            return np.zeros((rows, cols), dtype=src.dtype)

        r_idx = np.arange(rows, dtype=np.int64) % sh
        c_idx = np.arange(cols, dtype=np.int64) % sw
        return src[r_idx[:, None], c_idx[None, :]]

    background = nearest_resize(bl, out_h, out_w)
    overlay = tile_crop(br, out_h, out_w)

    result = background.copy()
    result[overlay != 0] = overlay[overlay != 0]

    return result