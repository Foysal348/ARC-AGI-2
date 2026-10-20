import numpy as np
from scipy import ndimage

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform grid based on marker (color 4) orientation and position.
    Creates periodic pattern of 3s and 4s based on marker properties.
    """
    result = grid.copy()
    rows, cols = grid.shape
    
    # Find marker
    marker_positions = np.argwhere(grid == 4)
    if len(marker_positions) == 0:
        return result
    
    marker_rows = marker_positions[:, 0]
    marker_cols = marker_positions[:, 1]
    m_rmin, m_rmax = marker_rows.min(), marker_rows.max()
    m_cmin, m_cmax = marker_cols.min(), marker_cols.max()
    m_h = m_rmax - m_rmin + 1
    m_w = m_cmax - m_cmin + 1
    
    if m_w >= m_h:
        orientation = 'horizontal'
        m_start, m_end = m_cmin, m_cmax
    else:
        orientation = 'vertical'
        m_start, m_end = m_rmin, m_rmax
    
    # Find shape color
    bg = grid[0, 0]
    shape_color = None
    for c in np.unique(grid):
        if c != bg and c != 4:
            shape_color = c
            break
    
    if orientation == 'horizontal':
        for band in range((rows + 5) // 6):
            bs = band * 6
            be = min(bs + 6, rows)
            if bs >= rows:
                break
            top = bs
            
            # 3s columns: marker range + shape touching marker in band
            three = set(range(m_start, m_end + 1))
            mset = set(range(m_start, m_end + 1))
            
            # Shape mask in band
            smask = (grid[bs:be, :] == shape_color)
            labeled, n = ndimage.label(smask)
            
            for cid in range(1, n + 1):
                cpx = np.argwhere(labeled == cid)
                ccols = set(cpx[:, 1].tolist())
                # Component must touch marker range
                if ccols & mset:
                    three.update(ccols)
            
            for c in three:
                if c < cols:
                    result[top, c] = 3
            
            for off in [2, 4]:
                r = bs + off
                if r < rows:
                    for c in range(m_start, m_end + 1):
                        if c < cols:
                            result[r, c] = 4
    else:
        for band in range((cols + 5) // 6):
            bs = band * 6
            be = min(bs + 6, cols)
            if bs >= cols:
                break
            left = bs
            
            three = set(range(m_start, m_end + 1))
            mset = set(range(m_start, m_end + 1))
            
            smask = (grid[:, bs:be] == shape_color)
            labeled, n = ndimage.label(smask)
            
            for cid in range(1, n + 1):
                cpx = np.argwhere(labeled == cid)
                crows = set(cpx[:, 0].tolist())
                if crows & mset:
                    three.update(crows)
            
            for r in three:
                if r < rows:
                    result[r, left] = 3
            
            for off in [2, 4]:
                c = bs + off
                if c < cols:
                    for r in range(m_start, m_end + 1):
                        if r < rows:
                            result[r, c] = 4
    
    return result