import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Solve the observed ARC transformation.

    The input contains:
      * one "template" object that contains a small colored pattern
        (a 2x2 block in the examples),
      * three other monochromatic objects.

    The output is a square whose area is the total number of non-zero cells
    in those four objects. The four objects are packed into the square:
      * the template object goes to the top-left corner,
      * the other three objects are rotated by multiples of 90 degrees and
        placed in the other three corners,
      * the four output colors are taken from the template's small pattern.

    The implementation extracts the relevant masks, tries all assignments and
    rotations of the three corner pieces, and returns the first exact tiling.
    """

    def connected_components_nonzero(g):
        """Return all connected non-zero components as binary masks."""
        H, W = g.shape
        visited = np.zeros((H, W), dtype=bool)
        comps = []
        for i in range(H):
            for j in range(W):
                if g[i, j] != 0 and not visited[i, j]:
                    q = deque([(i, j)])
                    visited[i, j] = True
                    mask = np.zeros((H, W), dtype=bool)
                    mask[i, j] = True
                    while q:
                        x, y = q.popleft()
                        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < H and 0 <= ny < W and not visited[nx, ny] and g[nx, ny] != 0:
                                visited[nx, ny] = True
                                mask[nx, ny] = True
                                q.append((nx, ny))
                    comps.append(mask)
        return comps

    def mask_bbox(m):
        """Crop a binary mask to its bounding box."""
        rows = np.any(m, axis=1)
        cols = np.any(m, axis=0)
        if not rows.any() or not cols.any():
            return np.zeros((0, 0), dtype=bool)
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        return m[r0:r1 + 1, c0:c1 + 1].astype(bool)

    def all_rotations(mask):
        """Return the 4 rotations of a binary mask."""
        res = []
        m = mask
        for _ in range(4):
            res.append(m.copy())
            m = m.T[::-1, :]
        return res

    def extract_template_colors(template_mask, template_grid):
        """
        Find the small pattern of non-main colors inside the template object.
        In the examples it is a 2x2 block.  We find the bounding box of all
        cells inside the template component that are not the dominant color.
        """
        vals = template_grid[template_mask]
        vals = vals[vals != 0]
        if len(vals) == 0:
            return None

        # Dominant color in the template component is considered the frame color.
        vals, counts = np.unique(vals, return_counts=True)
        main_color = vals[np.argmax(counts)]

        non_main = template_mask & (template_grid != main_color)
        if not non_main.any():
            return None

        rows = np.any(non_main, axis=1)
        cols = np.any(non_main, axis=0)
        if not rows.any() or not cols.any():
            return None
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]

        pattern = template_grid[r0:r1 + 1, c0:c1 + 1].copy()
        pattern[~template_mask[r0:r1 + 1, c0:c1 + 1]] = 0
        pattern[pattern == main_color] = 0

        # For the observed task, the pattern is 2x2.
        if pattern.shape != (2, 2):
            return None
        colors = pattern.reshape(-1)
        if len(set(colors.tolist())) != 4:
            return None
        return pattern

    def place_corner(out, mask, corner):
        """Place a mask into an output grid at the requested corner."""
        H, W = out.shape
        h, w = mask.shape
        if h > H or w > W:
            return False

        if corner == 'TL':
            r0, c0 = 0, 0
        elif corner == 'TR':
            r0, c0 = 0, W - w
        elif corner == 'BL':
            r0, c0 = H - h, 0
        else:  # BR
            r0, c0 = H - h, W - w

        out[r0:r0 + h, c0:c0 + w] |= mask
        return True

    if not isinstance(grid, np.ndarray):
        return np.asarray(grid, dtype=np.uint8)

    g = np.asarray(grid)

    # Basic edge case: no objects.
    if not g.any():
        return np.zeros_like(g, dtype=np.uint8)

    components = connected_components_nonzero(g)
    if len(components) < 4:
        # Fallback: return zeros for unexpected malformed inputs.
        return np.zeros((0, 0), dtype=np.uint8)

    # Identify the template component: the component whose cropped area
    # contains more than one distinct non-zero color.
    template_mask = None
    for m in components:
        sub = g[m]
        if len(np.unique(sub)) > 1:
            template_mask = m
            break

    if template_mask is None:
        return np.zeros((0, 0), dtype=np.uint8)

    pattern = extract_template_colors(template_mask, g)
    if pattern is None:
        return np.zeros((0, 0), dtype=np.uint8)

    tl_color, tr_color, bl_color, br_color = pattern.reshape(4).tolist()

    # Template output mask is the whole non-zero template component.
    template_out = mask_bbox(template_mask)

    # The other three pieces.
    pieces = [mask_bbox(m) for m in components if m is not template_mask]
    if len(pieces) != 3:
        return np.zeros((0, 0), dtype=np.uint8)

    total_area = int(template_out.sum()) + sum(int(p.sum()) for p in pieces)
    side = int(round(total_area ** 0.5))
    if side * side != total_area:
        return np.zeros((0, 0), dtype=np.uint8)

    # Try all assignments of the three pieces to TR/BL/BR and all rotations.
    corners = ['TR', 'BL', 'BR']
    corner_colors = {
        'TL': tl_color,
        'TR': tr_color,
        'BL': bl_color,
        'BR': br_color,
    }

    import itertools

    for perm in itertools.permutations(range(3)):
        # perm[i] is the piece assigned to corners[i]
        rotated = [all_rotations(pieces[i]) for i in range(3)]

        for r0 in itertools.product(range(4), repeat=3):
            base = np.zeros((side, side), dtype=bool)

            # Template fixed at top-left.
            base[0:template_out.shape[0], 0:template_out.shape[1]] = template_out

            ok = True
            placements = {}

            for idx, corner in enumerate(corners):
                m = rotated[idx][r0[idx]]
                h, w = m.shape

                if corner == 'TR':
                    r, c = 0, side - w
                elif corner == 'BL':
                    r, c = side - h, 0
                else:
                    r, c = side - h, side - w

                block = np.zeros((side, side), dtype=bool)
                block[r:r + h, c:c + w] = m

                # Must not overlap already placed masks.
                if np.any(block & base):
                    ok = False
                    break

                base |= block
                placements[corner] = block

            if not ok:
                continue

            # Must tile the whole square.
            if base.sum() != side * side:
                continue

            out = np.zeros((side, side), dtype=np.uint8)
            out[base & (np.zeros((side, side), dtype=bool) | (True))] = 0  # no-op

            # Paint template region.
            tl_block = np.zeros((side, side), dtype=bool)
            tl_block[0:template_out.shape[0], 0:template_out.shape[1]] = template_out
            out[tl_block] = tl_color

            for corner in corners:
                out[placements[corner]] = corner_colors[corner]

            return out

    # If no packing is found, return an empty array as a graceful fallback.
    return np.zeros((0, 0), dtype=np.uint8)