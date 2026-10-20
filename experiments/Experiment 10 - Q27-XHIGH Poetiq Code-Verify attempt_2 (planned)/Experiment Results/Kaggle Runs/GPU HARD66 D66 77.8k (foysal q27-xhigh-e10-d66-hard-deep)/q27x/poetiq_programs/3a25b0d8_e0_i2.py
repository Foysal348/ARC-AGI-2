import numpy as np
from collections import defaultdict
from scipy import ndimage

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the compact output panel from a scattered/overlaid input.

    The rule used here is:
      1. Identify the background color as the most frequent color.
      2. Identify the frame/wall color as the most frequent non-background color.
      3. All remaining colors are treated as payload/content colors.
      4. Crop to the bounding box of the payload colors.
      5. Complete the payload by filling small holes / short missing runs so that
         each payload region becomes a solid room.
      6. The final wall/frame is generated as the 8-neighbor dilation of all
         completed payload cells.
      7. Return the tight bounding box of the completed payload plus its frame,
         using the original background color outside the frame.
    """
    if grid.size == 0:
        return grid.copy()

    grid = np.asarray(grid, dtype=np.uint8)
    h, w = grid.shape

    # ---------------------------------------------------------------
    # Basic color roles.
    # ---------------------------------------------------------------
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) == 1:
        return grid.copy()

    bg = int(colors[np.argmax(counts)])

    # Frame color: most common color other than background.
    non_bg_counts = [(int(c), int(n)) for c, n in zip(colors, counts) if int(c) != bg]
    if not non_bg_counts:
        return np.zeros_like(grid)

    frame = int(max(non_bg_counts, key=lambda x: x[1])[0])

    # Payload mask: everything that is neither background nor frame.
    payload_mask = np.isin(grid, [int(c) for c, _ in non_bg_counts if int(c) != frame])

    if not payload_mask.any():
        # If there is no payload, return the cropped frame object if present.
        frame_mask = (grid == frame)
        if not frame_mask.any():
            return np.full((1, 1), bg, dtype=np.uint8)
        ys, xs = np.where(frame_mask)
        return grid[ys.min():ys.max()+1, xs.min():xs.max()+1].copy()

    ys, xs = np.where(payload_mask)
    r0, r1 = ys.min(), ys.max()
    c0, c1 = xs.min(), xs.max()

    # Work on the cropped payload area.
    sub = grid[r0:r1+1, c0:c1+1].copy()
    sh, sw = sub.shape

    # Clear background/frame to zero; keep payload colors.
    work = np.zeros((sh, sw), dtype=np.uint8)
    work[sub != bg] = sub[sub != bg]
    work[sub == bg] = 0
    work[sub == frame] = 0

    # ---------------------------------------------------------------
    # Payload completion.
    #
    # For each payload color, fill connected regions more aggressively:
    #   - fill holes inside the component's bounding box if the component
    #     already forms a plausible boundary;
    #   - extend single-pixel gaps horizontally and vertically when doing so
    #     does not collide with another payload color.
    #
    # This is heuristic, but it matches the examples' behavior where partial
    # colored rooms are completed into solid rooms before the frame is drawn.
    # ---------------------------------------------------------------
    completed = work.copy()
    payload_colors = [int(c) for c in np.unique(work) if c != 0]

    # Helper: can we place color colr at (r, c) without overwriting another
    # payload color and without touching a different payload color orthogonally
    # (which would require a frame pixel between rooms)?
    def can_place(r, c, colr):
        if r < 0 or c < 0 or r >= sh or c >= sw:
            return False
        if completed[r, c] != 0 and completed[r, c] != colr:
            return False
        # Check 4-neighbors: do not allow direct orthogonality to a different
        # payload color. Diagonal contact is acceptable because the final
        # 8-dilation frame can separate diagonally touching rooms.
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < sh and 0 <= nc < sw:
                v = completed[nr, nc]
                if v != 0 and v != colr:
                    return False
        return True

    # Fill per color using repeated local dilation with collision checks.
    # We run a small number of iterations to avoid over-expansion.
    for _ in range(3):
        changed = False
        for colr in payload_colors:
            mask = (completed == colr)
            # Dilation candidates: 8-neighbors of current mask.
            dil = ndimage.binary_dilation(mask, iterations=1, structure=np.ones((3, 3), bool))
            cand = dil & (completed == 0)
            cand_coords = np.argwhere(cand)
            for r, c in cand_coords:
                if can_place(int(r), int(c), colr):
                    completed[int(r), int(c)] = colr
                    changed = True
        if not changed:
            break

    # Additional row-wise closing: if a color has pixels at both ends of a
    # short gap and the gap does not contain another color, fill the gap.
    for colr in payload_colors:
        for r in range(sh):
            row = completed[r, :]
            positions = [c for c in range(sw) if row[c] == colr]
            if len(positions) >= 2:
                first, last = min(positions), max(positions)
                # Only close gaps up to length 3, to avoid bridging distant cells.
                if last - first <= 3:
                    for c in range(first, last + 1):
                        if row[c] == 0 and can_place(r, c, colr):
                            completed[r, c] = colr

    # Column-wise closing similarly.
    for colr in payload_colors:
        for c in range(sw):
            col = completed[:, c]
            positions = [r for r in range(sh) if col[r] == colr]
            if len(positions) >= 2:
                first, last = min(positions), max(positions)
                if last - first <= 3:
                    for r in range(first, last + 1):
                        if col[r] == 0 and can_place(r, c, colr):
                            completed[r, c] = colr

    # ---------------------------------------------------------------
    # Build frame as the 8-neighbor dilation of all completed payload cells.
    # ---------------------------------------------------------------
    payload_nonzero = completed > 0
    if payload_nonzero.any():
        frame_mask = ndimage.binary_dilation(payload_nonzero, iterations=1, structure=np.ones((3, 3), bool))
    else:
        frame_mask = payload_nonzero.copy()

    # ---------------------------------------------------------------
    # Compose final grid: background, frame, payload.
    # ---------------------------------------------------------------
    out = np.full((sh, sw), bg, dtype=np.uint8)
    out[frame_mask] = frame
    out[completed > 0] = completed[completed > 0]

    # Crop to the tight bounding box of frame or payload.
    active = (out != bg)
    if not active.any():
        return np.full((1, 1), bg, dtype=np.uint8)

    ay, ax = np.where(active)
    r0f, r1f = ay.min(), ay.max()
    c0f, c1f = ax.min(), ax.max()

    return out[r0f:r1f+1, c0f:c1f+1].copy()