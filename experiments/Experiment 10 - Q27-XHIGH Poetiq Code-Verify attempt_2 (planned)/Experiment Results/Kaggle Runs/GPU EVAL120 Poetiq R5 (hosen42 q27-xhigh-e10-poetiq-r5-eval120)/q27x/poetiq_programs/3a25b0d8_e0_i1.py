import numpy as np
from collections import defaultdict
from scipy import ndimage

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the compact object from a scattered/corrupted grid.

    The implementation assumes the following ARC-style structure:
      * One color is the global background.
      * One color acts as a frame/wall color.
      * Other colors are payload colors inside/near the frame.
    The output is a cleaned, compact crop of the main payload-bearing frame
    component.  Small payload intrusions that are not supported by horizontal
    runs are removed, and simple missing payload rows are filled by vertical
    projection.
    """
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape
    colors = np.unique(grid)

    # If the grid is already tiny, return it as-is to avoid degenerate behavior.
    if h <= 2 or w <= 2:
        return grid.copy()

    # Count colors.
    flat = grid.ravel()
    vals, counts = np.unique(flat, return_counts=True)
    color_counts = dict(zip(vals.tolist(), counts.tolist()))

    # Background = most frequent color.
    bg = max(color_counts, key=lambda c: (color_counts[c], -c))

    # Frame = most frequent non-background color that appears often.
    non_bg = [c for c in color_counts if c != bg]
    if not non_bg:
        return grid.copy()

    # Choose the second most common color as frame.
    frame = max(non_bg, key=lambda c: (color_counts[c], -c))

    payload_colors = [c for c in non_bg if c != frame]

    # If there are no payload colors, just crop the largest frame component.
    def crop_component(comp_mask):
        if not comp_mask.any():
            return np.zeros_like(grid, dtype=grid.dtype)
        rows = np.any(comp_mask, axis=1)
        cols = np.any(comp_mask, axis=0)
        r0, r1 = np.argmax(rows), h - np.argmax(rows[::-1])
        c0, c1 = np.argmax(cols), w - np.argmax(cols[::-1])
        return grid[r0:r1, c0:c1].copy()

    # Build mask of main structure: frame cells plus payload cells near frame.
    if not payload_colors:
        frame_mask = (grid == frame)
        lab, n = ndimage.label(frame_mask, structure=np.ones((3, 3), dtype=bool))
        if n == 0:
            return grid.copy()
        # Largest frame component.
        sizes = ndimage.sum(frame_mask, lab, range(1, n + 1))
        best = int(np.argmax(sizes)) + 1
        return crop_component(lab == best)

    payload_mask = np.isin(grid, payload_colors)

    # Find frame components and score them by nearby payload content.
    frame_mask = (grid == frame)
    lab, n = ndimage.label(frame_mask, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        # No frame component; fall back to payload bounding box.
        rows = np.any(payload_mask, axis=1)
        cols = np.any(payload_mask, axis=0)
        if not rows.any() or not cols.any():
            return grid.copy()
        r0, r1 = np.argmax(rows), h - np.argmax(rows[::-1])
        c0, c1 = np.argmax(cols), w - np.argmax(cols[::-1])
        return grid[r0:r1, c0:c1].copy()

    best_score = -1
    best_mask = None
    struct = np.ones((3, 3), dtype=bool)

    # Dilate payload mask to associate payloads with frame components.
    payload_dilated = ndimage.binary_dilation(payload_mask, structure=struct, iterations=3)

    for i in range(1, n + 1):
        comp = lab == i
        # Score: number of distinct payload colors within/near component.
        region = comp & payload_dilated
        distinct = len(np.unique(grid[region & payload_mask]))
        payload_pixels = int(np.sum(region & payload_mask))
        frame_pixels = int(np.sum(comp))
        # Prefer components that contain many payload colors and pixels.
        score = 1000 * distinct + payload_pixels + 0.01 * frame_pixels
        if score > best_score:
            best_score = score
            best_mask = comp

    if best_mask is None:
        best_mask = lab == 1

    # Expand best frame component to include adjacent payload pixels.
    mask = ndimage.binary_dilation(best_mask, structure=struct, iterations=2)
    mask |= payload_mask & ndimage.binary_dilation(best_mask, structure=struct, iterations=4)

    # Crop to this main component.
    out = crop_component(mask)

    # Now clean the cropped output.
    if out.size == 0:
        return out

    ch, cw = out.shape

    # Remove payload pixels that are isolated horizontally and vertically,
    # but only if a stronger horizontal run of the same color exists nearby.
    cleaned = out.copy()

    for c in payload_colors:
        c_mask = (out == c)
        if not c_mask.any():
            continue

        # Find rows with long horizontal runs of this color.
        row_has_long_run = np.zeros(ch, dtype=bool)
        for r in range(ch):
            row = c_mask[r]
            if not row.any():
                continue
            # longest run in this row
            best_run = 0
            run = 0
            for v in row:
                if v:
                    run += 1
                    best_run = max(best_run, run)
                else:
                    run = 0
            if best_run >= max(3, int(0.6 * cw)):
                row_has_long_run[r] = True

        if not row_has_long_run.any():
            continue

        # Rows far from a strong run can be pruned if they only have isolated pixels.
        for r in range(ch):
            row = c_mask[r]
            if not row.any() or row_has_long_run[r]:
                continue
            # Count horizontal run lengths in this row.
            runs = []
            run = 0
            for v in row:
                if v:
                    run += 1
                else:
                    if run:
                        runs.append(run)
                    run = 0
            if run:
                runs.append(run)

            # If this row has only short runs and is vertically near a strong row,
            # remove it if it is above a strong row (intrusion into a border row).
            max_run = max(runs) if runs else 0
            if max_run <= 1:
                # Check vertical proximity to a strong row.
                near_strong = any(row_has_long_run[max(0, r - 1):r + 2])
                if near_strong:
                    cleaned[r, c_mask[r]] = (out == frame) if frame in color_counts else bg

    # Simple vertical fill: if a row has a single payload pixel centered under a
    # full horizontal run of the same color, fill that row to the same span.
    for r in range(ch - 1, 0, -1):
        for c in payload_colors:
            cur = (cleaned[r] == c)
            if cur.sum() != 1:
                continue
            col = int(np.where(cur)[0][0])
            above = (cleaned[r - 1] == c)
            if not above.any():
                continue
            # Find contiguous runs in above row.
            idx = np.where(above)[0]
            runs = []
            start = idx[0]
            prev = idx[0]
            for x in idx[1:]:
                if x == prev + 1:
                    prev = x
                else:
                    runs.append((start, prev))
                    start = x
                    prev = x
            runs.append((start, prev))

            # If the single pixel is inside a run above, fill that run.
            for a, b in runs:
                if a <= col <= b and (b - a + 1) >= 3:
                    cleaned[r, a:b + 1] = c
                    break

    # Fill obvious missing bottom copies for solid 3-wide blocks near a frame row.
    for r in range(ch - 2, -1, -1):
        for c in payload_colors:
            cur = (cleaned[r] == c)
            if cur.sum() < 3:
                continue
            idx = np.where(cur)[0]
            if idx[-1] - idx[0] + 1 != cur.sum():
                continue
            # If next row is mostly frame/background and not another payload row,
            # copy the solid row downward once.
            below = cleaned[r + 1]
            below_payload = np.isin(below, payload_colors).sum()
            if below_payload == 0 and r + 1 < ch:
                # Only if there is vertical room and the row below is not part of another module.
                cleaned[r + 1, idx[0]:idx[-1] + 1] = c

    return cleaned