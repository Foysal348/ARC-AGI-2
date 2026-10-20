import numpy as np
from collections import deque

def _neighbors4(n, m):
    for r in range(n):
        for c in range(m):
            if r + 1 < n:
                yield (r, c), (r + 1, c)
            if c + 1 < m:
                yield (r, c), (r, c + 1)

def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background holes that are not connected to the border."""
    if mask.size == 0:
        return mask.copy()
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    q = deque()

    for r in range(h):
        for c in range(w):
            if mask[r, c] == 0 and (r == 0 or r == h - 1 or c == 0 or c == w - 1):
                q.append((r, c))
                visited[r, c] = True

    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and mask[nr, nc] == 0:
                visited[nr, nc] = True
                q.append((nr, nc))

    out = mask.copy()
    out[~visited & (mask == 0)] = 1
    return out

def _component_mask(grid: np.ndarray, r0: int, c0: int, color: int) -> np.ndarray:
    h, w = grid.shape
    if grid[r0, c0] != color:
        return np.zeros_like(grid, dtype=bool)
    seen = np.zeros_like(grid, dtype=bool)
    q = deque([(r0, c0)])
    seen[r0, c0] = True
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == color:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen

def _crop_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return mask[0:1, 0:1]
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def _rotate(mask: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(mask, k)

def _resize_nearest(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    if h <= 0 or w <= 0:
        return np.zeros((h, w), dtype=bool)
    if mask.size == 0:
        return np.zeros((h, w), dtype=bool)
    H, W = mask.shape
    rows = np.clip((np.arange(h) * H // h), 0, H - 1)
    cols = np.clip((np.arange(w) * W // w), 0, W - 1)
    return mask[np.ix_(rows, cols)]

def _find_palette_block(grid: np.ndarray):
    """
    Returns:
      (block, positions)
      block: 2x2 or 3x3 palette values
      positions: list of (r, c, color) relative to block top-left
    """
    h, w = grid.shape

    # Prefer 3x3 palette with a zero center and 8 non-zero entries.
    for r in range(h - 2):
        for c in range(w - 2):
            b = grid[r:r+3, c:c+3]
            if b[1, 1] == 0:
                vals = [int(x) for x in b.flat if x != 0]
                if len(vals) >= 4 and len(set(vals)) >= 4:
                    pos = [(i, j, int(b[i, j])) for i in range(3) for j in range(3) if b[i, j] != 0]
                    return b, pos

    # Fallback: 2x2 with four distinct non-zero colors.
    for r in range(h - 1):
        for c in range(w - 1):
            b = grid[r:r+2, c:c+2]
            vals = [int(x) for x in b.flat]
            if all(v != 0 for v in vals) and len(set(vals)) == 4:
                pos = [(i, j, int(b[i, j])) for i in range(2) for j in range(2)]
                return b, pos

    return None, []

def _is_3x3(palette):
    return palette is not None and palette.shape == (3, 3)

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the scattered pieces into a compact palette-colored mosaic.

    The input contains:
      - a small palette block (2x2 or 3x3) inside a frame,
      - several colored pieces.

    The output is a square mosaic where each palette position receives one
    piece.  Pieces are filled, rotated, and packed by trying corner placements.
    """
    if grid.size == 0:
        return grid.copy()

    palette, palette_pos = _find_palette_block(grid)
    if palette is None:
        return grid.copy()

    h, w = grid.shape

    # Find the frame color adjacent to the palette.
    pr, pc = None, None
    # Use the top-left palette coordinate in absolute grid.
    # We need absolute coordinates; _find_palette_block only returned block values.
    # Redo a light search to locate it absolutely.
    found = None
    if _is_3x3(palette):
        for r in range(h - 2):
            for c in range(w - 2):
                if np.array_equal(grid[r:r+3, c:c+3], palette):
                    found = (r, c)
                    break
            if found:
                break
    else:
        for r in range(h - 1):
            for c in range(w - 1):
                if np.array_equal(grid[r:r+2, c:c+2], palette):
                    found = (r, c)
                    break
            if found:
                break

    if found is None:
        return grid.copy()

    r0, c0 = found
    palette_h, palette_w = palette.shape

    # Frame color: look just outside the palette block.
    frame_color = None
    candidates = [
        (r0 - 1, c0),
        (r0 + palette_h, c0),
        (r0, c0 - 1),
        (r0, c0 + palette_w),
    ]
    for r, c in candidates:
        if 0 <= r < h and 0 <= c < w and grid[r, c] not in (0,) and int(grid[r, c]) not in [p[2] for p in palette_pos]:
            frame_color = int(grid[r, c])
            break

    if frame_color is None:
        # Fallback: most common non-zero color touching the palette.
        touched = set()
        for r in range(r0 - 1, r0 + palette_h + 1):
            for c in range(c0 - 1, c0 + palette_w + 1):
                if 0 <= r < h and 0 <= c < w and grid[r, c] != 0 and not (r0 <= r < r0 + palette_h and c0 <= c < c0 + palette_w):
                    touched.add(int(grid[r, c]))
        touched.difference_update([p[2] for p in palette_pos])
        if touched:
            frame_color = max(touched, key=lambda x: (grid == x).sum())

    # Extract frame mask.
    frame_mask = None
    if frame_color is not None:
        # Find a frame cell adjacent to palette.
        start = None
        for r in range(max(0, r0 - 1), min(h, r0 + palette_h + 1)):
            for c in range(max(0, c0 - 1), min(w, c0 + palette_w + 1)):
                if 0 <= r < h and 0 <= c < w and grid[r, c] == frame_color:
                    start = (r, c)
                    break
            if start:
                break
        if start:
            fm = _component_mask(grid, start[0], start[1], frame_color)
            # Include palette cells in the frame mask before filling holes.
            fm[r0:r0+palette_h, c0:c0+palette_w] |= (grid[r0:r0+palette_h, c0:c0+palette_w] != 0)
            frame_mask = _fill_holes(fm)
            frame_mask = _crop_mask(frame_mask)

    # Extract other pieces: all non-zero components not in palette and not frame.
    palette_mask = np.zeros_like(grid, dtype=bool)
    palette_mask[r0:r0+palette_h, c0:c0+palette_w] = grid[r0:r0+palette_h, c0:c0+palette_w] != 0

    used = palette_mask.copy()
    if frame_mask is not None:
        # Need absolute mask for used; we only have cropped frame. Re-extract full frame mask.
        pass

    # Re-extract full frame mask for used.
    full_frame = None
    if frame_color is not None:
        start = None
        for r in range(max(0, r0 - 1), min(h, r0 + palette_h + 1)):
            for c in range(max(0, c0 - 1), min(w, c0 + palette_w + 1)):
                if 0 <= r < h and 0 <= c < w and grid[r, c] == frame_color:
                    start = (r, c)
                    break
            if start:
                break
        if start:
            full_frame = _component_mask(grid, start[0], start[1], frame_color)
            full_frame[r0:r0+palette_h, c0:c0+palette_w] |= (grid[r0:r0+palette_h, c0:c0+palette_w] != 0)
            used |= full_frame

    pieces = []
    if frame_mask is not None:
        pieces.append((frame_mask, 'frame'))

    seen = np.zeros_like(grid, dtype=bool)
    seen |= used

    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0 and not seen[r, c]:
                color = int(grid[r, c])
                cm = _component_mask(grid, r, c, color)
                # Ignore components that are only palette digits.
                if np.array_equal(cm & used, cm):
                    continue
                pieces.append((_fill_holes(_crop_mask(cm)), f'color_{color}'))
                seen |= cm

    # Determine palette target positions.
    if _is_3x3(palette):
        target_cells = []
        for r in range(3):
            for c in range(3):
                if palette[r, c] != 0:
                    target_cells.append((r, c, int(palette[r, c])))
        # Map 3x3 position to a corner/side anchor.
        def anchor_kind(r, c):
            top = r == 0
            bottom = r == 2
            left = c == 0
            right = c == 2
            if top and left:
                return 'TL'
            if top and right:
                return 'TR'
            if bottom and left:
                return 'BL'
            if bottom and right:
                return 'BR'
            if top:
                return 'T'
            if bottom:
                return 'B'
            if left:
                return 'L'
            if right:
                return 'R'
            return 'C'
        targets = [(anchor_kind(r, c), r, c, col) for r, c, col in target_cells]
    else:
        # 2x2
        tl = int(palette[0, 0]); tr = int(palette[0, 1]); bl = int(palette[1, 0]); br = int(palette[1, 1])
        targets = [('TL', 0, 0, tl), ('TR', 0, 1, tr), ('BL', 1, 0, bl), ('BR', 1, 1, br)]

    # We need exactly len(targets) pieces. If frame included but not needed, drop it.
    if len(pieces) > len(targets):
        # Prefer dropping frame if there are enough non-frame pieces.
        non_frame = [p for p in pieces if p[1] != 'frame']
        if len(non_frame) >= len(targets):
            pieces = non_frame
        else:
            pieces = pieces[:len(targets)]
    elif len(pieces) < len(targets):
        # Duplicate the largest piece to fill missing slots.
        if pieces:
            biggest = max(pieces, key=lambda x: (x[0].shape[0], x[0].shape[1]))
            while len(pieces) < len(targets):
                pieces.append((biggest[0].copy(), 'filler'))
        else:
            # Should not happen.
            unit = np.ones((1, 1), dtype=bool)
            pieces = [(unit, 'filler')] * len(targets)

    # Try candidate square sizes.
    max_dim = max(max(p[0].shape) for p in pieces) if pieces else 2
    best = None

    # Candidate side lengths: around max_dim, and based on input size.
    candidates = set()
    for s in range(max(2, max_dim), max(30, int(np.ceil(np.sqrt(grid.size))) + 4)):
        candidates.add(s)
    candidates.add(10)
    candidates.add(8)

    for S in sorted(candidates):
        # Try all permutations of pieces to targets.
        import itertools
        for perm in itertools.permutations(range(len(pieces))):
            placed = np.zeros((S, S), dtype=bool)
            out = np.zeros((S, S), dtype=int)
            ok = True
            score = 0

            for (kind, pr, pc, col), pi in zip(targets, perm):
                mask = pieces[pi][0].astype(bool)

                # Choose a rotation that best fits the side.
                best_rot = None
                best_score = -1
                for k in range(4):
                    rm = _rotate(mask, k)
                    H, W = rm.shape
                    # Target size: cap to S.
                    th = min(H, S)
                    tw = min(W, S)
                    rmask = _resize_nearest(rm, th, tw)

                    # Position by anchor.
                    if kind in ('TL', 'T', 'L', 'C'):
                        if kind == 'TL':
                            rstart, cstart = 0, 0
                        elif kind == 'T':
                            rstart, cstart = 0, (S - tw) // 2
                        elif kind == 'L':
                            rstart, cstart = (S - th) // 2, 0
                        else:
                            rstart, cstart = (S - th) // 2, (S - tw) // 2
                    elif kind in ('TR', 'R'):
                        rstart, cstart = 0, S - tw
                    elif kind in ('BL', 'B'):
                        rstart, cstart = S - th, 0
                    else:  # BR
                        rstart, cstart = S - th, S - tw

                    rstart = max(0, min(S - th, rstart))
                    cstart = max(0, min(S - tw, cstart))

                    sub = placed[rstart:rstart+th, cstart:cstart+tw]
                    overlap = int(np.sum(sub & rmask))
                    fill = int(np.sum(rmask))
                    # Prefer no overlap and larger filled area.
                    rot_score = -1000 * overlap + fill
                    if rot_score > best_score:
                        best_score = rot_score
                        best_rot = (rmask, rstart, cstart, th, tw)

                rmask, rstart, cstart, th, tw = best_rot
                region = np.zeros((S, S), dtype=bool)
                region[rstart:rstart+th, cstart:cstart+tw] = rmask
                overlap = int(np.sum(region & placed))
                if overlap > 0:
                    # Penalize heavily; still allow if no alternative.
                    score -= 1000 * overlap
                    # Keep only non-overlapping part to maintain tiling.
                    region[region & placed] = False
                placed |= region
                out[region] = col
                score += int(np.sum(region))

            # Fill gaps with nearest target color.
            if np.any(placed == 0):
                # Compute distance to each target's anchor point.
                anchor_points = []
                for kind, pr, pc, col in targets:
                    if kind == 'TL':
                        anchor_points.append((0, 0, col))
                    elif kind == 'TR':
                        anchor_points.append((0, S - 1, col))
                    elif kind == 'BL':
                        anchor_points.append((S - 1, 0, col))
                    elif kind == 'BR':
                        anchor_points.append((S - 1, S - 1, col))
                    elif kind == 'T':
                        anchor_points.append((0, S // 2, col))
                    elif kind == 'B':
                        anchor_points.append((S - 1, S // 2, col))
                    elif kind == 'L':
                        anchor_points.append((S // 2, 0, col))
                    elif kind == 'R':
                        anchor_points.append((S // 2, S - 1, col))
                    else:
                        anchor_points.append((S // 2, S // 2, col))

                gap = (~placed).nonzero()
                if len(gap[0]) > 0:
                    min_dist = None
                    for r, c in zip(gap[0], gap[1]):
                        d = [(abs(r - ar) + abs(c - ac), col) for ar, ac, col in anchor_points]
                        min_dist = min(min_dist, d) if min_dist is not None else d
                        out[r, c] = min(d, key=lambda x: x[0])[1]
                    score -= int(len(gap[0]))

            # Prefer compact, no-overlap, full coverage.
            coverage = int(np.sum(placed))
            final_score = score + 10 * coverage - 0.1 * S
            if best is None or final_score > best[0]:
                best = (final_score, out.copy())

    if best is not None:
        return best[1]

    # Fallback: return the palette itself.
    return palette.astype(int)