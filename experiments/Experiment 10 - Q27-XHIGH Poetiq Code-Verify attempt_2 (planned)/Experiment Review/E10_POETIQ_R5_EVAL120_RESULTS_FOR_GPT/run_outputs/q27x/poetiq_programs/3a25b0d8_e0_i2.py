import numpy as np
from collections import defaultdict, Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the intended compact object from a scattered/damaged grid.

    The rule inferred from the examples is:

      * One colour is the global background.
      * One other colour is the "frame" / wall colour.  It usually has the
        longest straight runs and appears in multiple copies of the object.
      * The input contains several shifted/corrupted HxW crops of the same
        target object.
      * The output is the consensus of all valid HxW crops:
          - non-background cells are voted by majority,
          - frame colour wins ties against payload colours,
          - small missing interior regions are completed by local filling.

    The implementation is deliberately heuristic because ARC inputs can be
    noisy.  It works in the following stages:

    1. Detect background and frame colour.
    2. Infer the target width from the longest frame run.
    3. Infer a base HxW crop from the longest/most supporting frame run.
    4. Search for additional HxW crops that agree with the current consensus.
    5. Build a colour-vote consensus.
    6. Apply conservative local completion:
         - fill isolated background holes with the surrounding colour,
         - complete small frame gaps on the object boundary,
         - extend payload rooms when strongly supported by adjacent pixels.
    """

    if grid.size == 0:
        return grid.copy()

    grid = np.array(grid, copy=True)
    H_in, W_in = grid.shape
    if H_in <= 1 or W_in <= 1:
        return grid

    flat = grid.ravel()
    vals, counts = np.unique(flat, return_counts=True)
    colour_count = dict(zip(vals.tolist(), counts.tolist()))

    # ---------------------------------------------------------------------
    # 1. Background: most frequent colour.
    #    Frame: non-background colour with the strongest straight-run evidence.
    # ---------------------------------------------------------------------
    bg = max(colour_count, key=lambda c: (colour_count[c], -c))

    def max_h_run(mask):
        best = 0
        for row in mask:
            run = 0
            for v in row:
                if v:
                    run += 1
                    best = max(best, run)
                else:
                    run = 0
        return best

    def max_v_run(mask):
        best = 0
        for col in mask.T:
            run = 0
            for v in col:
                if v:
                    run += 1
                    best = max(best, run)
                else:
                    run = 0
        return best

    non_bg = [c for c in colour_count if c != bg]
    if not non_bg:
        return np.full_like(grid, bg)

    frame = None
    best_score = -1
    for c in non_bg:
        mask = (grid == c)
        score = 1000 * max_h_run(mask) + 500 * max_v_run(mask) + colour_count[c]
        if score > best_score:
            best_score = score
            frame = c

    if frame is None or frame == bg:
        return grid.copy()

    frame_mask = (grid == frame)

    # ---------------------------------------------------------------------
    # 2/3. Infer target width and a base crop.
    # ---------------------------------------------------------------------
    # Find all maximal horizontal frame runs.
    runs = []
    for r in range(H_in):
        c = 0
        while c < W_in:
            if frame_mask[r, c]:
                c0 = c
                while c < W_in and frame_mask[r, c]:
                    c += 1
                runs.append((r, c0, c - 1))
            else:
                c += 1

    if not runs:
        # No usable frame: crop all non-background as a fallback.
        nonbg = (grid != bg)
        if not nonbg.any():
            return np.full_like(grid, bg)
        rows = np.any(nonbg, axis=1)
        cols = np.any(nonbg, axis=0)
        r0, r1 = np.argmax(rows), H_in - np.argmax(rows[::-1])
        c0, c1 = np.argmax(cols), W_in - np.argmax(cols[::-1])
        return grid[r0:r1 + 1, c0:c1 + 1].copy()

    target_w = max(length for _, _, _ in [(r, a, b) for r, a, b in runs] for length in [b - a + 1])

    def vertical_span_for_window(r0c, c0c, width, min_cells=2):
        """Expand vertically while the width-window still contains frame."""
        top = r0c
        bottom = r0c
        while top > 0 and np.sum(frame_mask[top - 1, c0c:c0c + width]) >= min_cells:
            top -= 1
        while bottom + 1 < H_in and np.sum(frame_mask[bottom + 1, c0c:c0c + width]) >= min_cells:
            bottom += 1
        return top, bottom

    best_run = None
    best_run_score = -1

    for r, a, b in runs:
        length = b - a + 1
        if length < max(2, target_w - 1):
            continue
        c0 = a
        if length > target_w:
            # Use the best aligned target_w sub-run.
            c0 = a
        width = min(length, target_w)
        top, bottom = vertical_span_for_window(r, c0, width)
        span_h = bottom - top + 1
        frame_support = int(np.sum(frame_mask[top:bottom + 1, c0:c0 + width]))
        score = 1000 * span_h + frame_support
        if score > best_run_score:
            best_run_score = score
            best_run = (r, c0, width, top, bottom, span_h, frame_support)

    if best_run is None:
        r, a, b = max(runs, key=lambda x: x[2] - x[1])
        best_run = (r, a, b - a + 1, r, r, 1, 1)

    run_row, base_c0, run_len, span_top, span_bottom, span_h, frame_support = best_run
    base_w = max(target_w, run_len)

    if span_h >= base_w:
        target_h = span_h
        base_r0 = span_top
    else:
        target_h = base_w
        base_r0 = run_row - base_w // 2
        if base_r0 < span_top:
            base_r0 = span_top
        if base_r0 + target_h > H_in:
            base_r0 = max(0, H_in - target_h)

    # Keep base window inside the grid.
    if base_c0 < 0:
        base_c0 = 0
    if base_c0 + base_w > W_in:
        base_c0 = max(0, W_in - base_w)
    if base_r0 < 0:
        base_r0 = 0
    if base_r0 + target_h > H_in:
        base_r0 = max(0, H_in - target_h)

    # If inferred size is too large for a useful crop, shrink slightly.
    # This avoids pathological all-grid crops in noisy cases.
    if target_h > H_in // 2 + 3 and base_w > W_in // 2 + 3:
        target_h = max(1, min(target_h, H_in - 1))
        base_w = max(1, min(base_w, W_in - 1))
        base_r0 = min(base_r0, max(0, H_in - target_h))
        base_c0 = min(base_c0, max(0, W_in - base_w))

    H = target_h
    W = base_w

    # ---------------------------------------------------------------------
    # 4. Collect candidate HxW crops.
    # ---------------------------------------------------------------------
    base_window = grid[base_r0:base_r0 + H, base_c0:base_c0 + W].copy()

    def consensus_from_positions(positions):
        counts = [[[0 for _ in range(10)] for _ in range(H)] for _ in range(W)]
        for r0, c0 in positions:
            win = grid[r0:r0 + H, c0:c0 + W]
            for i in range(H):
                row_win = win[i]
                row_cnt = counts[i]
                for j in range(W):
                    v = int(row_win[j])
                    if v != bg:
                        row_cnt[j][v] += 1
        out = np.full((H, W), bg, dtype=grid.dtype)
        for i in range(H):
            for j in range(W):
                cnts = counts[i][j]
                best_colour = bg
                best_val = 0
                # Tie-break: frame > more globally common colour.
                for col, n in enumerate(cnts):
                    if n == 0:
                        continue
                    key = (n, col == frame, colour_count.get(col, 0))
                    if key > (best_val, best_colour == frame, colour_count.get(best_colour, 0)):
                        best_val = n
                        best_colour = col
                out[i, j] = best_colour
        return out

    initial_consensus = consensus_from_positions([(base_r0, base_c0)])
    template_frame = (initial_consensus == frame)

    selected = [(base_r0, base_c0)]

    def overlap_with_selected(r0, c0, frac=0.45):
        for sr0, sc0 in selected:
            hr = max(0, min(r0 + H, sr0 + H) - max(r0, sr0))
            hc = max(0, min(c0 + W, sc0 + W) - max(c0, sc0))
            if hr * hc >= frac * H * W:
                return True
        return False

    candidates = []
    for r0 in range(max(0, H_in - H + 1)):
        for c0 in range(max(0, W_in - W + 1)):
            if overlap_with_selected(r0, c0):
                continue
            win = grid[r0:r0 + H, c0:c0 + W]
            obs = win != bg
            if int(np.sum(obs)) < max(5, W // 2):
                continue

            # Frame consistency with current consensus.
            f_obs = (win == frame)
            match = int(np.sum(f_obs & template_frame))
            conflict = int(np.sum(f_obs & ~template_frame))
            if match < max(3, 0.15 * int(np.sum(f_obs))):
                continue
            if conflict > 0.35 * max(1, int(np.sum(f_obs))):
                continue

            # General observation score against current consensus.
            m = int(np.sum(obs & (win == initial_consensus)))
            cf = int(np.sum(obs & (win != initial_consensus)))
            score = m - cf
            if score <= 0:
                continue
            if cf > 0.45 * max(1, m + cf):
                continue

            candidates.append((score, r0, c0, m, cf))

    # Add strongest, mutually non-overlapping candidates.
    candidates.sort(key=lambda x: (-x[0], x[2], x[1]))
    added = 0
    for score, r0, c0, m, cf in candidates:
        if overlap_with_selected(r0, c0):
            continue
        selected.append((r0, c0))
        added += 1
        if added >= 8:  # More than enough in typical tasks.
            break

    # Iteratively refine with the updated consensus.
    for _ in range(2):
        current = consensus_from_positions(selected)
        tf = (current == frame)
        # Re-score a few best unselected windows and add if clearly good.
        extra = []
        for r0 in range(max(0, H_in - H + 1)):
            for c0 in range(max(0, W_in - W + 1)):
                if overlap_with_selected(r0, c0):
                    continue
                win = grid[r0:r0 + H, c0:c0 + W]
                obs = win != bg
                nobs = int(np.sum(obs))
                if nobs < max(5, W // 2):
                    continue
                f_obs = (win == frame)
                fm = int(np.sum(f_obs & tf))
                fc = int(np.sum(f_obs & ~tf))
                if fm < max(3, 0.15 * max(1, nobs)):
                    continue
                if fc > 0.35 * max(1, nobs):
                    continue
                m = int(np.sum(obs & (win == current)))
                cf = int(np.sum(obs & (win != current)))
                if m <= 0 or cf > 0.40 * (m + cf):
                    continue
                extra.append((m - cf, r0, c0))
        extra.sort(reverse=True)
        for _, r0, c0 in extra[:3]:
            if not overlap_with_selected(r0, c0):
                selected.append((r0, c0))

    out = consensus_from_positions(selected)

    # ---------------------------------------------------------------------
    # 5. Conservative local completion.
    # ---------------------------------------------------------------------

    def neighbour_count(mask, i, j, diag=True):
        s = 0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and mask[ni, nj]:
                    s += 1
        return s

    # 5.1 Fill small background holes that are clearly inside the object.
    for _ in range(2):
        changed = False
        for i in range(H):
            for j in range(W):
                if out[i, j] == bg:
                    # Must be surrounded by non-background.
                    if all(0 <= ni < H and 0 <= nj < W and out[ni, nj] != bg
                           for ni in (i - 1, i, i + 1)
                           for nj in (j - 1, j, j + 1)
                           if not (ni == i and nj == j)):
                        # Choose the most common non-background colour in 3x3.
                        cnt = Counter(int(out[ni, nj])
                                      for ni in (i - 1, i, i + 1)
                                      for nj in (j - 1, j, j + 1)
                                      if 0 <= ni < H and 0 <= nj < W
                                      and out[ni, nj] != bg)
                        if cnt:
                            out[i, j] = cnt.most_common(1)[0][0]
                            changed = True
        if not changed:
            break

    # 5.2 Complete small frame gaps on rows/columns that are clearly frame lines.
    for _ in range(1):
        # Horizontal gaps: frame at both sides, same row, gap length <= 3,
        # and the row is not the very top row.
        for i in range(1, H):
            j = 0
            while j < W:
                if out[i, j] == frame:
                    c0 = j
                    while j < W and out[i, j] == frame:
                        j += 1
                    c1 = j - 1
                    if j < W and out[i, j] != bg and out[i, j] != frame:
                        # Find a possible gap immediately after a frame run.
                        gap_start = j
                        gap_len = 0
                        k = j
                        while k < W and out[i, k] != frame and gap_len <= 3:
                            gap_len += 1
                            k += 1
                        if k < W and out[i, k] == frame and gap_len <= 3:
                            # Only fill if the row above has object in this span.
                            if i > 0 and np.any(out[i - 1, gap_start:k] != bg):
                                for g in range(gap_start, k):
                                    if out[i, g] == bg:
                                        out[i, g] = frame
                                j = k
                                continue
                    continue
                j += 1

        # Vertical gaps: frame above and below, gap length <= 3.
        for j in range(W):
            i = 0
            while i < H:
                if out[i, j] == frame:
                    r0 = i
                    while i < H and out[i, j] == frame:
                        i += 1
                    r1 = i - 1
                    if i < H and out[i, j] != bg and out[i, j] != frame:
                        gap_start = i
                        gap_len = 0
                        k = i
                        while k < H and out[k, j] != frame and gap_len <= 3:
                            gap_len += 1
                            k += 1
                        if k < H and out[k, j] == frame and gap_len <= 3:
                            if j > 0 and np.any(out[gap_start:k, j - 1] != bg):
                                for g in range(gap_start, k):
                                    if out[g, j] == bg:
                                        out[g, j] = frame
                                i = k
                                continue
                    continue
                i += 1

    # 5.3 Payload room completion.
    #     For each payload colour, extend a strongly supported run into
    #     adjacent cells that are otherwise frame/background and are locally
    #     enclosed by the same payload.
    payload_cols = [c for c in np.unique(out) if c != bg and c != frame]

    for col in payload_cols:
        col_mask = (out == col)
        if not col_mask.any():
            continue

        # Work on a copy so a colour can be completed iteratively.
        fill = col_mask.copy()

        # Repeat a few times.
        for _ in range(3):
            changed = False
            for i in range(H):
                for j in range(W):
                    if fill[i, j] or out[i, j] == col:
                        continue
                    if out[i, j] not in (bg, frame):
                        continue

                    # Need strong same-colour support in 3x3.
                    support = 0
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < H and 0 <= nj < W and out[ni, nj] == col:
                                support += 1
                    if support < 3:
                        continue

                    # Must be horizontally or vertically connected to a run of length >= 2.
                    horiz = (
                        (j > 0 and out[i, j - 1] == col and (j + 1 < W and out[i, j + 1] == col)) or
                        (j + 1 < W and out[i, j + 1] == col and (j - 1 >= 0 and out[i, j - 1] == col))
                    )
                    vert = (
                        (i > 0 and out[i - 1, j] == col and (i + 1 < H and out[i + 1, j] == col)) or
                        (i + 1 < H and out[i + 1, j] == col and (i - 1 >= 0 and out[i - 1, j] == col))
                    )
                    if not (horiz or vert):
                        continue

                    # Avoid crossing a different payload colour in the 4-neighbourhood.
                    blocked = False
                    for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                        if 0 <= ni < H and 0 <= nj < W:
                            v = int(out[ni, nj])
                            if v != bg and v != frame and v != col:
                                blocked = True
                                break
                    if blocked:
                        continue

                    # Do not fill cells that are on the outer border unless they
                    # are already supported by frame on the inside.
                    if i == 0 or i == H - 1 or j == 0 or j == W - 1:
                        inside_support = 0
                        if i > 0 and out[i - 1, j] in (col, frame):
                            inside_support += 1
                        if i < H - 1 and out[i + 1, j] in (col, frame):
                            inside_support += 1
                        if j > 0 and out[i, j - 1] in (col, frame):
                            inside_support += 1
                        if j < W - 1 and out[i, j + 1] in (col, frame):
                            inside_support += 1
                        if inside_support < 2:
                            continue

                    fill[i, j] = True
                    changed = True
            if not changed:
                break

        # Apply only the newly filled cells.
        new_cells = fill & ~col_mask
        if new_cells.any():
            out[new_cells] = col

    # ---------------------------------------------------------------------
    # 6. Final crop: remove all-background border rows/columns if present.
    # ---------------------------------------------------------------------
    nonbg = (out != bg)
    if not nonbg.any():
        return out

    rows = np.any(nonbg, axis=1)
    cols = np.any(nonbg, axis=0)
    r0, r1 = int(np.argmax(rows)), H - int(np.argmax(rows[::-1]))
    c0, c1 = int(np.argmax(cols)), W - int(np.argmax(cols[::-1]))

    # Keep at least one row/col.
    return out[r0:r1 + 1, c0:c1 + 1].copy()