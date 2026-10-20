import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill the rectangle indicated by the yellow (5) corner brackets.

    The solver combines several heuristics that match the observed families:
      * if the marked rectangle is a 2-wide multi-color vertical "block stack"
        (as in Example 3), infer the 2x2 block cycle from a nearby strip;
      * otherwise, use row-wise pattern completion (periods and arithmetic
        run progressions) for rows that already contain known data;
      * finally, use 2D translation symmetry to fill any remaining unknown
        cells from high-confidence shifts in the input.
    """
    g = np.array(grid, dtype=int)
    if g.size == 0:
        return np.zeros_like(g)

    if 5 not in np.unique(g):
        return np.zeros_like(g)

    # ------------------------------------------------------------------
    # 1. Locate the yellow marker rectangle.
    # ------------------------------------------------------------------
    ys, xs = np.where(g == 5)
    r0, r1 = int(ys.min()), int(ys.max())
    c0, c1 = int(xs.min()), int(xs.max())
    H = r1 - r0 + 1
    W = c1 - c0 + 1

    work = g.copy()
    work[work == 5] = 0

    # ------------------------------------------------------------------
    # Helpers: connected components, cycles, sequence extrapolation.
    # ------------------------------------------------------------------
    def components_nonzero(arr):
        R, C = arr.shape
        seen = np.zeros((R, C), dtype=bool)
        comps = []
        for i in range(R):
            for j in range(C):
                if arr[i, j] != 0 and not seen[i, j]:
                    q = deque([(i, j)])
                    seen[i, j] = True
                    minr = maxr = i
                    minc = maxc = j
                    while q:
                        r, c = q.popleft()
                        minr = min(minr, r)
                        maxr = max(maxr, r)
                        minc = min(minc, c)
                        maxc = max(maxc, c)
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < R and 0 <= nc < C and arr[nr, nc] != 0 and not seen[nr, nc]:
                                seen[nr, nc] = True
                                q.append((nr, nc))
                    comps.append({
                        "r0": minr, "r1": maxr,
                        "c0": minc, "c1": maxc,
                        "w": maxc - minc + 1,
                        "h": maxr - minr + 1,
                    })
        return comps

    def adjust_width(arr, mr0, mr1, mc0, mW):
        """
        Sometimes the yellow brackets mark a rectangle one pixel narrower
        than the object width (e.g. a 3-wide ladder bracket drawn with 2
        pixels). Expand the width conservatively if nearby objects clearly
        have a larger common width.
        """
        R, C = arr.shape
        comps = components_nonzero(arr)
        nearby = []
        for comp in comps:
            if comp["r1"] < mr0 - 2 or comp["r0"] > mr1 + 2:
                continue
            if comp["c1"] < mc0:
                dist = mc0 - comp["c1"]
            elif comp["c0"] > mc0 + mW - 1:
                dist = comp["c0"] - (mc0 + mW - 1)
            else:
                dist = 0
            if dist < max(10, mW):
                nearby.append(comp)

        if not nearby:
            return mW, mc0

        widths = [comp["w"] for comp in nearby]
        greater = [w for w in widths if w > mW]
        cand = None
        if greater:
            cand = min(greater)
        else:
            vals, counts = np.unique(widths, return_counts=True)
            mode = int(vals[np.argmax(counts)])
            if mode > mW:
                cand = mode

        if cand is not None and cand - mW <= 2:
            if mc0 + cand - 1 < C:
                return cand, mc0
            if mc0 - (cand - mW) >= 0:
                return cand, mc0 - (cand - mW)
        return mW, mc0

    W, c0 = adjust_width(work, r0, r1, c0, W)
    c1 = c0 + W - 1

    def find_cycle(seq):
        seq = list(seq)
        if len(seq) <= 1:
            return seq
        L = len(seq)
        for p in range(1, L // 2 + 1):
            ok = True
            for i in range(L):
                if seq[i] != seq[i % p]:
                    ok = False
                    break
            if ok:
                return seq[:p]
        return seq

    def first_occurrence_order(seq):
        out = []
        seen = set()
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(int(x))
        return out

    def color_cycle(colors):
        colors = [int(x) for x in colors]
        cyc = find_cycle(colors)
        if len(cyc) < len(colors):
            return cyc
        return first_occurrence_order(colors)

    def most_common_color(arr):
        nz = arr[arr != 0]
        if nz.size == 0:
            return 0
        vals, cnts = np.unique(nz, return_counts=True)
        return int(vals[np.argmax(cnts)])

    def extend_sequence(vals, extra):
        vals = list(vals)
        if extra <= 0 or len(vals) == 0:
            return vals
        if len(vals) == 1:
            vals += [vals[-1]] * extra
            return vals
        diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        if len(diffs) >= 2:
            second = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
            if all(d == second[0] for d in second):
                last_diff = diffs[-1]
                for _ in range(extra):
                    last_diff += second[0]
                    vals.append(vals[-1] + last_diff)
                return vals
        step = diffs[-1]
        for _ in range(extra):
            vals.append(vals[-1] + step)
        return vals

    # ------------------------------------------------------------------
    # 2. Special solver: 2-wide multi-color vertical block stacks.
    #    This is the pattern in Example #3.
    # ------------------------------------------------------------------
    def find_block_strip(arr, mr0, mr1, mc0, mW, mH):
        R, C = arr.shape
        if mW != 2:
            return None
        best = None
        for start in range(C - mW + 1):
            strip = arr[:, start:start + mW]
            nz = strip[strip != 0]
            if nz.size == 0:
                continue
            if len(np.unique(nz)) <= 1:
                continue
            rows = np.where((strip != 0).any(axis=1))[0]
            if rows.size == 0:
                continue
            overlap = int(np.intersect1d(rows, np.arange(mr0, mr1 + 1)).size)
            wr0 = max(0, mr0 - mH)
            wr1 = min(R, mr1 + mH + 1)
            count = int((strip[wr0:wr1] != 0).sum())
            if count < 2:
                continue
            if start + mW <= mc0:
                dist = mc0 - (start + mW)
            elif start >= mc0 + mW:
                dist = start - (mc0 + mW - 1)
            else:
                dist = 0
            score = count * 10 + overlap * 5 - dist
            if best is None or score > best[0]:
                best = (score, start)
        if best is None:
            return None
        return arr[:, best[1]:best[1] + mW]

    def render_block_stack(arr, strip, mr0, mH):
        R, C = arr.shape
        out = np.zeros((mH, 2), dtype=int)

        # Snap to a full row near the top of the missing rectangle.
        r_start = mr0
        full_rows = []
        for rr in range(max(0, mr0 - 3), min(R, mr0 + 4)):
            if (strip[rr] != 0).all():
                full_rows.append(rr)
        if full_rows:
            r_start = min(full_rows)

        mains = []
        max_blocks = min(mH // 2, max(0, (R - r_start) // 2))
        for b in range(max_blocks):
            blk = strip[r_start + 2 * b:r_start + 2 * b + 2]
            nz = blk[blk != 0]
            if nz.size == 0:
                break
            vals, cnts = np.unique(nz, return_counts=True)
            mains.append(int(vals[np.argmax(cnts)]))
            if (blk == 0).any():
                break

        cycle = find_cycle(mains) if mains else [most_common_color(arr)]
        if not cycle:
            cycle = [most_common_color(arr)]

        total_blocks = mH // 2
        for idx in range(total_blocks):
            main = cycle[idx % len(cycle)]
            out[2 * idx:2 * idx + 2, :] = main
            odd = 0 if idx == total_blocks - 1 else cycle[(idx + 1) % len(cycle)]
            notch_r = 2 * idx + 1
            notch_c = 0 if idx % 2 == 0 else 1
            out[notch_r, notch_c] = odd

        return out

    if W == 2 and H % 2 == 0:
        strip = find_block_strip(work, r0, r1, c0, W, H)
        if strip is not None:
            res = render_block_stack(work, strip, r0, H)
            if res.size:
                return res

    # ------------------------------------------------------------------
    # 3. Row-wise completion: periodic rows and arithmetic run progressions.
    # ------------------------------------------------------------------
    def periodic_row_fill(row, known, fc0, fW):
        C = len(row)
        best_p = None
        best_acc = 0.0
        max_p = min(C, 30)

        for p in range(1, max_p + 1):
            possible = 0
            matches = 0
            for c in range(C - p):
                if known[c] and known[c + p]:
                    possible += 1
                    if row[c] == row[c + p]:
                        matches += 1
            if possible >= max(4, C // 10):
                acc = matches / possible
                if acc > best_acc:
                    best_acc = acc
                    best_p = p

        if best_p is None or best_acc < 0.8:
            return None

        fill = np.zeros(fW, dtype=int)
        filled = 0
        max_k = (C // best_p) + 3
        for i in range(fW):
            c = fc0 + i
            found = False
            val = 0
            for k in range(1, max_k + 1):
                for cc in (c - k * best_p, c + k * best_p):
                    if 0 <= cc < C and known[cc] and not found:
                        val = int(row[cc])
                        found = True
                        break
                if found:
                    break
            if found:
                fill[i] = val
                filled += 1
        return fill, filled, filled == fW

    def run_row_fill(row, known, fc0, fW):
        C = len(row)
        runs = []
        c = 0
        while c < C:
            if known[c] and row[c] != 0:
                col = int(row[c])
                start = c
                length = 0
                multi = False
                while c < C and known[c] and row[c] != 0:
                    if int(row[c]) != col:
                        multi = True
                    c += 1
                    length += 1
                if not multi:
                    runs.append((start, length, col))
            else:
                c += 1

        if len(runs) < 2:
            return None

        starts = [r[0] for r in runs]
        lengths = [r[1] for r in runs]
        colors = [r[2] for r in runs]

        ext_starts = extend_sequence(starts, 12)
        ext_lengths = extend_sequence(lengths, 12)
        cycle = color_cycle(colors)

        fill = np.zeros(fW, dtype=int)
        filled = 0

        for i, s in enumerate(ext_starts):
            l = ext_lengths[i] if i < len(ext_lengths) else lengths[-1]
            if l <= 0:
                continue
            if s + l <= fc0:
                continue
            if s >= fc0 + fW:
                break
            col = colors[i] if i < len(colors) else int(cycle[i % len(cycle)])
            a = max(s, fc0)
            b = min(s + l - 1, fc0 + fW - 1)
            if a <= b:
                fill[a - fc0:b - fc0 + 1] = col
                filled += b - a + 1

        if filled > 0:
            return fill, filled, True
        return None

    def complete_row(row, known, fc0, fW):
        per = periodic_row_fill(row, known, fc0, fW)
        run = run_row_fill(row, known, fc0, fW)

        if per is not None and (run is None or per[1] > run[1] or per[2]):
            return per
        if run is not None:
            return run
        return None

    out = np.zeros((H, W), dtype=int)
    filled = np.zeros((H, W), dtype=bool)

    for rr in range(r0, r1 + 1):
        known = np.ones(work.shape[1], dtype=bool)
        known[c0:c0 + W] = False
        # Only use this row if it has some nonzero evidence outside the box.
        if (work[rr, known] != 0).any():
            res = complete_row(work[rr], known, c0, W)
            if res is not None:
                fill, _filled_count, full = res
                out[rr - r0, :] = fill
                if full:
                    filled[rr - r0, :] = True
                else:
                    filled[rr - r0, fill != 0] = True

    # ------------------------------------------------------------------
    # 4. 2D translation symmetry fallback for rows with no local evidence.
    # ------------------------------------------------------------------
    unknown = np.zeros((H, W), dtype=bool)
    # Start with all cells in the rectangle unknown, then mark filled ones known.
    unknown[:] = True
    unknown[filled] = False

    full_unknown = np.zeros(work.shape, dtype=bool)
    full_unknown[r0:r1 + 1, c0:c0 + W] = unknown
    known_full = ~full_unknown
    known_cells = np.argwhere(known_full).tolist()

    maxd = min(20, max(work.shape) // 2)
    candidates = []

    for dy in range(-maxd, maxd + 1):
        for dx in range(-maxd, maxd + 1):
            if dy == 0 and dx == 0:
                continue
            possible = 0
            mask_match = 0
            nonzero_pairs = 0
            same = 0

            for r, c in known_cells:
                r2, c2 = r + dy, c + dx
                if 0 <= r2 < work.shape[0] and 0 <= c2 < work.shape[1] and known_full[r2, c2]:
                    possible += 1
                    v1 = work[r, c]
                    v2 = work[r2, c2]
                    if (v1 == 0) == (v2 == 0):
                        mask_match += 1
                    if v1 != 0 and v2 != 0:
                        nonzero_pairs += 1
                        if v1 == v2:
                            same += 1

            if possible >= 10:
                ratio = mask_match / possible
                bonus = same / nonzero_pairs if nonzero_pairs > 0 else 0.0
                if ratio >= 0.6:
                    candidates.append((ratio, bonus, mask_match, -(abs(dy) + abs(dx)), dy, dx))

    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
    candidates = candidates[:20]

    if candidates:
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c0 + W):
                if filled[rr - r0, cc - c0]:
                    continue
                counts = {}
                total_weight = 0
                for ratio, bonus, matches, _negdist, dy, dx in candidates:
                    sr, sc = rr - dy, cc - dx
                    if 0 <= sr < work.shape[0] and 0 <= sc < work.shape[1] and known_full[sr, sc]:
                        val = int(work[sr, sc])
                        wgt = max(1, int(matches))
                        counts[val] = counts.get(val, 0) + wgt
                        total_weight += wgt
                if not counts:
                    continue
                best_val = max(counts.items(), key=lambda kv: (kv[1], kv[0] != 0))[0]
                best_weight = counts[best_val]
                if best_weight >= max(2, total_weight // 2):
                    out[rr - r0, cc - c0] = best_val
                    filled[rr - r0, cc - c0] = True

    return out