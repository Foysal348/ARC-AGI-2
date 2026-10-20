import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Heuristic ARC rule:
    - Larger same-color clusters act as containers.
    - Pixels of another color inside a container's bounding box are treated as
      interior tokens.
    - Each interior token group is collapsed into a straight line of length
      equal to its vertical span.
    - The line is placed just outside the container on a chosen side:
      * opposite the token center,
      * preferring the side with fewer container pixels,
      * only if it fits in the grid,
      * aligned using the nearest token edge / container side.
    - Token pixels outside the container bounding box remain unchanged.
    """
    arr = np.array(grid, dtype=int, copy=True)
    if arr.ndim != 2:
        return arr

    h, w = arr.shape
    if h == 0 or w == 0:
        return arr

    pixels_by_color = {}
    for r in range(h):
        for c in range(w):
            v = int(arr[r, c])
            if v != 0:
                pixels_by_color.setdefault(v, []).append((r, c))

    if not pixels_by_color:
        return arr

    def components8(pixels):
        """Return 8-connected components for a list of pixels."""
        remaining = set(pixels)
        comps = []
        for p in pixels:
            if p in remaining:
                stack = [p]
                remaining.remove(p)
                comp = []
                while stack:
                    r, c = stack.pop()
                    comp.append((r, c))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in remaining:
                                remaining.remove((nr, nc))
                                stack.append((nr, nc))
                comps.append(comp)
        return comps

    def bbox_of(pixels):
        rs = [p[0] for p in pixels]
        cs = [p[1] for p in pixels]
        return min(rs), max(rs), min(cs), max(cs)

    def gap_can_merge(b1, b2):
        """
        Merge same-color components if their bboxes touch/overlap, or are
        separated by one cell that is not background. This handles cases where
        a container is visually one object but separated by token pixels.
        """
        gr = max(0, b1[0] - b2[1] - 1, b2[0] - b1[1] - 1)
        gc = max(0, b1[2] - b2[3] - 1, b2[2] - b1[3] - 1)

        if gr == 0 and gc == 0:
            return True

        if gr > 0 and gc == 0:
            if b1[1] < b2[0]:
                r = b1[1] + 1
            else:
                r = b2[1] + 1
            c0 = max(b1[2], b2[2])
            c1 = min(b1[3], b2[3])
            if c0 <= c1:
                for c in range(c0, c1 + 1):
                    if 0 <= r < h and 0 <= c < w and arr[r, c] != 0:
                        return True
            return False

        if gc > 0 and gr == 0:
            if b1[3] < b2[2]:
                c = b1[3] + 1
            else:
                c = b2[3] + 1
            r0 = max(b1[0], b2[0])
            r1 = min(b1[1], b2[1])
            if r0 <= r1:
                for r in range(r0, r1 + 1):
                    if 0 <= r < h and 0 <= c < w and arr[r, c] != 0:
                        return True
            return False

        if gr > 0 and gc > 0:
            if b1[1] < b2[0]:
                r = b1[1] + 1
            else:
                r = b2[1] + 1
            if b1[3] < b2[2]:
                c = b1[3] + 1
            else:
                c = b2[3] + 1
            if 0 <= r < h and 0 <= c < w and arr[r, c] != 0:
                return True
            return False

        return False

    # Build same-color clusters, merging components when appropriate.
    clusters = []
    for color, pixels in pixels_by_color.items():
        comps = components8(pixels)
        cluster_pixels = [comp for comp in comps if comp]

        changed = True
        while changed:
            changed = False
            bboxes = [bbox_of(px) for px in cluster_pixels]
            for i in range(len(cluster_pixels)):
                for j in range(i + 1, len(cluster_pixels)):
                    if gap_can_merge(bboxes[i], bboxes[j]):
                        cluster_pixels[i] = cluster_pixels[i] + cluster_pixels[j]
                        del cluster_pixels[j]
                        changed = True
                        break
                if changed:
                    break

        for px in cluster_pixels:
            rmin, rmax, cmin, cmax = bbox_of(px)
            clusters.append({
                'color': color,
                'pixels': px,
                'rmin': rmin,
                'rmax': rmax,
                'cmin': cmin,
                'cmax': cmax,
                'count': len(px)
            })

    def side_count(C, side):
        cnt = 0
        for r, c in C['pixels']:
            if side == 'top' and r == C['rmin']:
                cnt += 1
            elif side == 'bottom' and r == C['rmax']:
                cnt += 1
            elif side == 'left' and c == C['cmin']:
                cnt += 1
            elif side == 'right' and c == C['cmax']:
                cnt += 1
        return cnt

    def line_cells(side, C, ref, length):
        cells = []
        if side == 'top':
            for i in range(1, length + 1):
                cells.append((C['rmin'] - i, ref))
        elif side == 'bottom':
            for i in range(1, length + 1):
                cells.append((C['rmax'] + i, ref))
        elif side == 'left':
            for i in range(1, length + 1):
                cells.append((ref, C['cmin'] - i))
        elif side == 'right':
            for i in range(1, length + 1):
                cells.append((ref, C['cmax'] + i))
        return cells

    def choose_reference(side, C, F_pixels, fr, fc, length, protected):
        """Choose the alignment coordinate for the output line."""
        if side in ('top', 'bottom'):
            if side == 'top':
                minr = min(r for r, c in F_pixels)
                edge_cols = sorted({c for r, c in F_pixels if r == minr})
            else:
                maxr = max(r for r, c in F_pixels)
                edge_cols = sorted({c for r, c in F_pixels if r == maxr})

            if not edge_cols:
                edge_cols = sorted({c for r, c in F_pixels})

            side_cols = sorted({
                c for r, c in C['pixels']
                if (side == 'top' and r == C['rmin']) or (side == 'bottom' and r == C['rmax'])
            })
            cc = (C['cmin'] + C['cmax']) / 2.0

            if side_cols:
                edge_sorted = sorted(edge_cols, key=lambda c: (
                    min(abs(c - x) for x in side_cols),
                    abs(c - cc)
                ))
            else:
                edge_sorted = sorted(edge_cols, key=lambda c: abs(c - cc))

            prefs = list(edge_sorted)
            for c in side_cols:
                if c not in prefs:
                    prefs.append(c)
            cc_int = int(round(cc))
            if cc_int not in prefs:
                prefs.append(cc_int)

            for ref in prefs:
                if not (C['cmin'] <= ref <= C['cmax']):
                    continue
                cells = line_cells(side, C, ref, length)
                if all(0 <= r < h and 0 <= c < w and (r, c) not in protected for r, c in cells):
                    if all(arr[r, c] == 0 for r, c in cells):
                        return ref, cells

            # Fallback: use first spatially valid alignment even if blocked.
            for ref in prefs:
                if not (C['cmin'] <= ref <= C['cmax']):
                    continue
                cells = line_cells(side, C, ref, length)
                if all(0 <= r < h and 0 <= c < w and (r, c) not in protected for r, c in cells):
                    return ref, cells

            return None, None

        else:
            if side == 'right':
                side_rows = sorted({r for r, c in C['pixels'] if c == C['cmax']})
                maxc = max(c for r, c in F_pixels)
                edge_rows = sorted({r for r, c in F_pixels if c == maxc})
            else:
                side_rows = sorted({r for r, c in C['pixels'] if c == C['cmin']})
                minc = min(c for r, c in F_pixels)
                edge_rows = sorted({r for r, c in F_pixels if c == minc})

            if not edge_rows:
                edge_rows = sorted({r for r, c in F_pixels})

            cr = (C['rmin'] + C['rmax']) / 2.0
            prefs = []

            if side_rows:
                for r in sorted(side_rows, key=lambda r: abs(r - fr)):
                    prefs.append(r)
                for r in sorted(edge_rows, key=lambda r: abs(r - fr)):
                    if r not in prefs:
                        prefs.append(r)
            else:
                for r in sorted(edge_rows, key=lambda r: abs(r - cr)):
                    prefs.append(r)

            cr_int = int(round(cr))
            if cr_int not in prefs:
                prefs.append(cr_int)

            for ref in prefs:
                if not (C['rmin'] <= ref <= C['rmax']):
                    continue
                cells = line_cells(side, C, ref, length)
                if all(0 <= r < h and 0 <= c < w and (r, c) not in protected for r, c in cells):
                    if all(arr[r, c] == 0 for r, c in cells):
                        return ref, cells

            # Fallback: use first spatially valid alignment even if blocked.
            for ref in prefs:
                if not (C['rmin'] <= ref <= C['rmax']):
                    continue
                cells = line_cells(side, C, ref, length)
                if all(0 <= r < h and 0 <= c < w and (r, c) not in protected for r, c in cells):
                    return ref, cells

            return None, None

    def choose_side(C, F_pixels, protected):
        """Choose output side and cells for a candidate container/token pair."""
        if not F_pixels:
            return None

        rs = [r for r, c in F_pixels]
        cs = [c for r, c in F_pixels]
        length = max(rs) - min(rs) + 1

        fr = sum(rs) / len(rs)
        fc = sum(cs) / len(cs)

        cr = (C['rmin'] + C['rmax']) / 2.0
        cc = (C['cmin'] + C['cmax']) / 2.0

        eps = 1e-9
        dirs = []
        if fr < cr - eps:
            dirs.append('bottom')
        if fr > cr + eps:
            dirs.append('top')
        if fc < cc - eps:
            dirs.append('right')
        if fc > cc + eps:
            dirs.append('left')

        if not dirs:
            dirs = ['top', 'bottom', 'left', 'right']

        priority = {'top': 0, 'bottom': 1, 'left': 2, 'right': 3}
        best = None

        for side in dirs:
            ref, cells = choose_reference(side, C, F_pixels, fr, fc, length, protected)
            if cells is None:
                continue

            count = side_count(C, side)
            blocked = any(arr[r, c] != 0 for r, c in cells)

            if side == 'top':
                margin = C['rmin'] - length
            elif side == 'bottom':
                margin = (h - 1) - (C['rmax'] + length)
            elif side == 'left':
                margin = C['cmin'] - length
            else:
                margin = (w - 1) - (C['cmax'] + length)

            score = (count, blocked, -margin, priority[side])
            if best is None or score < best[0]:
                best = (score, side, ref, cells, length)

        if best is None:
            return None

        _, side, ref, cells, length = best
        return {'side': side, 'ref': ref, 'cells': cells, 'length': length}

    # Generate candidate container/token pairs.
    candidates = []
    for C in clusters:
        # Very small clusters are usually tokens, not containers.
        if C['count'] < 3:
            continue

        for F_color, F_all in pixels_by_color.items():
            if F_color == C['color']:
                continue

            F_inside = [
                p for p in F_all
                if C['rmin'] <= p[0] <= C['rmax'] and C['cmin'] <= p[1] <= C['cmax']
            ]

            if not F_inside:
                continue

            # Container should be larger than the interior token set.
            if C['count'] <= len(F_inside):
                continue

            candidates.append({
                'C': C,
                'F': F_inside,
                'color': F_color
            })

    # Prefer larger containers, then more token pixels.
    candidates.sort(key=lambda x: (x['C']['count'], len(x['F'])), reverse=True)

    selected = []
    used_filler = set()
    protected = set()

    for cand in candidates:
        C = cand['C']
        F_color = cand['color']

        F_filtered = [p for p in cand['F'] if p not in protected and p not in used_filler]
        if not F_filtered:
            continue

        if C['count'] <= len(F_filtered):
            continue

        if any(p in used_filler for p in C['pixels']):
            continue

        if set(C['pixels']) & protected:
            continue

        info = choose_side(C, F_filtered, protected)
        if info is None:
            continue

        if any(cell in protected for cell in info['cells']):
            continue

        selected.append({
            'color': F_color,
            'F': F_filtered,
            'cells': info['cells']
        })

        used_filler.update(F_filtered)
        protected.update(C['pixels'])

    # Apply transformations.
    out = arr.copy()

    # Remove interior tokens.
    for sel in selected:
        for r, c in sel['F']:
            out[r, c] = 0

    # Add collapsed line segments.
    for sel in selected:
        for r, c in sel['cells']:
            if 0 <= r < h and 0 <= c < w and (out[r, c] == 0 or out[r, c] == sel['color']):
                out[r, c] = sel['color']

    return out