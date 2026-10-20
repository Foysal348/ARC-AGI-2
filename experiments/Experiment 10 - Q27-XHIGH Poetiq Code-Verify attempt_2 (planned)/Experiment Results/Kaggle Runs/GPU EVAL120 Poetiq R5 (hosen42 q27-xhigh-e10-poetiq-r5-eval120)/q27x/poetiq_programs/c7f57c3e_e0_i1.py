import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform framed objects by swapping their two observed "states".

    For each connected object:
      - color 1 is treated as the stable frame.
      - non-1 cells outside the bounding box of the 1-frame are the variable cap.
      - non-1 cells inside that frame bbox are the interior/core.

    The puzzle contains two variants/states of the same object type.
    The transform swaps the cap side/color and the core color between
    the two states, while preserving the frame and the object's scale.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.size == 0:
        return grid

    H, W = grid.shape
    bg = int(np.bincount(grid.ravel()).argmax())
    out = grid.copy()

    # 8-connected components of non-background cells.
    nonbg = (grid != bg)
    visited = np.zeros((H, W), dtype=bool)
    components = []

    for i in range(H):
        for j in range(W):
            if nonbg[i, j] and not visited[i, j]:
                stack = [(i, j)]
                visited[i, j] = True
                cells = []
                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W and nonbg[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                if cells:
                    components.append(cells)

    FRAME = 1
    infos = []

    for cells in components:
        frame_cells = [(r, c) for r, c in cells if grid[r, c] == FRAME]
        if not frame_cells:
            continue

        fr_min_r = min(r for r, c in frame_cells)
        fr_max_r = max(r for r, c in frame_cells)
        fr_min_c = min(c for r, c in frame_cells)
        fr_max_c = max(c for r, c in frame_cells)

        cap_cells = []
        core_cells = []

        for r, c in cells:
            v = int(grid[r, c])
            if v == FRAME:
                continue
            if r < fr_min_r or r > fr_max_r or c < fr_min_c or c > fr_max_c:
                cap_cells.append((r, c, v))
            else:
                core_cells.append((r, c, v))

        if not cap_cells:
            continue

        # Determine which side of the frame the cap is on.
        counts = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        for r, c, v in cap_cells:
            if r < fr_min_r:
                counts['top'] += 1
            elif r > fr_max_r:
                counts['bottom'] += 1
            elif c < fr_min_c:
                counts['left'] += 1
            elif c > fr_max_c:
                counts['right'] += 1

        side = None
        best = 0
        for s in ('top', 'bottom', 'left', 'right'):
            if counts[s] > best:
                best = counts[s]
                side = s

        if side is None:
            continue

        vertical = side in ('top', 'bottom')
        lat_orig = min(c for r, c, v in cap_cells) if vertical else min(r for r, c, v in cap_cells)

        # Build an oriented cap pattern:
        # row/axis 0 is the side adjacent to the frame.
        h = 0
        w = 0
        for r, c, v in cap_cells:
            if vertical:
                d = fr_min_r - r if side == 'top' else r - fr_max_r
                l = c - lat_orig
            else:
                d = fr_min_c - c if side == 'left' else c - fr_max_c
                l = r - lat_orig
            if d >= 1:
                h = max(h, d)
                w = max(w, l + 1)

        if h == 0 or w == 0:
            h, w = 1, 1

        pattern = np.zeros((h, w), dtype=int)
        for r, c, v in cap_cells:
            if vertical:
                d = fr_min_r - r if side == 'top' else r - fr_max_r
                l = c - lat_orig
            else:
                d = fr_min_c - c if side == 'left' else c - fr_max_c
                l = r - lat_orig

            d = min(max(1, d), h)
            l = min(max(0, l), w - 1)
            pattern[d - 1, l] = v

        cap_colors = tuple(sorted(set(int(x) for x in pattern[pattern != 0].tolist())))
        if not cap_colors:
            continue

        core_values = [v for r, c, v in core_cells]
        core_colors = tuple(sorted(set(core_values))) if core_values else ()
        core_color = int(np.bincount(core_values).argmax()) if core_values else 0

        sig = (side, cap_colors, core_colors)
        infos.append({
            'cap_cells': cap_cells,
            'core_cells': core_cells,
            'fr_min_r': fr_min_r,
            'fr_max_r': fr_max_r,
            'fr_min_c': fr_min_c,
            'fr_max_c': fr_max_c,
            'side': side,
            'lat_orig': lat_orig,
            'h': h,
            'w': w,
            'pattern': pattern,
            'cap_colors': cap_colors,
            'core_values': core_values,
            'core_color': core_color,
            'sig': sig
        })

    if not infos:
        return out

    # Find the two states and swap them.
    sig_counts = {}
    for info in infos:
        sig_counts[info['sig']] = sig_counts.get(info['sig'], 0) + 1

    if len(sig_counts) < 2:
        return out

    if len(sig_counts) == 2:
        s1, s2 = list(sig_counts.keys())
        mapping = {s1: s2, s2: s1}
    else:
        ordered = sorted(sig_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        s1, s2 = ordered[0][0], ordered[1][0]
        mapping = {s1: s2, s2: s1}

    relevant = set(mapping.keys())
    state_protos = {s: [] for s in relevant}
    state_core_color = {s: 0 for s in relevant}

    for info in infos:
        if info['sig'] in state_protos:
            state_protos[info['sig']].append(info)

    for s, protos in state_protos.items():
        cores = []
        for p in protos:
            cores.extend(p['core_values'])
        if cores:
            state_core_color[s] = int(np.bincount(cores).argmax())

    def scale_pattern(proto, th, tw):
        if proto is None or proto.size == 0:
            return np.zeros((th, tw), dtype=int)
        ph, pw = proto.shape
        if ph == th and pw == tw:
            return proto.copy()

        scaled = np.zeros((th, tw), dtype=int)
        for i in range(th):
            pi = int((i * (ph - 1)) / (th - 1)) if th > 1 else 0
            for j in range(tw):
                pj = int((j * (pw - 1)) / (tw - 1)) if tw > 1 else 0
                scaled[i, j] = proto[pi, pj]
        return scaled

    # Apply the state swap.
    for info in infos:
        target_sig = mapping.get(info['sig'])
        if target_sig is None or target_sig == info['sig']:
            continue

        # Remove original cap.
        for r, c, v in info['cap_cells']:
            out[r, c] = bg

        # Recolor core to the target state's core color.
        target_core = state_core_color[target_sig]
        if target_core != 0:
            for r, c, v in info['core_cells']:
                out[r, c] = target_core

        target_side = target_sig[0]
        target_cap_colors = target_sig[1]

        h = info['h']
        w = info['w']
        mask = info['pattern'] != 0
        target_vertical = target_side in ('top', 'bottom')

        if len(target_cap_colors) == 1:
            uniform_color = target_cap_colors[0]
            scaled = None
            fallback = 0
        else:
            uniform_color = 0
            protos = state_protos.get(target_sig, [])
            best = None
            bestdiff = None
            for p in protos:
                diff = abs(p['h'] * p['w'] - h * w) + abs(p['h'] - h) + abs(p['w'] - w)
                if bestdiff is None or diff < bestdiff:
                    bestdiff = diff
                    best = p

            scaled = scale_pattern(best['pattern'], h, w) if best is not None else np.zeros((h, w), dtype=int)
            nz = scaled[scaled != 0]
            fallback = int(np.bincount(nz).argmax()) if nz.size else (target_cap_colors[-1] if target_cap_colors else 0)

        # Place the new cap on the target side, preserving the current object's scale.
        for i in range(h):
            for j in range(w):
                if not mask[i, j]:
                    continue

                d = i + 1
                if target_vertical:
                    c = info['lat_orig'] + j
                    if target_side == 'top':
                        r = info['fr_min_r'] - d
                    else:
                        r = info['fr_max_r'] + d
                    if not (0 <= r < H and 0 <= c < W):
                        continue
                else:
                    r = info['lat_orig'] + j
                    if target_side == 'left':
                        c = info['fr_min_c'] - d
                    else:
                        c = info['fr_max_c'] + d
                    if not (0 <= r < H and 0 <= c < W):
                        continue

                if len(target_cap_colors) == 1:
                    color = uniform_color
                else:
                    color = int(scaled[i, j]) if scaled[i, j] != 0 else fallback

                out[r, c] = color

    return out