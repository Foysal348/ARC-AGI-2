import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    bg = 0
    colors, cnt = np.unique(grid, return_counts=True)
    for c, n in zip(colors, cnt):
        if n == max(cnt):
            bg = int(c)
            break

    out = grid.copy()

    # find seed pixels: non-background, non-4, not touching any 4 pixel
    seeds = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == bg or grid[r, c] == 4:
                continue
            near4 = False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == 4:
                        near4 = True
            if not near4:
                seeds.append((r, c, int(grid[r, c])))

    mask = (grid != bg)
    lab = np.zeros((h, w), dtype=int)
    cid = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] and lab[r, c] == 0:
                cid += 1
                stack = [(r, c)]
                lab[r, c] = cid
                while stack:
                    a, b = stack.pop()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + dr, b + dc
                        if 0 <= na < h and 0 <= nb < w and mask[na, nb] and lab[na, nb] == 0:
                            lab[na, nb] = cid
                            stack.append((na, nb))

    bars = []
    for cid2 in range(1, cid + 1):
        rs = [r for r in range(h) for c in range(w) if lab[r, c] == cid2]
        cs = [c for r in rs for c in range(w) if lab[r, c] == cid2]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        rh, rw = r1 - r0 + 1, c1 - c0 + 1
        if rh == 2 and rw >= 3:
            if all(grid[r0, c] != 4 for c in range(c0, c1 + 1)) and all(grid[r1, c] != 4 for c in range(c0, c1 + 1)):
                continue
            bars.append({'orient': 'h', 'row': r0, 'col0': c0, 'col1': c1})
        elif rw == 2 and rh >= 3:
            if all(grid[r, c0] != 4 for r in range(r0, r1 + 1)) and all(grid[r, c1] != 4 for r in range(r0, r1 + 1)):
                continue
            bars.append({'orient': 'v', 'col': c0, 'row0': r0, 'row1': r1})
        else:
            continue

    seed_colors = sorted(set(x[2] for x in seeds))
    n = len(seed_colors)
    if len(bars) > 0 and n > 0:
        L = next(b['row1'] - b['row0'] + 1 for b in bars if b['orient'] == 'v') if any(b['orient'] == 'v' for b in bars) else next(b['col1'] - b['col0'] + 1 for b in bars)
        M = max(1, L - 1)
        used = set()
        color_to_pos = {}
        for col in sorted(seed_colors, reverse=True):
            r, c = next((sr, sc) for sr, sc, scol in seeds if scol == col)
            for cand in (r % M, c % M, (r + c) % M):
                if cand == 0:
                    continue
                if cand not in used:
                    used.add(cand)
                    color_to_pos[col] = cand
                    break
        for col in seed_colors:
            if col not in color_to_pos:
                for cand in range(1, L):
                    if cand not in used:
                        used.add(cand)
                        color_to_pos[col] = cand
                        break

        for b in bars:
            if b['orient'] == 'h':
                # project up from the plain (top) row
                for k, off in enumerate(range(1, L)):
                    if off in color_to_pos.values():
                        col = [cc for cc, vv in color_to_pos.items() if vv == off][0]
                        pos = b['col0'] + off
                        for rr in range(0, b['row']):
                            out[rr, pos] = col
            else:
                # project from the plain column, direction = away from the capped column
                cc0 = b['col']
                cc1 = b['col'] + 1
                cap0 = sum(1 for rr in range(b['row0'], b['row1'] + 1) if grid[rr, cc0] == 4)
                cap1 = sum(1 for rr in range(b['row0'], b['row1'] + 1) if grid[rr, cc1] == 4)
                proj_col = cc0 if cap0 <= cap1 else cc1
                direction = -1 if proj_col == cc0 else 1
                ref = b['row0'] if (grid[b['row0'], cc0] == 4 or grid[b['row0'], cc1] == 4) else b['row1']
                step = 1 if ref == b['row0'] else -1
                for off in range(1, L):
                    if off in color_to_pos.values():
                        col = [cc for cc, vv in color_to_pos.items() if vv == off][0]
                        rr = ref + step * off
                        if direction == -1:
                            for ccc in range(0, proj_col):
                                out[rr, ccc] = col
                        else:
                            for ccc in range(proj_col + 1, w):
                                out[rr, ccc] = col
    return out