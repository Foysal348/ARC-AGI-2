import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    if h <= 1 or w <= 1:
        return grid.copy()

    bg = np.bincount(grid.ravel()).argmax()

    # collect non-background cells per color
    cells = {}
    for r in range(h):
        for c in range(w):
            v = int(grid[r, c])
            if v != bg:
                cells.setdefault(v, []).append((r, c))

    if not cells:
        return grid.copy()

    def bbox(cs):
        rs = [p[0] for p in cs]
        cs2 = [p[1] for p in cs]
        return min(rs), max(rs), min(cs2), max(cs2)

    objs = []
    for col, cs in cells.items():
        t, b, l, r = bbox(cs)
        objs.append(dict(color=col, cells=cs, t=t, b=b, l=l, r=r,
                         n=len(cs), hh=b - t + 1, ww=r - l + 1))

    # objects with exactly two cells form a "connector" and go to the bottom
    def is_two(o):
        return o['n'] == 2

    # T-like: a bar of 4+ cells in one row or column
    def has_bar(o):
        rowc = {}
        colc = {}
        for (rr, cc) in o['cells']:
            rowc[rr] = rowc.get(rr, 0) + 1
            colc[cc] = colc.get(cc, 0) + 1
        return (max(rowc.values()) >= 4) or (max(colc.values()) >= 4)

    if any(is_two(o) for o in objs):
        top_set = {o['color'] for o in objs if not is_two(o)}
        bot_set = {o['color'] for o in objs if is_two(o)}
    else:
        # T-shape on the side where the shorter end points up goes top,
        # T-shape with shorter end pointing down goes bottom; everything
        # else goes to the top edge.
        top_set = set()
        bot_set = set()
        for o in objs:
            rowc = {}
            colc = {}
            for (rr, cc) in o['cells']:
                rowc[rr] = rowc.get(rr, 0) + 1
                colc[cc] = colc.get(cc, 0) + 1
            is_t = False
            if max(rowc.values()) >= 4:
                rb = max(rowc, key=lambda k: (rowc[k], -k))
                bar_len = rowc[rb]
                rest = o['n'] - bar_len
                if rest > 0 and rest < bar_len:
                    rows_of_rest = sorted(rr for rr in rowc if rr != rb)
                    if all(rr > rb for rr in rows_of_rest):
                        is_t = True  # stem below
                        bot_set.add(o['color'])
                    elif all(rr < rb for rr in rows_of_rest):
                        is_t = True  # stem above
                        top_set.add(o['color'])
            if (not is_t) and max(colc.values()) >= 4:
                cb = max(colc, key=lambda k: (colc[k], -k))
                bar_len = colc[cb]
                rest = o['n'] - bar_len
                if rest > 0 and rest < bar_len:
                    cols_of_rest = sorted(cc for (rr, cc) in o['cells'] if cc != cb)
                    if all(cc > cb for cc in cols_of_rest):
                        is_t = True
                        top_set.add(o['color'])
                    elif all(cc < cb for cc in cols_of_rest):
                        is_t = True
                        top_set.add(o['color'])
            if not is_t:
                top_set.add(o['color'])

    out = np.full((h, w), bg, dtype=np.int64)
    for o in objs:
        if o['color'] in top_set:
            shift = -o['t']
        else:
            shift = (h - 1) - o['b']
        for (rr, cc) in o['cells']:
            nrr = rr + shift
            if 0 <= nrr < h:
                out[nrr, cc] = o['color']

    return out