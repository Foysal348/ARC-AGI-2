import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by moving 8/9 objects to 9-only anchor clusters.

    Each object containing 8s and 9s is paired with a 9-only anchor cluster
    having the same number of 9s. The object is transformed by one of the
    8 symmetries so that:
      * the object's 9-cluster maps to the anchor 9-cluster,
      * border-colored endpoint pixels face the matching border/barrier side.
    Then the transformed object is placed around the anchor 9s.
    """
    if grid is None or grid.size == 0:
        return grid

    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    # Background is the most frequent color.
    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    # Label 4-connected non-background components.
    mask = g != bg
    labels = np.zeros((h, w), dtype=int)
    comps = []

    for i in range(h):
        for j in range(w):
            if mask[i, j] and labels[i, j] == 0:
                cid = len(comps) + 1
                q = deque([(i, j)])
                labels[i, j] = cid
                pix = []
                while q:
                    r, c = q.popleft()
                    pix.append((r, c))
                    for nr, nc in ((r + 1, c), (r - 1, c),
                                   (r, c + 1), (r, c - 1)):
                        if 0 <= nr < h and 0 <= nc < w:
                            if mask[nr, nc] and labels[nr, nc] == 0:
                                labels[nr, nc] = cid
                                q.append((nr, nc))
                comps.append(pix)

    sources = []
    targets = []
    source_id_set = set()

    for idx, pix in enumerate(comps):
        has8 = any(g[r, c] == 8 for r, c in pix)
        has9 = any(g[r, c] == 9 for r, c in pix)

        if has8 and has9:
            source_id_set.add(idx)
            src9 = [(r, c) for r, c in pix if g[r, c] == 9]
            non9 = [(r, c, int(g[r, c])) for r, c in pix if g[r, c] != 9]
            sources.append({
                "pixels": pix,
                "9s": src9,
                "non9": non9,
            })
        elif has9 and not has8:
            targets.append({
                "pixels": pix,
                "9s": [(r, c) for r, c in pix],
            })

    if not sources or not targets:
        return g.copy()

    # Frame/barrier pixels by color, excluding source objects.
    frame_by_color = {}
    for i in range(h):
        for j in range(w):
            color = int(g[i, j])
            if color == bg or color == 8 or color == 9:
                continue
            if labels[i, j] in source_id_set:
                continue
            frame_by_color.setdefault(color, []).append((i, j))

    # Apply a symmetry to a vector.
    def apply_sym(v, sym):
        dr, dc = v
        if sym == 0:      # identity
            return dr, dc
        if sym == 1:      # 90 deg CW
            return dc, -dr
        if sym == 2:      # 180
            return -dr, -dc
        if sym == 3:      # 90 deg CCW
            return -dc, dr
        if sym == 4:      # reflect vertical axis
            return dr, -dc
        if sym == 5:      # reflect horizontal axis
            return -dr, dc
        if sym == 6:      # reflect main diagonal
            return dc, dr
        if sym == 7:      # reflect anti-diagonal
            return -dc, -dr
        return dr, dc

    def transform_coords(pts, ref_s, ref_t, sym):
        rs, cs = ref_s
        rt, ct = ref_t
        out = []
        for r, c in pts:
            ndr, ndc = apply_sym((r - rs, c - cs), sym)
            out.append((rt + ndr, ct + ndc))
        return out

    def simple_direction(v):
        vr, vc = v
        if abs(vr) < 1e-9 and abs(vc) < 1e-9:
            return None
        if abs(vr) >= abs(vc):
            return "up" if vr < 0 else "down"
        return "left" if vc < 0 else "right"

    def side_match(v, desired, th, tw):
        vr, vc = v
        if desired == "right":
            return vc > 0 and abs(vr) <= max(1.0, th / 2.0 + 1.0)
        if desired == "left":
            return vc < 0 and abs(vr) <= max(1.0, th / 2.0 + 1.0)
        if desired == "up":
            return vr < 0 and abs(vc) <= max(1.0, tw / 2.0 + 1.0)
        if desired == "down":
            return vr > 0 and abs(vc) <= max(1.0, tw / 2.0 + 1.0)
        return False

    def nearest_frame_direction(color, center):
        pts = frame_by_color.get(color, [])
        if not pts:
            return None
        cr, cc = center
        best = min(pts, key=lambda p: (p[0] - cr) ** 2 + (p[1] - cc) ** 2)
        return simple_direction((best[0] - cr, best[1] - cc))

    def bbox_height_width(pts):
        rs = [p[0] for p in pts]
        cs = [p[1] for p in pts]
        return max(rs) - min(rs), max(cs) - min(cs)

    def build_data(sym, ref_s, ref_t, src9, non9):
        t9 = transform_coords(src9, ref_s, ref_t, sym)
        pix_map = {}
        eights = set()
        endpoints = set()

        for r, c, color in non9:
            pos = transform_coords([(r, c)], ref_s, ref_t, sym)[0]
            if not (0 <= pos[0] < h and 0 <= pos[1] < w):
                continue

            old = pix_map.get(pos)
            if old is None:
                pix_map[pos] = color
            elif color != 8 and old == 8:
                pix_map[pos] = color

            if color == 8:
                eights.add(pos)
            else:
                endpoints.add(pos)

        return {
            "sym": sym,
            "ref_s": ref_s,
            "ref_t": ref_t,
            "map": pix_map,
            "eights": eights,
            "endpoints": endpoints,
            "t9": t9,
        }

    def score_candidate(data, target):
        tgt9 = set(target["9s"])
        t9set = set(data["t9"])
        exact = (t9set == tgt9)
        subset = t9set.issubset(tgt9) or tgt9.issubset(t9set)

        if not exact and not subset:
            return None

        score = 10.0 if exact else -5.0

        # Anchor center and anchor bbox.
        rs = [p[0] for p in target["9s"]]
        cs = [p[1] for p in target["9s"]]
        tr, tc = sum(rs) / len(rs), sum(cs) / len(cs)
        th = max(rs) - min(rs)
        tw = max(cs) - min(cs)

        # Endpoint direction score.
        by_color = {}
        for pos in data["endpoints"]:
            col = data["map"].get(pos)
            if col == 8:
                continue
            by_color.setdefault(col, []).append(pos)

        for col, pts in by_color.items():
            cr = sum(p[0] for p in pts) / len(pts)
            cc = sum(p[1] for p in pts) / len(pts)
            v = (cr - tr, cc - tc)
            if abs(v[0]) < 1e-9 and abs(v[1]) < 1e-9:
                continue
            desired = nearest_frame_direction(col, (tr, tc))
            if desired and side_match(v, desired, th, tw):
                score += 2.0

        # Mild penalty for 8s far outside endpoint bbox.
        if data["endpoints"]:
            ers = [p[0] for p in data["endpoints"]]
            ecs = [p[1] for p in data["endpoints"]]
            minr, maxr = min(ers), max(ers)
            minc, maxc = min(ecs), max(ecs)
            for p in data["eights"]:
                if data["map"].get(p) != 8:
                    continue
                if not (minr <= p[0] <= maxr and minc <= p[1] <= maxc):
                    score -= 0.5

        return score

    def fallback_target_index(src):
        if not targets:
            return None
        cr = sum(p[0] for p in src["9s"]) / len(src["9s"])
        cc = sum(p[1] for p in src["9s"]) / len(src["9s"])

        same = [ti for ti, t in enumerate(targets) if len(t["9s"]) == len(src["9s"])]
        pool = same if same else range(len(targets))
        return min(
            pool,
            key=lambda ti: (
                (sum(p[0] for p in targets[ti]["9s"]) / len(targets[ti]["9s"]) - cr) ** 2 +
                (sum(p[1] for p in targets[ti]["9s"]) / len(targets[ti]["9s"]) - cc) ** 2
            )
        )

    candidates = []

    for si, s in enumerate(sources):
        best = None
        best_t = None

        for ti, t in enumerate(targets):
            tgt9set = set(t["9s"])

            for ref_s in s["9s"]:
                for ref_t in t["9s"]:
                    for sym in range(8):
                        data = build_data(sym, ref_s, ref_t, s["9s"], s["non9"])
                        sc = score_candidate(data, t)
                        if sc is None:
                            continue
                        if best is None or sc > best:
                            best = sc
                            best_t = ti
                            best_data = data

        if best is None:
            ti = fallback_target_index(s)
            if ti is not None:
                data = build_data(0, s["9s"][0], targets[ti]["9s"][0], s["9s"], s["non9"])
                candidates.append((-1000.0, si, ti, data))
        else:
            candidates.append((best, si, best_t, best_data))

    # Greedy assignment: highest score first.
    candidates.sort(reverse=True, key=lambda x: x[0])
    assigned_s = [False] * len(sources)
    assigned_t = [False] * len(targets)
    assignments = []

    for score, si, ti, data in candidates:
        if assigned_s[si] or assigned_t[ti]:
            continue
        assignments.append((si, ti, data))
        assigned_s[si] = True
        assigned_t[ti] = True

    # Assign remaining sources to any remaining target as a fallback.
    for si in range(len(sources)):
        if assigned_s[si]:
            continue
        for ti in range(len(targets)):
            if not assigned_t[ti]:
                data = build_data(0, sources[si]["9s"][0], targets[ti]["9s"][0],
                                  sources[si]["9s"], sources[si]["non9"])
                assignments.append((si, ti, data))
                assigned_s[si] = True
                assigned_t[ti] = True
                break

    # Erase all original movable objects.
    out = g.copy()
    cleared = np.zeros((h, w), dtype=bool)
    for s in sources:
        for r, c in s["pixels"]:
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = bg
                cleared[r, c] = True

    # Place transformed objects.
    for si, ti, data in assignments:
        t = targets[ti]
        eights = [p for p in data["eights"] if data["map"].get(p) == 8]
        endpoints = [p for p in data["endpoints"] if data["map"].get(p) != 8]

        # Cleanup used in the examples: for a single-anchor object with
        # endpoints on opposite horizontal/vertical sides, keep only 8s
        # inside the endpoint bounding box.
        if len(t["9s"]) == 1 and endpoints:
            tr = t["9s"][0][0]
            tc = t["9s"][0][1]
            ers = [p[0] for p in endpoints]
            ecs = [p[1] for p in endpoints]
            minr, maxr = min(ers), max(ers)
            minc, maxc = min(ecs), max(ecs)
            horiz = minc < tc < maxc
            vert = minr < tr < maxr
            if horiz or vert:
                eights = [p for p in eights
                          if minr <= p[0] <= maxr and minc <= p[1] <= maxc]

        # Place 8s first, only where cleared/background.
        for r, c in eights:
            if 0 <= r < h and 0 <= c < w:
                if cleared[r, c] or out[r, c] == bg:
                    out[r, c] = 8

        # Place endpoint colors.
        for (r, c), col in data["map"].items():
            if col == 8:
                continue
            if 0 <= r < h and 0 <= c < w:
                if cleared[r, c] or out[r, c] == bg or out[r, c] == col:
                    out[r, c] = col

    # Preserve all anchor 9s.
    for t in targets:
        for r, c in t["9s"]:
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = 9

    return out