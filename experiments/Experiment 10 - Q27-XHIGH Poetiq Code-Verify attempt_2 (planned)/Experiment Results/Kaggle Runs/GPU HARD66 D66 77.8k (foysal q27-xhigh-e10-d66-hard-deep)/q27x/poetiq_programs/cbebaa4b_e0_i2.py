import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered colored frames/pieces around the fixed color-4 anchor.

    The idea:
      * Color 2 cells are "handles" attached to a nearby colored piece.
      * Each colored component, together with its nearby 2-handles, is treated
        as a rigid local pattern.
      * The color-4 component is kept in its original position.
      * All other patterns are translated without rotation/reflection.
      * The correct translation is found by a small beam search that prefers:
          - coincidences of 2-handles from different pieces,
          - handle cells adjacent to existing colored cells,
          - compact placement close to the anchor.
      * Colored cells may not overlap; 2-cells may overlap and are kept as 2s.
    """
    if grid is None or grid.size == 0 or len(grid.shape) < 2:
        return grid

    h, w = grid.shape
    A = grid.tolist()
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    # ------------------------------------------------------------------
    # Small state / piece containers.
    # ------------------------------------------------------------------
    class Piece:
        __slots__ = (
            "color", "r0", "c0", "h", "w",
            "center_dr", "center_dc",
            "colored_rel", "handles_rel",
            "min_dr", "max_dr", "min_dc", "max_dc"
        )

        def __init__(self, color, r0, c0, colored_rel, handles_rel):
            self.color = color
            self.r0 = r0
            self.c0 = c0
            self.colored_rel = colored_rel
            self.handles_rel = handles_rel

            self.h = max(r for r, _ in colored_rel) + 1
            self.w = max(c for _, c in colored_rel) + 1
            self.center_dr = self.h // 2
            self.center_dc = self.w // 2

            all_rel = colored_rel + handles_rel
            self.min_dr = min(r for r, _ in all_rel)
            self.max_dr = max(r for r, _ in all_rel)
            self.min_dc = min(c for _, c in all_rel)
            self.max_dc = max(c for _, c in all_rel)

    class State:
        __slots__ = (
            "colored", "two",
            "colored_cells", "two_cells",
            "total_score", "placements"
        )

        def __init__(self, colored, two, colored_cells, two_cells,
                     total_score, placements):
            self.colored = colored
            self.two = two
            self.colored_cells = colored_cells
            self.two_cells = two_cells
            self.total_score = total_score
            self.placements = placements

    # ------------------------------------------------------------------
    # 1. Extract colored components.
    # ------------------------------------------------------------------
    visited = [[False] * w for _ in range(h)]
    raw_pieces = []

    for r in range(h):
        for c in range(w):
            val = A[r][c]
            if val in (0, 2) or visited[r][c]:
                continue

            color = val
            stack = [(r, c)]
            cells = []

            while stack:
                rr, cc = stack.pop()
                if visited[rr][cc] or A[rr][cc] != color:
                    continue
                visited[rr][cc] = True
                cells.append((rr, cc))

                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h and 0 <= nc < w
                        and not visited[nr][nc]
                        and A[nr][nc] == color
                    ):
                        stack.append((nr, nc))

            if cells:
                r0 = min(r for r, _ in cells)
                c0 = min(c for _, c in cells)
                raw_pieces.append({
                    "color": color,
                    "cells": cells,
                    "handles": [],
                    "r0": r0,
                    "c0": c0,
                })

    if not raw_pieces:
        return np.zeros_like(grid)

    # ------------------------------------------------------------------
    # 2. Assign nearby 2-cells to the closest colored component.
    #    Then discard any handle that is not locally 4-connected to it.
    # ------------------------------------------------------------------
    for r in range(h):
        for c in range(w):
            if A[r][c] != 2:
                continue

            best = None
            best_key = None

            for pi, comp in enumerate(raw_pieces):
                min_d = 10**9
                for cr, cc in comp["cells"]:
                    d = abs(r - cr) + abs(c - cc)
                    if d < min_d:
                        min_d = d

                if min_d > 2:
                    continue

                key = (min_d, comp["color"], comp["r0"], comp["c0"], pi)
                if best is None or key < best_key:
                    best = pi
                    best_key = key

            if best is not None:
                raw_pieces[best]["handles"].append((r, c))

    pieces = []
    for comp in raw_pieces:
        r0 = comp["r0"]
        c0 = comp["c0"]

        colored_rel = sorted({(r - r0, c - c0) for r, c in comp["cells"]})
        handles_rel = sorted({(r - r0, c - c0) for r, c in comp["handles"]})

        if not colored_rel:
            continue

        # Keep only handles connected to the colored component in the local pattern.
        all_local = set(colored_rel) | set(handles_rel)
        connected = set(colored_rel)
        stack = list(colored_rel)

        while stack:
            r, c = stack.pop()
            for dr, dc in dirs:
                nb = (r + dr, c + dc)
                if nb in all_local and nb not in connected:
                    connected.add(nb)
                    stack.append(nb)

        handles_rel = [x for x in handles_rel if x in connected]

        pieces.append(Piece(comp["color"], r0, c0, colored_rel, handles_rel))

    if not pieces:
        return np.zeros_like(grid)

    # ------------------------------------------------------------------
    # 3. Choose the fixed anchor: the color-4 piece if present.
    # ------------------------------------------------------------------
    anchor_idx = -1
    for i, p in enumerate(pieces):
        if p.color == 4:
            if (
                anchor_idx == -1
                or len(p.colored_rel) > len(pieces[anchor_idx].colored_rel)
            ):
                anchor_idx = i

    if anchor_idx == -1:
        anchor_idx = max(
            range(len(pieces)),
            key=lambda i: (len(pieces[i].colored_rel), len(pieces[i].handles_rel))
        )

    anchor = pieces[anchor_idx]
    others = sorted(
        [p for i, p in enumerate(pieces) if i != anchor_idx],
        key=lambda p: (p.color, -len(p.colored_rel))
    )
    ordered = [anchor] + others
    P = len(ordered)

    # ------------------------------------------------------------------
    # 4. Initial state: anchor in its original position.
    # ------------------------------------------------------------------
    anchor_off = (anchor.r0, anchor.c0)
    anchor_center = (
        anchor_off[0] + anchor.center_dr,
        anchor_off[1] + anchor.center_dc
    )

    init_colored = np.zeros((h, w), dtype=np.int8)
    init_two = np.zeros((h, w), dtype=np.int8)
    init_colored_cells = []
    init_two_cells = []

    for dr, dc in anchor.colored_rel:
        r, c = anchor_off[0] + dr, anchor_off[1] + dc
        if 0 <= r < h and 0 <= c < w:
            init_colored[r, c] = anchor.color
            init_colored_cells.append((r, c))

    for dr, dc in anchor.handles_rel:
        r, c = anchor_off[0] + dr, anchor_off[1] + dc
        if 0 <= r < h and 0 <= c < w and init_colored[r, c] == 0:
            if init_two[r, c] == 0:
                init_two_cells.append((r, c))
            init_two[r, c] += 1

    init_state = State(
        init_colored,
        init_two,
        init_colored_cells,
        init_two_cells,
        0,
        {0: anchor_off}
    )

    # ------------------------------------------------------------------
    # 5. Helper functions for scoring and placing pieces.
    # ------------------------------------------------------------------
    OVERLAP_W = 10000
    CONTACT_COLOR_W = 200
    CONTACT_TWO_W = 20
    COLORED_TWO_W = 20
    COLORED_COLOR_W = 10

    def evaluate(p, off, state):
        """Return (score, dist, overlap, contact_color) if valid."""
        if (
            off[0] + p.min_dr < 0 or off[0] + p.max_dr >= h
            or off[1] + p.min_dc < 0 or off[1] + p.max_dc >= w
        ):
            return None

        col_arr = state.colored
        two_arr = state.two

        # Colored cells cannot overlap any occupied cell.
        for dr, dc in p.colored_rel:
            r = off[0] + dr
            c = off[1] + dc
            if col_arr[r, c] != 0 or two_arr[r, c] != 0:
                return None

        # Handles cannot cover colored cells, but may overlap handles.
        for dr, dc in p.handles_rel:
            r = off[0] + dr
            c = off[1] + dc
            if col_arr[r, c] != 0:
                return None

        overlap = 0
        contact_color = 0
        contact_two = 0
        colored_two = 0
        colored_color = 0
        connected = False

        # Score handle behavior.
        for dr, dc in p.handles_rel:
            r = off[0] + dr
            c = off[1] + dc

            if two_arr[r, c] > 0:
                overlap += two_arr[r, c]
                connected = True

            hc = False
            ht = False
            for ndr, ndc in dirs:
                nr, nc = r + ndr, c + ndc
                if 0 <= nr < h and 0 <= nc < w:
                    if col_arr[nr, nc] != 0:
                        hc = True
                    elif two_arr[nr, nc] != 0:
                        ht = True

            if hc:
                contact_color += 1
                connected = True
            if ht:
                contact_two += 1
                connected = True

        # Score colored-cell behavior.
        for dr, dc in p.colored_rel:
            r = off[0] + dr
            c = off[1] + dc

            ct = False
            cc = False
            for ndr, ndc in dirs:
                nr, nc = r + ndr, c + ndc
                if 0 <= nr < h and 0 <= nc < w:
                    if col_arr[nr, nc] != 0:
                        cc = True
                    elif two_arr[nr, nc] != 0:
                        ct = True

            if ct:
                colored_two += 1
                connected = True
            if cc:
                colored_color += 1
                connected = True

        if not connected:
            return None

        dist = (
            abs(off[0] + p.center_dr - anchor_center[0])
            + abs(off[1] + p.center_dc - anchor_center[1])
        )

        score = (
            OVERLAP_W * overlap
            + CONTACT_COLOR_W * contact_color
            + CONTACT_TWO_W * contact_two
            + COLORED_TWO_W * colored_two
            + COLORED_COLOR_W * colored_color
            - dist
        )

        return score, dist, overlap, contact_color

    def generate_candidates(p, state):
        """Generate and score plausible placements for p in current state."""
        seen = set()
        results = []

        def consider(off):
            if off in seen:
                return
            seen.add(off)

            if (
                off[0] + p.min_dr < 0 or off[0] + p.max_dr >= h
                or off[1] + p.min_dc < 0 or off[1] + p.max_dc >= w
            ):
                return

            ev = evaluate(p, off, state)
            if ev is not None:
                results.append((off, ev[0], ev[1], ev[2], ev[3]))

        # Primary candidates: handles touching existing colored cells,
        # overlapping existing handles, or adjacent to existing handles.
        if p.handles_rel:
            for hr, hc in p.handles_rel:
                for er, ec in state.colored_cells:
                    for dr, dc in dirs:
                        consider((er + dr - hr, ec + dc - hc))

                for er, ec in state.two_cells:
                    consider((er - hr, ec - hc))
                    for dr, dc in dirs:
                        consider((er + dr - hr, ec + dc - hc))

        # Fallback / reinforcement candidates: colored cells adjacent to
        # existing colored cells or handles.
        if len(results) < 5:
            for cr, cc in p.colored_rel:
                for er, ec in state.colored_cells:
                    for dr, dc in dirs:
                        consider((er + dr - cr, ec + dc - cc))

                for er, ec in state.two_cells:
                    for dr, dc in dirs:
                        consider((er + dr - cr, ec + dc - cc))

        # Sort by score, then compactness, then useful overlaps/contacts.
        results.sort(key=lambda x: (
            -x[1],
            x[2],
            -x[3],
            -x[4],
            x[0][0],
            x[0][1]
        ))

        if P > 5:
            top_k = 10
        else:
            top_k = 15

        return [(off, (score, dist, overlap, contact))
                for off, score, dist, overlap, contact in results[:top_k]]

    def make_state(state, pid, off, ev):
        """Place a piece and return the new State."""
        p = ordered[pid]

        new_colored = state.colored.copy()
        new_two = state.two.copy()
        new_colored_cells = state.colored_cells.copy()
        new_two_cells = state.two_cells.copy()

        for dr, dc in p.colored_rel:
            r = off[0] + dr
            c = off[1] + dc
            new_colored[r, c] = p.color
            new_colored_cells.append((r, c))

        for dr, dc in p.handles_rel:
            r = off[0] + dr
            c = off[1] + dc
            if new_two[r, c] == 0:
                new_two_cells.append((r, c))
            new_two[r, c] += 1

        new_placements = state.placements.copy()
        new_placements[pid] = off

        return State(
            new_colored,
            new_two,
            new_colored_cells,
            new_two_cells,
            state.total_score + ev[0],
            new_placements
        )

    def bbox_area(state):
        rmin, rmax = h, -1
        cmin, cmax = w, -1

        for r, c in state.colored_cells:
            if r < rmin: rmin = r
            if r > rmax: rmax = r
            if c < cmin: cmin = c
            if c > cmax: cmax = c

        for r, c in state.two_cells:
            if r < rmin: rmin = r
            if r > rmax: rmax = r
            if c < cmin: cmin = c
            if c > cmax: cmax = c

        if rmax < 0 or cmax < 0:
            return 0

        return (rmax - rmin + 1) * (cmax - cmin + 1)

    def sum_dist(state):
        s = 0
        for pid, off in state.placements.items():
            p = ordered[pid]
            s += (
                abs(off[0] + p.center_dr - anchor_center[0])
                + abs(off[1] + p.center_dc - anchor_center[1])
            )
        return s

    def collision_only(p, off, state):
        """Used only in a last-resort fallback."""
        if (
            off[0] + p.min_dr < 0 or off[0] + p.max_dr >= h
            or off[1] + p.min_dc < 0 or off[1] + p.max_dc >= w
        ):
            return False

        col = state.colored
        two = state.two

        for dr, dc in p.colored_rel:
            r = off[0] + dr
            c = off[1] + dc
            if col[r, c] != 0 or two[r, c] != 0:
                return False

        for dr, dc in p.handles_rel:
            r = off[0] + dr
            c = off[1] + dc
            if col[r, c] != 0:
                return False

        return True

    # ------------------------------------------------------------------
    # 6. Beam search over piece placements.
    # ------------------------------------------------------------------
    if P > 5:
        BEAM_WIDTH = 20
    else:
        BEAM_WIDTH = 40

    beam = [init_state]

    for _ in range(P - 1):
        all_states = []

        for st in beam:
            for pid in range(1, P):
                if pid in st.placements:
                    continue

                for off, ev in generate_candidates(ordered[pid], st):
                    new_st = make_state(st, pid, off, ev)
                    if new_st is not None:
                        all_states.append(new_st)

        if not all_states:
            break

        # Deduplicate identical placement configurations.
        best_by_key = {}
        for st in all_states:
            key = tuple(st.placements.get(i) for i in range(P))
            old = best_by_key.get(key)
            if old is None or st.total_score > old.total_score:
                best_by_key[key] = st

        candidates = list(best_by_key.values())
        if not candidates:
            break

        candidates.sort(key=lambda st: (
            -st.total_score,
            bbox_area(st),
            sum_dist(st)
        ))

        beam = candidates[:BEAM_WIDTH]

    # ------------------------------------------------------------------
    # 7. Choose the best full/partial state.
    # ------------------------------------------------------------------
    full = [st for st in beam if len(st.placements) == P]

    if not full:
        # Greedy fallback.
        st = init_state
        for _ in range(P - 1):
            best = None

            for pid in range(1, P):
                if pid in st.placements:
                    continue

                cands = generate_candidates(ordered[pid], st)
                if not cands:
                    continue

                off, ev = cands[0]
                key = (-ev[0], ev[1], pid)

                if best is None or key < best[0]:
                    best = (key, pid, off, ev)

            if best is None:
                break

            _, pid, off, ev = best
            st = make_state(st, pid, off, ev)

        if len(st.placements) > max((len(x.placements) for x in beam), default=0):
            full = [st]
        else:
            beam.sort(key=lambda s: (
                -s.total_score,
                bbox_area(s),
                sum_dist(s)
            ))
            full = beam[:1] if beam else [init_state]

    best_state = min(full, key=lambda st: (
        -st.total_score,
        bbox_area(st),
        sum_dist(st)
    ))

    # Last-resort: if some piece is still missing, try its original spot.
    for pid in range(1, P):
        if pid in best_state.placements:
            continue

        p = ordered[pid]
        off = (p.r0, p.c0)

        if collision_only(p, off, best_state):
            dist = (
                abs(off[0] + p.center_dr - anchor_center[0])
                + abs(off[1] + p.center_dc - anchor_center[1])
            )
            best_state = make_state(best_state, pid, off, (0, dist, 0, 0))

    # ------------------------------------------------------------------
    # 8. Render output grid.
    # ------------------------------------------------------------------
    out = np.zeros((h, w), dtype=grid.dtype)

    col = best_state.colored
    two = best_state.two

    mask = col != 0
    out[mask] = col[mask]

    mask = two > 0
    out[mask] = 2

    return out