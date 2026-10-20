import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform 3x3 colored port symbols connected by gray/yellow wires.

    Rules inferred from the examples:
    * 8 and 5 form wire paths.
    * A 3x3 port may contain:
        - a solid 3x3 block,
        - a "cross" 3x3 pattern:
            0 c 0
            c 0 c
            0 c 0
        - a "corner+center" 3x3 pattern:
            c 0 c
            0 c 0
            c 0 c
        - nothing (empty).
    * Trace continuous wire curves from valid port endpoints:
        - at a corner (degree 2), continue to the only possible forward cell,
        - at a crossing (degree 4), continue straight,
        - at a branch/dead end, stop.
    * If two valid port endpoints are connected by one wire curve, the color is
      placed at both ends:
        - a solid/equivalent input becomes the cross pattern,
        - a corner+center input remains the corner+center pattern,
        - an empty paired port receives the opposite pattern.
    * Unpaired solid 3x3 ports are converted to the cross pattern; unpaired
      corner+center ports remain unchanged.
    """
    g = np.array(grid, dtype=int, copy=True)
    if g.ndim != 2:
        return g

    H, W = g.shape
    if H < 3 or W < 3:
        return g

    # Wire cells are 8 and 5.
    wire = (g == 8) | (g == 5)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def in_bounds(r, c):
        return 0 <= r < H and 0 <= c < W

    def wire_neighbors(r, c):
        res = []
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and wire[nr, nc]:
                res.append((nr, nc))
        return res

    def isolated_ring(r, c, val):
        """
        Check that a 3x3 symbol is not part of a larger same-color region.
        This prevents treating a 3x3 subblock of a larger colored blob as a
        valid symbol.
        """
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if abs(dr) <= 1 and abs(dc) <= 1:
                    continue
                nr, nc = r + dr, c + dc
                if in_bounds(nr, nc) and g[nr, nc] == val:
                    return False
        return True

    def analyze_3x3(r, c):
        """
        Return one of:
          ('empty', 0)
          ('solid', color)
          ('e', color)      edge-center cross
          ('x', color)      corners + center
        or None if the 3x3 is not a valid isolated symbol.
        """
        if not in_bounds(r - 1, c - 1) or not in_bounds(r + 1, c + 1):
            return None

        sub = g[r - 1:r + 2, c - 1:c + 2]

        # Valid port symbols do not contain wire cells.
        if np.any((sub == 8) | (sub == 5)):
            return None

        if np.all(sub == 0):
            return ('empty', 0)

        uniq = np.unique(sub)

        # Solid 3x3.
        if len(uniq) == 1:
            val = int(uniq[0])
            if val == 0:
                return ('empty', 0)
            if val in (8, 5):
                return None
            if isolated_ring(r, c, val):
                return ('solid', val)
            return None

        # Patterns with one nonzero color.
        nonzero = uniq[uniq != 0]
        if len(nonzero) == 1:
            val = int(nonzero[0])
            if val in (8, 5):
                return None

            # Corner + center:
            # c 0 c
            # 0 c 0
            # c 0 c
            if (
                sub[0, 0] == val and sub[0, 2] == val and
                sub[1, 1] == val and
                sub[2, 0] == val and sub[2, 2] == val and
                sub[0, 1] == 0 and sub[1, 0] == 0 and
                sub[1, 2] == 0 and sub[2, 1] == 0
            ):
                if isolated_ring(r, c, val):
                    return ('x', val)

            # Edge-center cross:
            # 0 c 0
            # c 0 c
            # 0 c 0
            if (
                sub[0, 1] == val and
                sub[1, 0] == val and sub[1, 2] == val and
                sub[2, 1] == val and
                sub[0, 0] == 0 and sub[1, 1] == 0 and sub[2, 2] == 0
            ):
                if isolated_ring(r, c, val):
                    return ('e', val)

        return None

    slots = {}

    def add_slot(center, state, color, endpoint=None, port_id=None):
        r, c = center
        if center in slots:
            slot = slots[center]
            if endpoint is not None and not slot.get('is_endpoint', False):
                slot['endpoint'] = endpoint
                slot['port_id'] = port_id
                slot['is_endpoint'] = True
                if state != 'empty':
                    slot['state'] = state
                    slot['color'] = color
            elif state != 'empty' and slot.get('state') == 'empty':
                slot['state'] = state
                slot['color'] = color
            return slot

        slot = {
            'center': center,
            'state': state,
            'color': color,
            'endpoint': endpoint,
            'port_id': port_id,
            'is_endpoint': endpoint is not None,
            'out_pattern': None,
            'out_color': None
        }
        slots[center] = slot
        return slot

    ports = []
    endpoint_to_port = {}

    # Find valid wire endpoints and their associated 3x3 ports.
    for r in range(H):
        for c in range(W):
            if not wire[r, c]:
                continue

            nbrs = wire_neighbors(r, c)
            if len(nbrs) != 1:
                continue

            nr, nc = nbrs[0]
            # Outward direction is opposite to the only inward wire neighbor.
            dr = r - nr
            dc = c - nc
            cr, cc = r + 2 * dr, c + 2 * dc

            analysis = analyze_3x3(cr, cc)
            if analysis is None:
                continue

            state, color = analysis
            color = None if color in (0, None) else int(color)
            center = (cr, cc)

            existing = slots.get(center)
            if existing is not None and existing.get('is_endpoint', False):
                # Two endpoints would point to the same port; keep first.
                continue

            port_id = len(ports)
            endpoint = (r, c)
            add_slot(center, state, color, endpoint=endpoint, port_id=port_id)

            ports.append({
                'id': port_id,
                'endpoint': endpoint,
                'center': center,
                'state': state,
                'color': color
            })
            endpoint_to_port[endpoint] = port_id

    # Also detect isolated 3x3 symbols that are not explicitly associated with
    # an endpoint. This makes the in-place solid -> cross change robust.
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if (r, c) in slots:
                continue
            analysis = analyze_3x3(r, c)
            if analysis is None:
                continue
            state, color = analysis
            if state == 'empty':
                continue
            color = int(color)
            add_slot((r, c), state, color)

    def trace_wire(start_endpoint):
        """
        Follow one continuous wire curve from a port endpoint.
        Returns the port id at the other end, or None if no valid paired port.
        """
        nbrs = wire_neighbors(*start_endpoint)
        if len(nbrs) != 1:
            return None

        prev = start_endpoint
        curr = nbrs[0]
        visited = {prev, curr}
        max_steps = H * W * 2

        for _ in range(max_steps):
            if curr in endpoint_to_port and curr != start_endpoint:
                return endpoint_to_port[curr]

            nbrs = wire_neighbors(*curr)
            dr = curr[0] - prev[0]
            dc = curr[1] - prev[1]

            # Prefer going straight, which is how crossings are resolved.
            straight = (curr[0] + dr, curr[1] + dc)
            if straight in nbrs:
                nxt = straight
            else:
                # Forced corner: exactly one possible forward cell.
                candidates = [n for n in nbrs if n != prev]
                if len(candidates) == 1:
                    nxt = candidates[0]
                else:
                    # Branch/dead end: do not switch curves.
                    return None

            if nxt in visited:
                return None

            visited.add(nxt)
            prev, curr = curr, nxt

        return None

    paired = [False] * len(ports)
    pairs = []

    for i, port in enumerate(ports):
        if paired[i]:
            continue
        j = trace_wire(port['endpoint'])
        if j is not None and j != i and not paired[j]:
            paired[i] = True
            paired[j] = True
            pairs.append((i, j))

    # Initial output state for every detected slot.
    for slot in slots.values():
        state = slot['state']
        if state in ('solid', 'e'):
            slot['out_pattern'] = 'e'
            slot['out_color'] = slot['color']
        elif state == 'x':
            slot['out_pattern'] = 'x'
            slot['out_color'] = slot['color']
        else:
            slot['out_pattern'] = None
            slot['out_color'] = None

    opposite = {'e': 'x', 'x': 'e'}

    def pattern_for_state(state):
        if state in ('solid', 'e'):
            return 'e'
        if state == 'x':
            return 'x'
        return None

    # Apply pair propagation.
    for i, j in pairs:
        p1 = ports[i]
        p2 = ports[j]

        color = p1['color'] if p1['color'] is not None else p2['color']
        if color is None:
            continue

        pat1 = pattern_for_state(p1['state'])
        pat2 = pattern_for_state(p2['state'])

        if pat1 is None:
            pat1 = opposite.get(pat2)
        if pat2 is None:
            pat2 = opposite.get(pat1)

        if pat1 is None or pat2 is None:
            continue

        s1 = slots[p1['center']]
        s2 = slots[p2['center']]

        s1['out_pattern'] = pat1
        s1['out_color'] = color
        s2['out_pattern'] = pat2
        s2['out_color'] = color

    out = g.copy()

    # Clear all regions that will be redrawn.
    for slot in slots.values():
        if slot['out_pattern'] is not None and slot['out_color'] is not None:
            r, c = slot['center']
            out[r - 1:r + 2, c - 1:c + 2] = 0

    # Draw final patterns.
    for slot in slots.values():
        if slot['out_pattern'] is None or slot['out_color'] is None:
            continue

        r, c = slot['center']
        col = int(slot['out_color'])
        pat = slot['out_pattern']

        if pat == 'e':
            out[r - 1, c] = col
            out[r + 1, c] = col
            out[r, c - 1] = col
            out[r, c + 1] = col
        elif pat == 'x':
            out[r - 1, c - 1] = col
            out[r - 1, c + 1] = col
            out[r, c] = col
            out[r + 1, c - 1] = col
            out[r + 1, c + 1] = col

    return out