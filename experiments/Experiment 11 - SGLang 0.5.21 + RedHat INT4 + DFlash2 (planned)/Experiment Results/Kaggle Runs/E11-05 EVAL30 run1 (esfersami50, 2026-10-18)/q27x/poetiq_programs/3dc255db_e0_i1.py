import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform several two-colored "funnel + payload" objects.

    In each relevant connected object there is a larger container color and a
    smaller payload color.  The payload is removed and replaced by a straight
    line of the same payload color.  The line extends outward from the container
    tip on the side opposite the payload's original position.

    The line length is the payload's bounding-box extent in the line direction:
      - vertical line: payload height
      - horizontal line: payload width

    Components with only one color are left unchanged.
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape
    out = grid.copy()

    visited = np.zeros((H, W), dtype=bool)

    # 8-connected components over non-zero cells.
    def neighbors(r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    yield nr, nc

    def component_cells(start_r, start_c):
        q = deque([(start_r, start_c)])
        visited[start_r, start_c] = True
        cells = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for nr, nc in neighbors(r, c):
                if not visited[nr, nc] and grid[nr, nc] != 0:
                    visited[nr, nc] = True
                    q.append((nr, nc))
        return cells

    for r in range(H):
        for c in range(W):
            if not visited[r, c] and grid[r, c] != 0:
                cells = component_cells(r, c)

                color_counts = {}
                for rr, cc in cells:
                    color_counts[grid[rr, cc]] = color_counts.get(grid[rr, cc], 0) + 1

                # Only process objects containing exactly two colors.
                if len(color_counts) != 2:
                    continue

                # The container is the more numerous color.
                container_color, content_color = max(color_counts.items(), key=lambda kv: kv[1])
                if container_color == content_color:
                    continue

                cont = [(rr, cc) for rr, cc in cells if grid[rr, cc] == container_color]
                cont_cells = [(rr, cc) for rr, cc in cells if grid[rr, cc] == content_color]

                # Remove the original payload.
                for rr, cc in cont_cells:
                    out[rr, cc] = 0

                # Centroids.
                cr = sum(r for r, _ in cont) / len(cont)
                cc = sum(c for _, c in cont) / len(cont)
                pr = sum(r for r, _ in cont_cells) / len(cont_cells)
                pc = sum(c for _, c in cont_cells) / len(cont_cells)

                dy = pr - cr
                dx = pc - cc

                # Direction: away from the payload side.
                if abs(dy) >= abs(dx):
                    if dy > 0:
                        dr, dc = -1, 0     # up
                    else:
                        dr, dc = 1, 0      # down
                    line_len = max(r for r, _ in cont_cells) - min(r for r, _ in cont_cells) + 1
                else:
                    if dx > 0:
                        dr, dc = 0, -1     # left
                    else:
                        dr, dc = 0, 1      # right
                    line_len = max(c for _, c in cont_cells) - min(c for _, c in cont_cells) + 1

                # Anchor: container tip in the chosen outward direction.
                if dr == -1:      # up: minimum row
                    candidates = [cell for cell in cont if cell[0] == min(x[0] for x in cont)]
                    anchor = min(candidates, key=lambda cell: abs(cell[1] - pc))
                elif dr == 1:     # down: maximum row
                    candidates = [cell for cell in cont if cell[0] == max(x[0] for x in cont)]
                    anchor = min(candidates, key=lambda cell: abs(cell[1] - pc))
                elif dc == -1:    # left: minimum column
                    candidates = [cell for cell in cont if cell[1] == min(x[1] for x in cont)]
                    anchor = min(candidates, key=lambda cell: abs(cell[0] - pr))
                else:             # right: maximum column
                    candidates = [cell for cell in cont if cell[1] == max(x[1] for x in cont)]
                    anchor = min(candidates, key=lambda cell: abs(cell[0] - pr))

                ar, ac = anchor
                for i in range(1, line_len + 1):
                    nr = ar + dr * i
                    nc = ac + dc * i
                    if 0 <= nr < H and 0 <= nc < W:
                        # Only write into empty cells or cells already of the payload color.
                        if out[nr, nc] in (0, content_color):
                            out[nr, nc] = content_color

    return out