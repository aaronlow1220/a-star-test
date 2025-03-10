from typing import List, Tuple, Dict, Set
import numpy as np
from math import dist


def create_node(
    position: Tuple[int, int],
    g: float = float("inf"),
    h: float = 0.0,
    parent: Dict = None,
) -> Dict:
    return {"position": position, "g": g, "h": h, "f": g + h, "parent": parent}


# Euclidean distance
def calculate_heuristic(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    x1, y1 = pos1
    x2, y2 = pos2
    return dist((x1, y1), (x2, y2))


def get_valid_neighbors(
    grid: np.ndarray, position: Tuple[int, int]
) -> List[Tuple[int, int]]:
    row, col = position
    num_rows, num_cols = grid.shape

    moves = [
        (1, 0),
        (-1, 0),  # Vertical moves (down, up)
        (0, 1),
        (0, -1),  # Horizontal moves (right, left)
        (1, 1),
        (-1, -1),  # Diagonal moves (bottom-right, top-left)
        (1, -1),
        (-1, 1),  # Diagonal moves (bottom-left, top-right)
    ]

    valid_neighbors = []
    for dr, dc in moves:
        r, c = row + dr, col + dc
        if 0 <= r < num_rows and 0 <= c < num_cols and grid[r, c] == 0:
            valid_neighbors.append((r, c))

    return valid_neighbors


def reconstruct_path(goal_node: Dict) -> List[Tuple[int, int]]:
    path = []
    current = goal_node

    while current is not None:
        path.append(current["position"])
        current = current["parent"]

    return path[::-1]


def find_path(
    grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]
) -> List[Tuple[int, int]]:

    start_node = create_node(position=start, g=0, h=calculate_heuristic(start, goal))

    # Initialize open and closed sets
    open_list = [start]  # List for storing open nodes
    open_dict = {start: start_node}  # For quick node lookup
    closed_set = set()  # Explored nodes

    while open_list:
        # Get node with the lowest f value manually
        current_pos = min(open_list, key=lambda pos: open_dict[pos]["f"])
        current_node = open_dict[current_pos]
        open_list.remove(current_pos)

        # Check if we've reached the goal
        if current_pos == goal:
            return reconstruct_path(current_node)

        closed_set.add(current_pos)

        # Explore neighbors
        for neighbor_pos in get_valid_neighbors(grid, current_pos):
            # Skip if already explored
            if neighbor_pos in closed_set:
                continue

            # Calculate new path cost
            tentative_g = current_node["g"] + calculate_heuristic(
                current_pos, neighbor_pos
            )

            # Create or update neighbor
            if neighbor_pos not in open_dict:
                neighbor = create_node(
                    position=neighbor_pos,
                    g=tentative_g,
                    h=calculate_heuristic(neighbor_pos, goal),
                    parent=current_node,
                )
                open_list.append(neighbor_pos)
                open_dict[neighbor_pos] = neighbor
            elif tentative_g < open_dict[neighbor_pos]["g"]:
                # Found a better path to the neighbor
                neighbor = open_dict[neighbor_pos]
                neighbor["g"] = tentative_g
                neighbor["f"] = tentative_g + neighbor["h"]
                neighbor["parent"] = current_node

    return []


grid = np.zeros((20, 20))

# obstacles
grid[5:15, 10] = 1  # Vertical wall
grid[5, 5:15] = 1  # Horizontal wall

# Define start and goal positions
start_pos = (2, 2)
goal_pos = (18, 18)

path = find_path(grid, start_pos, goal_pos)
if path:
    print(f"Path found: {path}")
else:
    print("No path found")
