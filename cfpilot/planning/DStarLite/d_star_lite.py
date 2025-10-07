"""
D* Lite grid planning
author: vss2sn (28676655+vss2sn@users.noreply.github.com)
Refactored by: Vahid Danesh to integrate with GridMap

Link to papers:
D* Lite (Link: http://idm-lab.org/bib/abstracts/papers/aaai02b.pdf)
Improved Fast Replanning for Robot Navigation in Unknown Terrain
(Link: http://www.cs.cmu.edu/~maxim/files/dlite_icra02.pdf)

Implements D* Lite algorithm with GridMap integration for efficient
dynamic replanning when obstacles are detected.
"""
import math
import sys
import os
from typing import overload
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from cfpilot.mapping import GridMap

pause_time = 0.001



class Spline2D:

    def __init__(self, x, y, kind="cubic"):
        self.s = self.__calc_s(x, y)
        self.sx = interpolate.interp1d(self.s, x, kind=kind)
        self.sy = interpolate.interp1d(self.s, y, kind=kind)

    def __calc_s(self, x, y):
        self.ds = np.hypot(np.diff(x), np.diff(y))
        s = [0.0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        """
        Calculate positions for array of s values.
        
        Args:
            s: scalar or array-like of path parameter values
            
        Returns:
            x, y: interpolated coordinates (scalars or arrays)
        """
        s = np.atleast_1d(s)
        x = self.sx(s)
        y = self.sy(s)
        return x, y




class Node:
    def __init__(self, x: int = 0, y: int = 0, cost: float = 0.0):
        self.x = x
        self.y = y
        self.cost = cost


def add_coordinates(node1: Node, node2: Node):
    new_node = Node()
    new_node.x = node1.x + node2.x
    new_node.y = node1.y + node2.y
    new_node.cost = node1.cost + node2.cost
    return new_node


def compare_coordinates(node1: Node, node2: Node):
    return node1.x == node2.x and node1.y == node2.y


class DStarLite:
    """
    D* Lite path planner with GridMap integration.
    
    Supports efficient dynamic replanning when the map is updated with new obstacles.
    """

    # Motion primitives: (dx, dy, cost)
    # Adjust heuristic function (h) if you change these motions
    motions = [
        Node(1, 0, 1),
        Node(0, 1, 1),
        Node(-1, 0, 1),
        Node(0, -1, 1),
        Node(1, 1, math.sqrt(2)),
        Node(1, -1, math.sqrt(2)),
        Node(-1, 1, math.sqrt(2)),
        Node(-1, -1, math.sqrt(2))
    ]

    def __init__(self, grid_map: GridMap, 
                 obstacle_threshold: float = 0.5, 
                 show_animation: bool = False):
        """
        Initialize D* Lite planner with a GridMap.
        
        :param grid_map: GridMap instance containing the environment
        :param obstacle_threshold: threshold value to consider a cell as obstacle (default 0.5)
        :param show_animation: whether to show animation during planning
        """
        # Store reference to the grid map
        self.grid_map = grid_map
        self.obstacle_threshold = obstacle_threshold
        
        # Use grid map dimensions
        self.x_max = grid_map.width
        self.y_max = grid_map.height
        
        # Current start and goal positions (in grid indices)
        self.start = Node(0, 0)
        self.goal = Node(0, 0)
        
        # D* Lite data structures
        self.U = []  # Priority queue (list of tuples: (node, key))
        self.km = 0.0  # Key modifier for incremental search
        self.rhs = self.create_grid(math.inf)  # One-step lookahead values
        self.g = self.create_grid(math.inf)     # Cost-to-goal estimates
        
        # Track previous map state for change detection
        self.previous_map = grid_map.data.copy()
        
        # Visualization
        self.show_animation = show_animation
        if self.show_animation:
            self.new_obstacles_x = []
            self.new_obstacles_y = []
        
        self.initialized = False

    def create_grid(self, val: float):
        """Create a grid initialized with a value."""
        return np.full((self.x_max, self.y_max), val)

    def is_obstacle(self, node: Node):
        """
        Check if a node is an obstacle using the GridMap.
        
        :param node: Node with grid indices
        :return: True if obstacle, False otherwise
        """
        if not self.is_valid(node):
            return True
        
        val = self.grid_map.get_value_from_xy_index(node.x, node.y)
        if val is None:
            return True
        
        return val >= self.obstacle_threshold

    def world_to_grid(self, x_world: float, y_world: float):
        """
        Convert world coordinates to grid indices.
        
        :param x_world: x position in world coordinates [m]
        :param y_world: y position in world coordinates [m]
        :return: tuple (x_ind, y_ind) or (None, None) if out of bounds
        """
        return self.grid_map.get_xy_index_from_xy_pos(x_world, y_world)
    
    def grid_to_world(self, x_ind: int, y_ind: int):
        """
        Convert grid indices to world coordinates.
        
        :param x_ind: x grid index
        :param y_ind: y grid index
        :return: tuple (x_pos, y_pos) or (None, None) if out of bounds
        """
        return self.grid_map.get_xy_poss_from_xy_indexes(x_ind, y_ind)

    def c(self, node1: Node, node2: Node):
        """Cost of moving from node1 to node2"""
        if self.is_obstacle(node2):
            return math.inf
        
        # Find matching motion primitive
        delta = Node(node1.x - node2.x, node1.y - node2.y)
        for motion in self.motions:
            if compare_coordinates(motion, delta):
                return motion.cost
        
        return math.inf

    def h(self, s: Node):
        # Cannot use the 2nd euclidean norm as this might sometimes generate
        # heuristics that overestimate the cost, making them inadmissible,
        # due to rounding errors etc (when combined with calculate_key)
        # To be admissible heuristic should
        # never overestimate the cost of a move
        # hence not using the line below
        # return math.hypot(self.start.x - s.x, self.start.y - s.y)

        # Below is the same as 1; modify if you modify the cost of each move in
        # motion
        # return max(abs(self.start.x - s.x), abs(self.start.y - s.y))
        return 1

    def calculate_key(self, s: Node):
        return (min(self.g[s.x][s.y], self.rhs[s.x][s.y]) + self.h(s)
                + self.km, min(self.g[s.x][s.y], self.rhs[s.x][s.y]))

    def is_valid(self, node: Node):
        """Check if node is within grid bounds"""
        return 0 <= node.x < self.x_max and 0 <= node.y < self.y_max

    def get_neighbours(self, u: Node):
        """Get valid neighbors of node u"""
        return [add_coordinates(u, motion) for motion in self.motions
                if self.is_valid(add_coordinates(u, motion))]

    def initialize(self, start_x: float, start_y: float, goal_x: float, goal_y: float):
        """
        Initialize D* Lite with start and goal in world coordinates.
        
        :param start_x: start x position in world coordinates [m]
        :param start_y: start y position in world coordinates [m]
        :param goal_x: goal x position in world coordinates [m]
        :param goal_y: goal y position in world coordinates [m]
        """
        # Convert world coordinates to grid indices
        start_x_ind, start_y_ind = self.world_to_grid(start_x, start_y)
        goal_x_ind, goal_y_ind = self.world_to_grid(goal_x, goal_y)
        
        if start_x_ind is None or goal_x_ind is None:
            raise ValueError("Start or goal position is out of grid bounds")
        
        self.start.x = start_x_ind
        self.start.y = start_y_ind
        self.goal.x = goal_x_ind
        self.goal.y = goal_y_ind
        
        if not self.initialized:
            self.initialized = True
            print('Initializing D* Lite')
            self.U = []
            self.km = 0.0
            self.rhs = self.create_grid(math.inf)
            self.g = self.create_grid(math.inf)
            self.rhs[self.goal.x][self.goal.y] = 0
            self.U.append((self.goal, self.calculate_key(self.goal)))
            
            # Store initial map state
            self.previous_map = self.grid_map.data.copy()

    def update_vertex(self, u: Node):
        """Update vertex u in the priority queue"""
        if not compare_coordinates(u, self.goal):
            self.rhs[u.x][u.y] = min([self.c(u, sprime) + self.g[sprime.x][sprime.y]
                                      for sprime in self.get_neighbours(u)])
        
        # Remove u from priority queue if present
        self.U = [(node, key) for node, key in self.U if not compare_coordinates(node, u)]
        
        # Add u to queue if inconsistent
        if self.g[u.x][u.y] != self.rhs[u.x][u.y]:
            self.U.append((u, self.calculate_key(u)))
            self.U.sort(key=lambda x: x[1])

    def compare_keys(self, key_pair1: tuple[float, float],
                     key_pair2: tuple[float, float]):
        return key_pair1[0] < key_pair2[0] or \
               (key_pair1[0] == key_pair2[0] and key_pair1[1] < key_pair2[1])

    def compute_shortest_path(self):
        """Main D* Lite algorithm - compute shortest path from start to goal"""
        self.U.sort(key=lambda x: x[1])
        
        while len(self.U) > 0 and \
              (self.compare_keys(self.U[0][1], self.calculate_key(self.start)) or
               self.rhs[self.start.x][self.start.y] != self.g[self.start.x][self.start.y]):
            
            kold = self.U[0][1]
            u = self.U[0][0]
            self.U.pop(0)
            
            if self.compare_keys(kold, self.calculate_key(u)):
                self.U.append((u, self.calculate_key(u)))
                self.U.sort(key=lambda x: x[1])
            elif self.g[u.x][u.y] > self.rhs[u.x][u.y]:
                self.g[u.x][u.y] = self.rhs[u.x][u.y]
                for s in self.get_neighbours(u):
                    self.update_vertex(s)
            else:
                self.g[u.x][u.y] = math.inf
                for s in self.get_neighbours(u) + [u]:
                    self.update_vertex(s)

    def detect_changes(self):
        """
        Detect changes in the grid map by comparing with previous state.
        
        :return: list of Nodes that have changed to obstacles
        """
        changed_vertices = []
        
        # Compare current map with previous map using numpy
        changes = (self.grid_map.data >= self.obstacle_threshold) & \
                  (self.previous_map < self.obstacle_threshold)
        
        # Get indices of changed cells
        changed_x, changed_y = np.where(changes)
        
        # Convert to Node list
        for x, y in zip(changed_x, changed_y):
            node = Node(x, y)
            # Don't mark start or goal as obstacles
            if compare_coordinates(node, self.start) or \
               compare_coordinates(node, self.goal):
                continue
            changed_vertices.append(node)
            
            if self.show_animation:
                x_world, y_world = self.grid_to_world(x, y)
                self.new_obstacles_x.append(x_world)
                self.new_obstacles_y.append(y_world)
                plt.plot(self.new_obstacles_x, self.new_obstacles_y, ".k")
                plt.pause(pause_time)
        
        # Update previous map state
        if len(changed_vertices) > 0:
            self.previous_map = self.grid_map.data.copy()
        
        return changed_vertices
    
    def update_map(self, changed_cells: list = None):
        """
        Update the planner when the map changes.
        
        :param changed_cells: optional list of (x_ind, y_ind) tuples that changed
                             If None, detects changes automatically
        :return: True if changes were detected and replanning occurred
        """
        if changed_cells is not None:
            # Use provided change list
            changed_vertices = []
            for x, y in changed_cells:
                node = Node(x, y)
                if compare_coordinates(node, self.start) or \
                   compare_coordinates(node, self.goal):
                    continue
                changed_vertices.append(node)
                
                if self.show_animation:
                    x_world, y_world = self.grid_to_world(x, y)
                    self.new_obstacles_x.append(x_world)
                    self.new_obstacles_y.append(y_world)
                    plt.plot(self.new_obstacles_x, self.new_obstacles_y, ".k")
                    plt.pause(pause_time)
            
            self.previous_map = self.grid_map.data.copy()
        else:
            # Auto-detect changes
            changed_vertices = self.detect_changes()
        
        if len(changed_vertices) == 0:
            return False
        
        # Update D* Lite structures
        last = Node(self.start.x, self.start.y)
        self.km += self.h(last)
        
        for u in changed_vertices:
            if compare_coordinates(u, self.start):
                continue
            self.rhs[u.x][u.y] = math.inf
            self.g[u.x][u.y] = math.inf
            self.update_vertex(u)
        
        # Recompute shortest path
        self.compute_shortest_path()
        
        print(f"Replanned after detecting {len(changed_vertices)} new obstacles")
        return True

    def compute_current_path(self):
        """Compute path from start to goal based on current g values"""
        path = []
        current_point = Node(self.start.x, self.start.y)
        
        while not compare_coordinates(current_point, self.goal):
            path.append(current_point)
            current_point = min(self.get_neighbours(current_point),
                                key=lambda sprime: self.c(current_point, sprime) + 
                                                  self.g[sprime.x][sprime.y])
        path.append(self.goal)
        return path
    
    def path_to_world_coords(self, node_path: list):
        """
        Convert a path of Nodes (grid indices) to world coordinates.
        
        :param node_path: list of Nodes with grid indices
        :return: tuple of (pathx, pathy) arrays in world coordinates
        """
        pathx = []
        pathy = []
        for node in node_path:
            x_world, y_world = self.grid_to_world(node.x, node.y)
            pathx.append(x_world)
            pathy.append(y_world)
        return np.array(pathx), np.array(pathy)

    def compare_paths(self, path1: list, path2: list):
        """Check if two paths are identical"""
        if len(path1) != len(path2):
            return False
        return all(compare_coordinates(n1, n2) for n1, n2 in zip(path1, path2))

    def plot_path(self, path, color: str = 'b', alpha: float = 0.7):
        """
        Display a path on the plot.
        
        Accepts multiple input formats for flexibility:
        - List of Nodes (grid indices) - will be converted to world coords
        - Tuple of (pathx, pathy) arrays/lists in world coordinates
        
        :param path: Either a list of Nodes OR tuple of (pathx, pathy) in world coords
        :param color: matplotlib color string (default 'b' for blue)
        :param alpha: transparency from 0 (transparent) to 1 (opaque)
        :return: matplotlib plot handle
        """
        # Check if path is a tuple of (pathx, pathy) - world coordinates
        if isinstance(path, tuple) and len(path) == 2:
            pathx, pathy = path
        # Otherwise assume it's a list of Nodes - convert to world coords
        elif isinstance(path, list) and len(path) > 0 and isinstance(path[0], Node):
            pathx, pathy = self.path_to_world_coords(path)
        else:
            raise ValueError(
                "path must be either a list of Nodes or tuple of (pathx, pathy) arrays"
            )
        
        drawing = plt.plot(pathx, pathy, color, alpha=alpha)
        plt.pause(pause_time)
        return drawing

    def simplify_path(self, pathx, pathy):
        """
        Remove unnecessary waypoints - keep only points where path must turn.
        Uses GridMap to check line-of-sight between waypoints.
        
        :param pathx: x coordinates of path waypoints (world coords)
        :param pathy: y coordinates of path waypoints (world coords)
        :return: tuple of (simplified_x, simplified_y) arrays
        """
        pathx = np.asarray(pathx)
        pathy = np.asarray(pathy)
        
        if len(pathx) <= 2:
            return pathx, pathy
        
        def line_clear(i, j):
            """Check if straight line between two points is obstacle-free"""
            # Extract scalar values from numpy arrays
            x_i = pathx[i].item() if hasattr(pathx[i], 'item') else pathx[i]
            y_i = pathy[i].item() if hasattr(pathy[i], 'item') else pathy[i]
            x_j = pathx[j].item() if hasattr(pathx[j], 'item') else pathx[j]
            y_j = pathy[j].item() if hasattr(pathy[j], 'item') else pathy[j]
            
            dist = np.hypot(x_j - x_i, y_j - y_i)
            n = int(dist / self.grid_map.resolution) * 2 + 1
            xs = np.linspace(x_i, x_j, n)
            ys = np.linspace(y_i, y_j, n)
            
            x_inds, y_inds = self.world_to_grid(xs, ys)
            if x_inds is None or y_inds is None:
                return False
            
            vals = self.grid_map.get_value_from_xy_index(x_inds, y_inds)
            if vals is None:
                return False
            
            return np.all(vals < self.obstacle_threshold)
        
        result = [0]  # Keep start
        i = 0
        
        while i < len(pathx) - 1:
            # Skip ahead as far as possible
            j = len(pathx) - 1
            while j > i + 1 and not line_clear(i, j):
                j -= 1
            
            # If we can't skip ahead, just move to next point
            if j == i:
                j = i + 1
            
            result.append(j)
            i = j
        
        result = np.array(result)
        return pathx[result], pathy[result]
    
    def smooth_path(self, pathx, pathy, ds=0.1):
        """Smooth a path using 2D spline interpolation
        
        :param pathx: x coordinates of path waypoints
        :param pathy: y coordinates of path waypoints
        :param ds: spacing between interpolated points along the path [m]
        :return: tuple of (smooth_x, smooth_y) arrays with interpolated coordinates
        """
        pathx = np.asarray(pathx)
        pathy = np.asarray(pathy)
        
        if len(pathx) < 2:
            return pathx, pathy
        

        # Create spline with appropriate interpolation type
        kind = "linear" if len(pathx) < 4 else "cubic"
        spline = Spline2D(pathx, pathy, kind=kind)
        
        # Generate smooth path with specified spacing
        s_points = np.arange(0, spline.s[-1], ds)
        s_points = np.append(s_points, spline.s[-1])
        
        smooth_x, smooth_y = spline.calc_position(s_points)
        return smooth_x, smooth_y


    def run(self, 
            start:tuple, 
            goal:tuple, 
            simplify:bool=True,
            smooth:bool=False,
            obstacle_callback=None):
        """
        Plan and execute path from start to goal with dynamic replanning.
        
        :param start_x: start x position in world coordinates [m]
        :param start_y: start y position in world coordinates [m]
        :param goal_x: goal x position in world coordinates [m]
        :param goal_y: goal y position in world coordinates [m]
        :param obstacle_callback: optional function that returns list of new
                                 obstacle (x_ind, y_ind) tuples at each step
        :return: tuple (success, pathx, pathy) with path in world coordinates
        """
        pathx = []
        pathy = []
        start_x, start_y = start
        goal_x, goal_y = goal
        # Initialize planning
        self.initialize(start_x, start_y, goal_x, goal_y)
        last = Node(self.start.x, self.start.y)
        self.compute_shortest_path()
        
        # Add start to path
        x_world, y_world = self.grid_to_world(self.start.x, self.start.y)
        pathx.append(x_world)
        pathy.append(y_world)

        if self.show_animation:
            current_path = self.compute_current_path()
            previous_path = current_path.copy()
            previous_path_image = self.plot_path(previous_path, ".c", alpha=0.3)
            current_path_image = self.plot_path(current_path, ".c")

        # Execute path with replanning
        while not compare_coordinates(self.goal, self.start):
            if self.g[self.start.x][self.start.y] == math.inf:
                print("No path possible")
                return False, pathx, pathy
            
            # Get next move
            self.start = min(self.get_neighbours(self.start),
                           key=lambda sprime: self.c(self.start, sprime) + 
                                             self.g[sprime.x][sprime.y])
            
            x_world, y_world = self.grid_to_world(self.start.x, self.start.y)
            pathx.append(x_world)
            pathy.append(y_world)
            
            if self.show_animation:
                current_path.pop(0)
                plt.plot(pathx, pathy, "-r")
                plt.pause(pause_time)
            
            # Check for obstacles
            if obstacle_callback is not None:
                new_obstacles = obstacle_callback()
                if len(new_obstacles) > 0:
                    self.update_map(new_obstacles)
                    
                    if self.show_animation:
                        new_path = self.compute_current_path()
                        if not self.compare_paths(current_path, new_path):
                            current_path_image[0].remove()
                            previous_path_image[0].remove()
                            previous_path = current_path.copy()
                            current_path = new_path.copy()
                            previous_path_image = self.plot_path(previous_path, ".c", alpha=0.3)
                            current_path_image = self.plot_path(current_path, ".c")
            else:
                # Auto-detect changes from GridMap
                changed_vertices = self.detect_changes()
                if len(changed_vertices) != 0:
                    self.km += self.h(last)
                    last = Node(self.start.x, self.start.y)
                    for u in changed_vertices:
                        if compare_coordinates(u, self.start):
                            continue
                        self.rhs[u.x][u.y] = math.inf
                        self.g[u.x][u.y] = math.inf
                        self.update_vertex(u)
                    self.compute_shortest_path()

                    if self.show_animation:
                        new_path = self.compute_current_path()
                        if not self.compare_paths(current_path, new_path):
                            current_path_image[0].remove()
                            previous_path_image[0].remove()
                            previous_path = current_path.copy()
                            current_path = new_path.copy()
                            previous_path_image = self.plot_path(previous_path, ".c", alpha=0.3)
                            current_path_image = self.plot_path(current_path, ".c")
                            plt.pause(pause_time)
        
        print("Path found")
        pathx = np.ravel(pathx)
        pathy = np.ravel(pathy)
        if simplify:
            pathx, pathy = self.simplify_path(pathx, pathy)
        if smooth:
            pathx, pathy = self.smooth_path(pathx, pathy)

        return True, pathx, pathy


def main():
    show_animation = True
    """
    Demo showing D* Lite path planning with GridMap integration.
    Demonstrates dynamic replanning when new obstacles are detected.
    """
    print("D* Lite Path Planning Demo with GridMap")
    print("=" * 50)

    # Start and goal positions (world coordinates)
    sx, sy = 10.0, 10.0  # [m]
    gx, gy = 50.0, 50.0  # [m]
    
    # Create grid map
    # Map spans from -10 to 60 in both x and y
    map_width = 70   # cells
    map_height = 70  # cells
    resolution = 1.0  # m per cell
    center_x = 25.0   # m
    center_y = 25.0   # m
    
    grid_map = GridMap(map_width, map_height, resolution, 
                      center_x, center_y, init_val=0.0)
    
    print(f"Created GridMap: {map_width}x{map_height} cells, resolution={resolution}m")
    
    # Add boundary obstacles
    grid_map.occupy_boundaries(boundary_width=1, val=1.0)
    
    # Add some wall obstacles
    # Vertical wall
    for i in range(-10, 40):
        grid_map.set_value_from_xy_pos(20.0, float(i), 1.0)
    
    # Another wall
    for i in range(0, 40):
        grid_map.set_value_from_xy_pos(40.0, float(60 - i), 1.0)
    
    print("Added initial obstacles to map")
    
    # Visualization setup
    if show_animation:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot the grid map
        grid_map.plot_grid_map(ax=ax, use_world_coords=True)
        
        # Plot start and goal
        plt.plot(sx, sy, "og", markersize=10, label="Start")
        plt.plot(gx, gy, "xb", markersize=10, label="Goal")
        plt.grid(True)
        plt.axis("equal")
        
        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', 
                      markersize=10, label='Start'),
            plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='b', 
                      markersize=10, label='Goal'),
            plt.Line2D([0], [0], color='r', linewidth=2, label='Path taken'),
            plt.Line2D([0], [0], marker='.', color='c', linestyle='', 
                      markersize=8, label='Current computed path'),
            plt.Line2D([0], [0], marker='.', color='c', linestyle='', 
                      markersize=8, alpha=0.3, label='Previous computed path'),
            plt.Line2D([0], [0], marker='.', color='k', linestyle='', 
                      markersize=8, label='New obstacles')
        ]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
                  loc='upper left', fontsize='small')
        
        plt.tight_layout()
        plt.pause(pause_time)
    
    # Create D* Lite planner
    dstarlite = DStarLite(grid_map, obstacle_threshold=0.5, show_animation=show_animation)
    print("Initialized D* Lite planner")
    
    # Define a callback that simulates dynamic obstacle detection
    # This adds obstacles at specific steps to demonstrate replanning
    obstacle_schedule = [
        (10, [(i, 40) for i in range(15, 26)]),
        (20, [(i, 40) for i in range(35, 46)]),
        (40, [(50, i) for i in range(20, 41)])
    ]
    step_counter = [0]  # Use list to allow modification in closure
    
    def obstacle_callback():
        """Simulate detection of new obstacles during execution"""
        step_counter[0] += 1
        
        for step_num, obstacles in obstacle_schedule:
            if step_counter[0] == step_num:
                print(f"\n⚠️  Detected {len(obstacles)} new obstacles at step {step_num}!")
                
                # Add obstacles to the grid map
                for x_ind, y_ind in obstacles:
                    if grid_map.valid(x_ind, y_ind):
                        grid_map.set_value_from_xy_index(x_ind, y_ind, 1.0)
                
                return obstacles
        
        return []
    
    # Run the planner
    print(f"\nPlanning path from ({sx}, {sy}) to ({gx}, {gy})...")
    success, pathx, pathy = dstarlite.run(start=(sx, sy), goal=(gx, gy),
                                          obstacle_callback=obstacle_callback)
    
    if success:
        print(f"\n✓ Path found with {len(pathx)} waypoints")
        
        # Optionally simplify and smooth the path
        print("\nPost-processing path...")
        simplified_x, simplified_y = dstarlite.simplify_path(pathx, pathy)
        print(f"  Simplified to {len(simplified_x)} waypoints")
        
        smooth_x, smooth_y = dstarlite.smooth_path(simplified_x, simplified_y, ds=0.5)
        print(f"  Smoothed to {len(smooth_x)} points")
        
        if show_animation:
            plt.plot(simplified_x, simplified_y, "o-", color='orange', 
                    linewidth=2, markersize=6, label="Simplified path", alpha=0.7)
            plt.plot(smooth_x, smooth_y, "-", color='green', 
                    linewidth=3, label="Smoothed path")
            plt.legend(handles=legend_elements + [
                plt.Line2D([0], [0], color='orange', linewidth=2, 
                          marker='o', label='Simplified path'),
                plt.Line2D([0], [0], color='green', linewidth=3, 
                          label='Smoothed path')
            ], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            plt.show()
    else:
        print("\n✗ No path found")
        if show_animation:
            plt.show()
    
    print("\nDemo complete!")


def simple_demo():
    """
    Simplified demo without dynamic obstacles.
    """
    print("Simple D* Lite Demo")
    print("=" * 50)
    
    # Create a simple grid map
    grid_map = GridMap(50, 50, 1.0, 25.0, 25.0, init_val=0.0)
    
    # Add boundaries
    grid_map.occupy_boundaries(boundary_width=1, val=1.0)
    
    # Add a circular obstacle
    grid_map.add_obstacle(center_x=25.0, center_y=25.0, radius=5.0, val=1.0)
    
    # Add some rectangular obstacles
    for x in range(10, 20):
        for y in range(10, 30):
            grid_map.set_value_from_xy_pos(float(x), float(y), 1.0)
    
    # Create planner and find path
    planner = DStarLite(grid_map, obstacle_threshold=0.5, show_animation=False)
    success, pathx, pathy = planner.run(start=(5.0, 5.0), goal=(45.0, 45.0))
    
    if success:
        print(f"✓ Path found with {len(pathx)} waypoints")
        
        # Visualize result
        fig, ax = plt.subplots(figsize=(8, 8))
        grid_map.plot_grid_map(ax=ax, use_world_coords=True)
        plt.plot(pathx, pathy, 'r-', linewidth=2, label='Path')
        plt.plot(pathx[0], pathy[0], 'go', markersize=10, label='Start')
        plt.plot(pathx[-1], pathy[-1], 'bx', markersize=10, label='Goal')
        plt.legend()
        plt.title('D* Lite Path Planning Result')
        plt.show()
    else:
        print("✗ No path found")


if __name__ == "__main__":
    # Run the main demo with dynamic replanning
    main()
    
    # Uncomment to run the simple demo instead
    # simple_demo()
