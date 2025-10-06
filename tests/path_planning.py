"""
Path Planning Module

Provides functions for path planning using D* Lite algorithm, path simplification,
and smoothing with cubic splines for Crazyflie navigation.
"""
import numpy as np
import matplotlib.pyplot as plt
import copy
from typing import Tuple, List, Optional

from cfpilot.mapping import GridMap
from cfpilot.planning.DStarLite.d_star_lite import DStarLite, Node
from cfpilot.planning.CubicSpline.cubic_spline_planner import CubicSpline2D, calc_spline_course
from cfpilot.planning.CubicSpline.spline_continuity import Spline2D


def create_test_environment(resolution: float = 0.05, 
                            width: float = 4.6, 
                            height: float = 2.6) -> GridMap:
    """
    Create a test grid map environment with obstacles
    
    Args:
        resolution: Grid resolution in meters
        width: Environment width in meters
        height: Environment height in meters
        
    Returns:
        GridMap object with obstacles
    """
    grid_map = GridMap(
        width=int(width / resolution),
        height=int(height / resolution),
        resolution=resolution,
        center_x=width / 2,
        center_y=height / 2
    )
    
    # Add circular obstacles
    grid_map.add_obstacle(center_x=1.0, center_y=2.2, radius=0.3)
    grid_map.add_obstacle(center_x=1.8, center_y=1.4, radius=0.3)
    grid_map.add_obstacle(center_x=3.0, center_y=0.7, radius=0.3)
    grid_map.add_obstacle(center_x=3.6, center_y=1.6, radius=0.3)
    
    # Occupy boundaries
    grid_map.occupy_boundaries(boundary_width=2)
    
    # Expand obstacles for safety
    grid_map.expand_grid(occupied_val=0.5)
    
    return grid_map


def plan_path(grid_map: GridMap,
              start_pos: Tuple[float, float],
              goal_pos: Tuple[float, float],
              obstacle_threshold: float = 0.5) -> Tuple[bool, List[int], List[int]]:
    """
    Plan a path using D* Lite algorithm
    
    Args:
        grid_map: GridMap object containing environment
        start_pos: Start position (x, y) in meters
        goal_pos: Goal position (x, y) in meters
        obstacle_threshold: Threshold for considering cells as obstacles
        
    Returns:
        Tuple of (path_found, pathx, pathy) where pathx and pathy are grid indices
    """
    # Get obstacle locations
    ox, oy = np.where(grid_map.data > obstacle_threshold)
    
    # Initialize planner
    planner = DStarLite(ox=ox, oy=oy)
    
    # Convert positions to grid indices
    start_index = grid_map.get_xy_index_from_xy_pos(x_pos=start_pos[0], y_pos=start_pos[1])
    goal_index = grid_map.get_xy_index_from_xy_pos(x_pos=goal_pos[0], y_pos=goal_pos[1])
    
    # Create nodes
    start = Node(start_index[0], start_index[1])
    goal = Node(goal_index[0], goal_index[1])
    
    # Plan path
    path_found, pathx, pathy = planner.main(
        start=start,
        goal=goal,
        spoofed_ox=[],
        spoofed_oy=[]
    )
    
    return path_found, pathx, pathy


def simplify_path(pathx: List[int], 
                 pathy: List[int], 
                 grid_map: GridMap,
                 obstacle_threshold: float = 0.5) -> Tuple[List[int], List[int]]:
    """
    Simplify path by removing unnecessary waypoints using line-of-sight checks
    
    Args:
        pathx, pathy: Original path as grid indices
        grid_map: GridMap containing obstacle information
        obstacle_threshold: Threshold for considering cells as obstacles
        
    Returns:
        Tuple of (simplified_x, simplified_y) as grid indices
    """
    if len(pathx) <= 2:
        return pathx, pathy
    
    def line_clear(i: int, j: int) -> bool:
        """Check if straight line between two points is obstacle-free"""
        n = int(np.hypot(pathx[j] - pathx[i], pathy[j] - pathy[i])) * 2 + 1
        xs = np.linspace(pathx[i], pathx[j], n).astype(int)
        ys = np.linspace(pathy[i], pathy[j], n).astype(int)
        return np.all(grid_map.data[xs, ys] <= obstacle_threshold)
    
    result = [0]  # Keep start
    i = 0
    
    while i < len(pathx) - 1:
        # Find farthest visible point
        j = len(pathx) - 1
        while j > i and not line_clear(i, j):
            j -= 1
        result.append(j)
        i = j
    
    return [pathx[k] for k in result], [pathy[k] for k in result]


def smooth_path(pos_x: List[float],
               pos_y: List[float],
               ds: float = 0.05,
               kind: str = 'cubic') -> Tuple[np.ndarray, np.ndarray]:
    """
    Smooth path using spline interpolation
    
    Args:
        pos_x, pos_y: Waypoint positions in meters
        ds: Distance between interpolated points in meters
        kind: Type of spline ('linear', 'quadratic', 'cubic')
        
    Returns:
        Tuple of (rx, ry) - interpolated path coordinates
    """
    sp = Spline2D(pos_x, pos_y, kind=kind)
    s = np.arange(0, sp.s[-1], ds)
    rx, ry = sp.calc_position(s)
    return rx, ry


def get_waypoints(grid_map: GridMap,
                 start_pos: Tuple[float, float],
                 goal_pos: Tuple[float, float],
                 simplify: bool = True,
                 smooth: bool = True,
                 ds: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete path planning pipeline: plan, simplify, and smooth
    
    Args:
        grid_map: GridMap object containing environment
        start_pos: Start position (x, y) in meters
        goal_pos: Goal position (x, y) in meters
        simplify: Whether to simplify the path
        smooth: Whether to smooth the path with splines
        ds: Distance between interpolated points if smoothing
        
    Returns:
        Tuple of (waypoints_x, waypoints_y) in meters
    """
    # Plan path
    path_found, pathx, pathy = plan_path(grid_map, start_pos, goal_pos)
    
    if not path_found:
        raise ValueError("No path found between start and goal")
    
    # Simplify if requested
    if simplify:
        pathx, pathy = simplify_path(pathx, pathy, grid_map)
    
    # Convert to world coordinates
    pos_x, pos_y = grid_map.get_xy_poss_from_xy_indexes(pathx, pathy)
    
    # Smooth if requested
    if smooth and len(pos_x) > 2:
        pos_x, pos_y = smooth_path(pos_x, pos_y, ds=ds, kind='cubic')
    
    return np.array(pos_x), np.array(pos_y)


def visualize_path(grid_map: GridMap,
                  start_pos: Tuple[float, float],
                  goal_pos: Tuple[float, float],
                  show_original: bool = True,
                  show_simplified: bool = True,
                  show_smooth: bool = True) -> None:
    """
    Visualize the complete path planning pipeline
    
    Args:
        grid_map: GridMap object
        start_pos: Start position (x, y)
        goal_pos: Goal position (x, y)
        show_original: Show original D* Lite path
        show_simplified: Show simplified path
        show_smooth: Show smoothed path
    """
    # Plan original path
    path_found, pathx, pathy = plan_path(grid_map, start_pos, goal_pos)
    
    if not path_found:
        print("❌ No path found!")
        return
    
    pos_x, pos_y = grid_map.get_xy_poss_from_xy_indexes(pathx, pathy)
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot grid map
    grid_map.plot_grid_map(use_world_coords=True)
    
    # Plot original path
    if show_original:
        plt.plot(pos_x, pos_y, 'b.', alpha=0.3, markersize=4, label=f'D* Lite path ({len(pathx)} pts)')
    
    # Plot simplified path
    if show_simplified or show_smooth:
        pathx_simp, pathy_simp = simplify_path(pathx, pathy, grid_map)
        pos_x_simp, pos_y_simp = grid_map.get_xy_poss_from_xy_indexes(pathx_simp, pathy_simp)
        
        if show_simplified:
            plt.plot(pos_x_simp, pos_y_simp, 'bx-', markersize=5, linewidth=1, 
                    label=f'Simplified ({len(pathx_simp)} waypoints)')
    
    # Plot smooth path
    if show_smooth and len(pos_x_simp) > 2:
        rx, ry = smooth_path(pos_x_simp, pos_y_simp, ds=0.05, kind='cubic')
        plt.plot(rx, ry, 'g-', linewidth=1, label=f'Smooth path ({len(rx)} pts)')
    
    # Highlight start and goal
    plt.plot(start_pos[0], start_pos[1], 'gx', markersize=7, label='Start', zorder=10)
    plt.plot(goal_pos[0], goal_pos[1], 'mx', markersize=7, label='Goal', zorder=10)
    
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlabel('x[m]')
    plt.ylabel('y[m]')
    plt.title('Path Planning: D* Lite → Simplification → Smoothing')
    plt.legend()
    plt.show()


def compare_spline_types(grid_map: GridMap,
                        pathx: List[int],
                        pathy: List[int]) -> None:
    """
    Compare different spline interpolation types
    
    Args:
        grid_map: GridMap object
        pathx, pathy: Path as grid indices
    """
    # Simplify path first
    pathx_simp, pathy_simp = simplify_path(pathx, pathy, grid_map)
    pos_x, pos_y = grid_map.get_xy_poss_from_xy_indexes(pathx_simp, pathy_simp)
    
    plt.figure(figsize=(12, 8))
    grid_map.plot_grid_map(use_world_coords=True)
    
    plt.plot(pos_x, pos_y, "xb", markersize=10, label="Waypoints", zorder=5)
    
    ds = 0.05
    for (kind, label, color) in [
        ("linear", "C0 (Linear spline)", "orange"),
        ("quadratic", "C0 & C1 (Quadratic spline)", "purple"),
        ("cubic", "C0 & C1 & C2 (Cubic spline)", "green")
    ]:
        rx, ry = smooth_path(pos_x, pos_y, ds=ds, kind=kind)
        plt.plot(rx, ry, "-", linewidth=2, label=label, color=color)
    
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.title("Spline Interpolation Comparison")
    plt.legend()
    plt.show()


# Example usage
if __name__ == "__main__":
    print("🗺️  Creating test environment...")
    grid_map = create_test_environment()
    
    print("📍 Planning path...")
    start_pos = (0.7, 1.3)
    goal_pos = (3.9, 0.7)
    
    # Get waypoints (complete pipeline)
    waypoints_x, waypoints_y = get_waypoints(
        grid_map,
        start_pos,
        goal_pos,
        simplify=True,
        smooth=True,
        ds=0.05
    )
    
    print(f"✅ Path planned with {len(waypoints_x)} waypoints")
    print(f"   Start: {start_pos}")
    print(f"   Goal: {goal_pos}")
    
    # Visualize
    print("📊 Visualizing path...")
    visualize_path(grid_map, start_pos, goal_pos)
    
    # Print first few waypoints
    print("\n📋 First 5 waypoints:")
    for i in range(min(5, len(waypoints_x))):
        print(f"   {i+1}. ({waypoints_x[i]:.3f}, {waypoints_y[i]:.3f})")
