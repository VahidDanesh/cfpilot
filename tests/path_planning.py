"""
Path Planning Module

Provides functions for path planning using D* Lite algorithm, path simplification,
and smoothing with cubic splines for Crazyflie navigation.
"""
import numpy as np
import matplotlib.pyplot as plt
import copy
from typing import Tuple, List, Optional
from cflib.utils.multiranger import Multiranger

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
    grid_map.add_obstacle(center_x=1.5, center_y=1.6, radius=0.3)
    grid_map.add_obstacle(center_x=2.5, center_y=1.5, radius=0.3)
    grid_map.add_obstacle(center_x=3.2, center_y=0.7, radius=0.3)
    grid_map.add_obstacle(center_x=3.2, center_y=2.3, radius=0.3)
    
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


# ============================================================================
# Real-time Obstacle Avoidance System
# ============================================================================

def calculate_avoidance_vector(multiranger: Multiranger,
                               current_yaw: float = 0.0,
                               danger_threshold: float = 0.5,
                               caution_threshold: float = 1.0) -> Tuple[float, float, bool]:
    """
    Calculate avoidance vector using potential field method from multiranger data
    
    Args:
        multiranger: Multiranger object with sensor readings (in mm)
        current_yaw: Current drone yaw in radians
        danger_threshold: Distance threshold for strong repulsion (meters)
        caution_threshold: Distance threshold for weak repulsion (meters)
        
    Returns:
        Tuple of (avoidance_x, avoidance_y, emergency_stop)
        - avoidance vector in world frame (m)
        - emergency_stop flag if obstacle too close
    """
    # Sensor readings in mm, convert to meters
    sensors = {
        'front': multiranger.front / 1000.0 if multiranger.front is not None else None,
        'back': multiranger.back / 1000.0 if multiranger.back is not None else None,
        'left': multiranger.left / 1000.0 if multiranger.left is not None else None,
        'right': multiranger.right / 1000.0 if multiranger.right is not None else None
    }
    
    # Sensor angles in body frame (radians)
    sensor_angles_body = {
        'front': 0.0,
        'right': -np.pi/2,
        'back': np.pi,
        'left': np.pi/2
    }
    
    # Initialize avoidance vector
    avoid_x, avoid_y = 0.0, 0.0
    emergency_stop = False
    
    for direction, distance in sensors.items():
        # Skip invalid readings
        if distance is None or distance > 4.0:
            continue
        
        # Check for emergency stop (very close obstacle)
        if distance < 0.1:
            emergency_stop = True
        
        # Calculate repulsive force based on distance
        if distance < danger_threshold:
            # Strong repulsion
            force_magnitude = (danger_threshold - distance) / danger_threshold
            force_magnitude = force_magnitude * 1.0  # Full strength
        elif distance < caution_threshold:
            # Weak repulsion
            force_magnitude = (caution_threshold - distance) / caution_threshold
            force_magnitude = force_magnitude * 0.3  # Scaled down
        else:
            continue  # No repulsion
        
        # Calculate repulsion direction (away from obstacle)
        sensor_angle_body = sensor_angles_body[direction]
        sensor_angle_world = current_yaw + sensor_angle_body
        
        # Repulsion is opposite to sensor direction
        repulsion_angle = sensor_angle_world + np.pi
        
        # Add to total avoidance vector
        avoid_x += force_magnitude * np.cos(repulsion_angle)
        avoid_y += force_magnitude * np.sin(repulsion_angle)
    
    return avoid_x, avoid_y, emergency_stop


def blend_waypoint_and_avoidance(target_x: float, target_y: float,
                                 current_x: float, current_y: float,
                                 avoid_x: float, avoid_y: float,
                                 avoidance_weight: float = 0.7) -> Tuple[float, float]:
    """
    Blend waypoint attraction with obstacle avoidance
    
    Args:
        target_x, target_y: Target waypoint position
        current_x, current_y: Current drone position
        avoid_x, avoid_y: Avoidance vector
        avoidance_weight: Weight for avoidance (0-1), higher = more avoidance
        
    Returns:
        Tuple of (adjusted_x, adjusted_y) - blended target position
    """
    # Vector to target waypoint
    to_target_x = target_x - current_x
    to_target_y = target_y - current_y
    
    # Normalize target vector
    target_dist = np.hypot(to_target_x, to_target_y)
    if target_dist > 0.01:
        to_target_x /= target_dist
        to_target_y /= target_dist
    
    # Blend: target attraction + obstacle repulsion
    blend_x = (1 - avoidance_weight) * to_target_x + avoidance_weight * avoid_x
    blend_y = (1 - avoidance_weight) * to_target_y + avoidance_weight * avoid_y
    
    # Normalize and scale to reasonable step size
    blend_dist = np.hypot(blend_x, blend_y)
    if blend_dist > 0.01:
        step_size = min(0.3, target_dist)  # Don't overshoot target
        blend_x = (blend_x / blend_dist) * step_size
        blend_y = (blend_y / blend_dist) * step_size
    
    # Calculate adjusted target
    adjusted_x = current_x + blend_x
    adjusted_y = current_y + blend_y
    
    return adjusted_x, adjusted_y


class ObstacleAvoidanceController:
    """
    Real-time obstacle avoidance controller combining reactive and replanning strategies
    """
    
    def __init__(self, grid_map: GridMap, multiranger: Multiranger,
                 replan_threshold: float = 0.8,
                 danger_threshold: float = 0.5,
                 update_frequency: int = 5):
        """
        Initialize obstacle avoidance controller
        
        Args:
            grid_map: GridMap for path planning
            multiranger: Multiranger sensor interface
            replan_threshold: Distance threshold to trigger replanning (meters)
            danger_threshold: Distance for reactive avoidance (meters)
            update_frequency: Update grid map every N measurements
        """
        self.grid_map = grid_map
        self.multiranger = multiranger
        self.replan_threshold = replan_threshold
        self.danger_threshold = danger_threshold
        self.update_frequency = update_frequency
        
        self.detection_history = {}
        self.measurement_count = 0
        self.last_replan_pos = None
        self.replan_cooldown = 0
        
    def update(self, current_x: float, current_y: float, current_yaw: float,
               target_x: float, target_y: float, goal_x: float, goal_y: float,
               waypoints_x: List[float], waypoints_y: List[float],
               current_waypoint_idx: int) -> dict:
        """
        Main update function - call this every control loop iteration
        
        Args:
            current_x, current_y: Current drone position
            current_yaw: Current yaw in radians
            target_x, target_y: Next waypoint target
            goal_x, goal_y: Final goal position
            waypoints_x, waypoints_y: Current path waypoints
            current_waypoint_idx: Index of current target waypoint
            
        Returns:
            Dictionary with control decisions:
            {
                'adjusted_target_x': float,
                'adjusted_target_y': float,
                'emergency_stop': bool,
                'should_replan': bool,
                'new_waypoints_x': Optional[np.ndarray],
                'new_waypoints_y': Optional[np.ndarray],
                'avoidance_active': bool
            }
        """
        result = {
            'adjusted_target_x': target_x,
            'adjusted_target_y': target_y,
            'emergency_stop': False,
            'should_replan': False,
            'new_waypoints_x': None,
            'new_waypoints_y': None,
            'avoidance_active': False
        }
        
        # Decrease replan cooldown
        if self.replan_cooldown > 0:
            self.replan_cooldown -= 1
        
        # Update obstacle map periodically
        self.measurement_count += 1
        if self.measurement_count % self.update_frequency == 0:
            newly_occupied, self.detection_history = update_obstacles_from_multiranger(
                self.grid_map, self.multiranger,
                current_x, current_y, np.rad2deg(current_yaw),
                detection_history=self.detection_history
            )
            
            # Check if path is blocked
            if newly_occupied > 0:
                path_clear = is_path_clear(
                    waypoints_x, waypoints_y, self.grid_map,
                    current_waypoint_idx, look_ahead=5
                )
                
                if not path_clear and self.replan_cooldown == 0:
                    result['should_replan'] = True
        
        # Calculate reactive avoidance vector
        avoid_x, avoid_y, emergency = calculate_avoidance_vector(
            self.multiranger, current_yaw,
            danger_threshold=self.danger_threshold,
            caution_threshold=self.replan_threshold
        )
        
        result['emergency_stop'] = emergency
        
        # Apply avoidance if needed
        avoidance_magnitude = np.hypot(avoid_x, avoid_y)
        if avoidance_magnitude > 0.1:
            result['avoidance_active'] = True
            adjusted_x, adjusted_y = blend_waypoint_and_avoidance(
                target_x, target_y, current_x, current_y,
                avoid_x, avoid_y, avoidance_weight=0.7
            )
            result['adjusted_target_x'] = adjusted_x
            result['adjusted_target_y'] = adjusted_y
        
        # Execute replanning if needed
        if result['should_replan']:
            try:
                new_x, new_y = dynamic_replan(
                    self.grid_map, (current_x, current_y), (goal_x, goal_y)
                )
                result['new_waypoints_x'] = new_x
                result['new_waypoints_y'] = new_y
                self.replan_cooldown = 20  # Cooldown ~2 seconds at 10Hz
                self.last_replan_pos = (current_x, current_y)
            except ValueError as e:
                print(f"⚠️ Replanning failed: {e}")
                result['should_replan'] = False
        
        return result
    
    def get_safe_velocity(self, min_distance: float = None) -> float:
        """
        Calculate safe velocity based on closest obstacle distance
        
        Args:
            min_distance: Manually specify minimum distance (meters), otherwise read from sensors
            
        Returns:
            Safe velocity in m/s (0.0 to 0.5)
        """
        if min_distance is None:
            # Read sensors (mm) and convert to meters
            distances = [
                self.multiranger.front / 1000.0 if self.multiranger.front is not None else float('inf'),
                self.multiranger.back / 1000.0 if self.multiranger.back is not None else float('inf'),
                self.multiranger.left / 1000.0 if self.multiranger.left is not None else float('inf'),
                self.multiranger.right / 1000.0 if self.multiranger.right is not None else float('inf')
            ]
            # Get minimum valid distance (< 4m)
            valid_distances = [d for d in distances if d < 4.0]
            min_distance = min(valid_distances) if valid_distances else 4.0
        
        if min_distance < 0.3:
            return 0.0  # Stop
        elif min_distance < 0.5:
            return 0.1  # Very slow
        elif min_distance < 1.0:
            return 0.2  # Slow
        elif min_distance < 1.5:
            return 0.3  # Medium
        else:
            return 0.5  # Normal speed


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
