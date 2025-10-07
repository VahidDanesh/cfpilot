"""
Path Planning & Obstacle Avoidance Module - Streamlined Version

Simple, composable functions for Crazyflie navigation with real-time obstacle avoidance.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
from cflib.utils.multiranger import Multiranger

from cfpilot.mapping import GridMap
from cfpilot.sensors import SensorFilter, read_multiranger_raw, get_min_distance
from cfpilot.planning.DStarLite.d_star_lite import DStarLite, Node
from cfpilot.planning.CubicSpline.spline_continuity import Spline2D


# ============================================================================
# Environment Setup
# ============================================================================

def create_test_environment(resolution: float = 0.05, 
                            width: float = 4.6, 
                            height: float = 2.6) -> GridMap:
    """Create a test grid map with obstacles"""
    grid_map = GridMap(
        width=int(width / resolution),
        height=int(height / resolution),
        resolution=resolution,
        center_x=width / 2,
        center_y=height / 2
    )
    
    # Add obstacles
    grid_map.add_obstacle(center_x=1.5, center_y=1.6, radius=0.3)
    grid_map.add_obstacle(center_x=2.5, center_y=1.5, radius=0.3)
    grid_map.add_obstacle(center_x=3.2, center_y=0.7, radius=0.3)
    grid_map.add_obstacle(center_x=3.2, center_y=2.3, radius=0.3)
    
    grid_map.occupy_boundaries(boundary_width=2)
    grid_map.expand_grid(occupied_val=0.5)
    
    return grid_map


# ============================================================================
# Core Path Planning
# ============================================================================

def plan_path(grid_map: GridMap,
              start: Tuple[float, float],
              goal: Tuple[float, float],
              simplify: bool = True) -> Tuple[List[float], List[float]]:
    """
    Plan a path using D* Lite and return waypoints in meters
    
    Returns:
        Tuple of (waypoints_x, waypoints_y) in meters
    """
    # Get obstacles
    ox, oy = np.where(grid_map.data > 0.5)
    
    # Convert to grid indices
    start_idx = grid_map.get_xy_index_from_xy_pos(start[0], start[1])
    goal_idx = grid_map.get_xy_index_from_xy_pos(goal[0], goal[1])
    
    # Plan
    planner = DStarLite(ox=ox, oy=oy)
    found, pathx, pathy = planner.main(
        Node(start_idx[0], start_idx[1]),
        Node(goal_idx[0], goal_idx[1]),
        spoofed_ox=[], spoofed_oy=[]
    )
    
    if not found:
        raise ValueError("No path found")
    
    # Simplify
    if simplify:
        pathx, pathy = GridMap.simplify_path(pathx, pathy, grid_map)
    
    # Convert to world coordinates
    pos_x, pos_y = grid_map.get_xy_poss_from_xy_indexes(pathx, pathy)
    
    return list(pos_x), list(pos_y)


# ============================================================================
# Obstacle Avoidance - Potential Field with Filtered Sensors
# ============================================================================

def calculate_repulsion(filtered_readings: Dict[str, Optional[float]], 
                       yaw: float,
                       danger_dist: float = 0.5) -> Tuple[float, float, bool]:
    """
    Calculate repulsion vector from obstacles using filtered sensor data
    
    Args:
        filtered_readings: Dict with filtered sensor distances in meters
        yaw: Current yaw in radians
        danger_dist: Distance threshold for repulsion
        
    Returns:
        (repulsion_x, repulsion_y, emergency_stop)
    """
    # Sensor angles in body frame
    angles = {'front': 0.0, 'right': -np.pi/2, 'back': np.pi, 'left': np.pi/2}
    
    repulsion_x, repulsion_y = 0.0, 0.0
    emergency = False
    
    for direction in ['front', 'back', 'left', 'right']:
        dist = filtered_readings.get(direction)
        if dist is None or dist > danger_dist:
            continue
        
        # Emergency stop if too close
        if dist < 0.15:  # 15cm emergency threshold
            emergency = True
        
        # Calculate repulsion (inverse square for stronger force when close)
        force = (danger_dist - dist) / danger_dist
        force = force ** 2  # Quadratic falloff
        
        # Direction in world frame (away from obstacle)
        angle = yaw + angles[direction] + np.pi  # +π to point away
        
        repulsion_x += force * np.cos(angle)
        repulsion_y += force * np.sin(angle)
    
    # Apply small deadzone to avoid jitter
    if abs(repulsion_x) < 0.01:
        repulsion_x = 0.0
    if abs(repulsion_y) < 0.01:
        repulsion_y = 0.0
    
    return repulsion_x, repulsion_y, emergency


def calculate_repulsion_legacy(multiranger: Multiranger, 
                               yaw: float,
                               danger_dist: float = 0.5) -> Tuple[float, float, bool]:
    """
    DEPRECATED: Use calculate_repulsion() with filtered readings instead
    
    Calculate repulsion vector from raw multiranger data
    """
    angles = {'front': 0.0, 'right': -np.pi/2, 'back': np.pi, 'left': np.pi/2}
    
    repulsion_x, repulsion_y = 0.0, 0.0
    emergency = False
    sensors = read_multiranger_raw(multiranger)
    
    for direction in ['front', 'back', 'left', 'right']:
        dist = sensors.get(direction)
        if dist is None or dist > danger_dist:
            continue
        
        if dist < 0.15:
            emergency = True
        
        force = (danger_dist - dist) / danger_dist
        force = force ** 2
        
        angle = yaw + angles[direction] + np.pi
        
        repulsion_x += force * np.cos(angle)
        repulsion_y += force * np.sin(angle)
    
    if abs(repulsion_x) < 0.01:
        repulsion_x = 0.0
    if abs(repulsion_y) < 0.01:
        repulsion_y = 0.0
        
    return repulsion_x, repulsion_y, emergency


def blend_target_and_avoidance(target: Tuple[float, float],
                               current: Tuple[float, float],
                               repulsion: Tuple[float, float],
                               weight: float = 0.7) -> Tuple[float, float]:
    """
    Blend waypoint attraction with obstacle repulsion
    
    Args:
        target: Target position (x, y)
        current: Current position (x, y)
        repulsion: Repulsion vector (x, y)
        weight: Avoidance weight (0-1), higher = more cautious
        
    Returns:
        Adjusted target position (x, y)
    """
    # Vector to target
    to_target = (target[0] - current[0], target[1] - current[1])
    target_dist = np.hypot(*to_target)
    
    if target_dist < 0.03:
        return target
    
    # Normalize
    to_target = (to_target[0] / target_dist, to_target[1] / target_dist)
    
    # Blend
    blend_x = (1 - weight) * to_target[0] + weight * repulsion[0]
    blend_y = (1 - weight) * to_target[1] + weight * repulsion[1]
    
    # Scale to reasonable step size
    blend_dist = np.hypot(blend_x, blend_y)
    if blend_dist > 0.01:
        step_size = min(0.3, target_dist)
        blend_x = (blend_x / blend_dist) * step_size
        blend_y = (blend_y / blend_dist) * step_size
    
    return (current[0] + blend_x, current[1] + blend_y)


def adaptive_velocity(min_distance: float) -> float:
    """Calculate safe velocity based on closest obstacle"""
    if min_distance < 0.3:
        return 0.0
    elif min_distance < 0.5:
        return 0.1
    elif min_distance < 1.0:
        return 0.2
    elif min_distance < 1.5:
        return 0.3
    else:
        return 0.5


# ============================================================================
# Dynamic Replanning
# ============================================================================

def update_map_with_sensors(grid_map: GridMap,
                           sensors: Dict[str, Optional[float]],
                           position: Tuple[float, float],
                           yaw: float,
                           history: Dict = None) -> Tuple[int, Dict]:
    """
    Update grid map with detected obstacles
    
    Args:
        grid_map: GridMap to update
        sensors: Sensor readings in meters
        position: Current position (x, y)
        yaw: Current yaw in degrees
        history: Detection history for confirmation
        
    Returns:
        (number of newly occupied cells, updated history)
    """
    if history is None:
        history = {}
    
    angles = {'front': 0, 'right': -90, 'back': 180, 'left': 90}
    newly_occupied = 0
    
    for direction in ['front', 'back', 'left', 'right']:
        dist = sensors.get(direction)
        if dist is None or dist > 3.0 or dist < 0.1:
            continue
        
        # Calculate obstacle position
        angle_rad = np.radians(yaw + angles[direction])
        obs_x = position[0] + dist * np.cos(angle_rad)
        obs_y = position[1] + dist * np.sin(angle_rad)
        
        # Get grid index
        try:
            idx = grid_map.get_xy_index_from_xy_pos(obs_x, obs_y)
            key = tuple(idx)
            
            # Confirm detection (3 times)
            history[key] = history.get(key, 0) + 1
            if history[key] == 3:
                grid_map.data[idx[0], idx[1]] = 1.0
                newly_occupied += 1
        except:
            continue
    
    return newly_occupied, history


def is_path_blocked(waypoints: List[Tuple[float, float]], 
                   grid_map: GridMap,
                   start_idx: int = 0,
                   look_ahead: int = 5) -> bool:
    """Check if upcoming waypoints are blocked"""
    end_idx = min(start_idx + look_ahead, len(waypoints))
    
    for i in range(start_idx, end_idx - 1):
        x1, y1 = waypoints[i]
        x2, y2 = waypoints[i + 1]
        
        idx1 = grid_map.get_xy_index_from_xy_pos(x1, y1)
        idx2 = grid_map.get_xy_index_from_xy_pos(x2, y2)
        
        n = int(np.hypot(idx2[0] - idx1[0], idx2[1] - idx1[1])) * 2 + 1
        xs = np.linspace(idx1[0], idx2[0], n).astype(int)
        ys = np.linspace(idx1[1], idx2[1], n).astype(int)
        
        if np.any(grid_map.data[xs, ys] > 0.5):
            return True
    
    return False


# ============================================================================
# High-Level Control Loop Helper
# ============================================================================

def fly_with_avoidance(position_commander,
                      multiranger: Multiranger,
                      grid_map: GridMap,
                      start: Tuple[float, float],
                      goal: Tuple[float, float],
                      get_state_fn,
                      height: float = 0.5,
                      max_time: float = 60.0,
                      update_interval: float = 0.1):
    """
    Complete flight control loop with obstacle avoidance
    
    Args:
        position_commander: PositionHlCommander instance
        multiranger: Multiranger instance
        grid_map: GridMap for planning
        start: Start position (x, y)
        goal: Goal position (x, y)
        get_state_fn: Function that returns (x, y, z, yaw_rad)
        height: Flight height
        max_time: Mission timeout
        update_interval: Control loop interval (seconds)
    """
    import time
    
    # Initial planning
    print(f"📍 Planning path from {start} to {goal}")
    waypoints_x, waypoints_y = plan_path(grid_map, start, goal)
    waypoints = list(zip(waypoints_x, waypoints_y))
    print(f"✅ Path planned with {len(waypoints)} waypoints")
    
    current_idx = 0
    history = {}
    update_count = 0
    replan_cooldown = 0
    stuck_counter = 0
    start_time = time.time()
    
    while current_idx < len(waypoints) and time.time() - start_time < max_time:
        x, y, z, yaw = get_state_fn()
        
        sensors = read_multiranger_raw(multiranger)
        min_dist = get_min_distance(sensors)
        
        # Target waypoint
        target = waypoints[current_idx]
        
        # Calculate repulsion
        rep_x, rep_y, emergency = calculate_repulsion(sensors, yaw, danger_dist=0.5)
        
        # Handle emergency
        if emergency:
            print("🛑 Emergency stop!")
            position_commander.go_to(x, y, height, velocity=0.0)
            time.sleep(0.5)
            stuck_counter += 1
            
            if stuck_counter > 30:  # 3 seconds
                print("⚠️ Forcing replan...")
                replan_cooldown = 0  # Allow immediate replan
                stuck_counter = 0
            continue
        
        # Update map periodically
        update_count += 1
        if update_count % 5 == 0:
            newly_occupied, history = update_map_with_sensors(
                grid_map, sensors, (x, y), np.degrees(yaw), history
            )
            
            # Check for replanning
            if newly_occupied > 0 and replan_cooldown == 0:
                if is_path_blocked(waypoints, grid_map, current_idx):
                    print("🔄 Path blocked - replanning...")
                    try:
                        grid_map.expand_grid(occupied_val=0.5)
                        waypoints_x, waypoints_y = plan_path(grid_map, (x, y), goal)
                        waypoints = list(zip(waypoints_x, waypoints_y))
                        current_idx = 0
                        replan_cooldown = 20
                        print(f"   New path: {len(waypoints)} waypoints")
                    except ValueError as e:
                        print(f"   Replanning failed: {e}")
        
        if replan_cooldown > 0:
            replan_cooldown -= 1
        
        # Calculate command
        if np.hypot(rep_x, rep_y) > 0.1:
            # Avoidance active
            cmd_x, cmd_y = blend_target_and_avoidance(
                target, (x, y), (rep_x, rep_y), weight=0.7
            )
        else:
            # Normal waypoint following
            cmd_x, cmd_y = target
        
        # Execute with adaptive velocity
        velocity = adaptive_velocity(min_dist)
        position_commander.go_to(cmd_x, cmd_y, height, velocity=velocity)
        
        # Check waypoint reached
        dist_to_target = np.hypot(x - target[0], y - target[1])
        if dist_to_target < 0.15:
            current_idx += 1
            stuck_counter = 0
            print(f"✅ Waypoint {current_idx}/{len(waypoints)} reached")
        
        time.sleep(update_interval)
    
    if current_idx >= len(waypoints):
        print("🎯 Goal reached!")
        return True
    else:
        print("⏰ Timeout")
        return False


# ============================================================================
# Visualization
# ============================================================================

def visualize_path(grid_map: GridMap,
                  start: Tuple[float, float],
                  goal: Tuple[float, float],
                  waypoints: Optional[List[Tuple[float, float]]] = None):
    """Visualize grid map and path"""
    plt.figure(figsize=(12, 8))
    grid_map.plot_grid_map(use_world_coords=True)
    
    if waypoints:
        wx, wy = zip(*waypoints)
        plt.plot(wx, wy, 'b.-', linewidth=2, markersize=8, label='Path')
    
    plt.plot(start[0], start[1], 'g^', markersize=12, label='Start')
    plt.plot(goal[0], goal[1], 'm*', markersize=12, label='Goal')
    
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Path Planning with Obstacles')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================================
# Example Usage with Sensor Filtering
# ============================================================================

if __name__ == "__main__":
    grid_map = create_test_environment()
    start = (0.7, 1.3)
    goal = (3.9, 1.0)
    
    waypoints_x, waypoints_y = plan_path(grid_map, start, goal)
    waypoints = list(zip(waypoints_x, waypoints_y))
    
    print(f"✅ Planned {len(waypoints)} waypoints")
    print(f"   Start: {start}")
    print(f"   Goal: {goal}")
    
    visualize_path(grid_map, start, goal, waypoints)
