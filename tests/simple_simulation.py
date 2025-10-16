"""
Simple Simulation Example: Drone Navigation with Sensor-based Obstacle Detection

A minimal example demonstrating:
1. Mock sensor readings based on true environment
2. Real-time map construction from sensor data
3. Path planning and replanning when obstacles are discovered

ONLY uses methods from cfpilot package (stable methods only)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from cfpilot.mapping import GridMap
from cfpilot.planning.DStarLite.d_star_lite import DStarLite, Node
from cfpilot.visualization import DroneNavigationVisualizer


class PositionFilter:
    """Exponential Moving Average filter for smooth position commands"""
    
    def __init__(self, alpha=0.3):
        """
        Args:
            alpha: Smoothing factor (0-1). Lower = smoother but slower response
        """
        self.alpha = alpha
        self.filtered_x = None
        self.filtered_y = None
    
    def update(self, target_x, target_y):
        """Apply exponential moving average filter"""
        if self.filtered_x is None:
            # Initialize
            self.filtered_x = target_x
            self.filtered_y = target_y
        else:
            # Smooth with EMA
            self.filtered_x = self.alpha * target_x + (1 - self.alpha) * self.filtered_x
            self.filtered_y = self.alpha * target_y + (1 - self.alpha) * self.filtered_y
        
        return self.filtered_x, self.filtered_y
    
    def reset(self, x, y):
        """Reset filter to new position"""
        self.filtered_x = x
        self.filtered_y = y


class MockMultiranger:
    """
    Mock Multiranger sensor that mimics cflib.utils.multiranger.Multiranger API
    This allows using the EXACT same interface as real Crazyflie hardware
    
    API Compatibility with cflib.utils.multiranger.Multiranger:
    ✓ Properties: front, back, left, right, up, down
    ✓ Returns None if no obstacle detected (>2m or out of range)
    ✓ Returns distance in meters if obstacle detected
    ✓ Same data format: distances in meters or None
    
    Usage in simulation:
        sensor = MockMultiranger(obstacles, drone_position, drone_yaw)
        sensor.update_pose(x, y, yaw)  # Update position each step
        distance = sensor.front  # Read sensor (just like real hardware)
    
    Transition to real hardware (cflib):
        from cflib.utils.multiranger import Multiranger
        sensor = Multiranger(scf, rate_ms=100)
        sensor.start()
        distance = sensor.front  # SAME property access!
    """
    
    def __init__(self, obstacles, drone_position=(0, 0), drone_yaw=0):
        """
        Args:
            obstacles: List of (x, y, radius) tuples representing circular obstacles
            drone_position: Current (x, y) position of drone
            drone_yaw: Current yaw angle in radians
        """
        self.obstacles = obstacles
        self._drone_x = drone_position[0]
        self._drone_y = drone_position[1]
        self._drone_yaw = drone_yaw
        
        # Sensor readings (mimics Multiranger properties)
        self._front_distance = None
        self._back_distance = None
        self._left_distance = None
        self._right_distance = None
        self._up_distance = None
        self._down_distance = None
        
        # Multiranger sensor specs
        self._max_range = 2.0  # 2 meters max range
        self._min_range = 0.01  # 1cm min range
    
    def update_pose(self, x, y, yaw):
        """Update drone position and orientation, then refresh sensor readings"""
        self._drone_x = x
        self._drone_y = y
        self._drone_yaw = yaw
        self._scan()
    
    def _raycast(self, direction_angle, max_range=2.0):
        """
        Raycast in a specific direction to find nearest obstacle
        Uses multiple rays (±5°) for better detection coverage
        
        Args:
            direction_angle: Angle in radians (world frame)
            max_range: Maximum detection range in meters
            
        Returns:
            Distance in meters or None if nothing detected
        """
        min_dist = max_range
        
        # Use 3 raycasts per direction for better coverage (matches real sensor cone)
        for angle_offset in [-0.087, 0.0, 0.087]:  # -5°, 0°, +5°
            scan_angle = direction_angle + angle_offset
            dx = np.cos(scan_angle)
            dy = np.sin(scan_angle)
            
            for obs_x, obs_y, radius in self.obstacles:
                # Vector from drone to obstacle center
                to_obs = np.array([obs_x - self._drone_x, obs_y - self._drone_y])
                
                # Project onto ray direction
                proj = np.dot(to_obs, [dx, dy])
                
                if proj > 0:  # Obstacle is in front of sensor
                    # Perpendicular distance to ray
                    perp_dist = np.linalg.norm(to_obs - proj * np.array([dx, dy]))
                    
                    # Check if ray intersects obstacle circle
                    if perp_dist <= radius:
                        # Distance to obstacle surface
                        hit_dist = proj - np.sqrt(radius**2 - perp_dist**2)
                        if hit_dist > self._min_range:
                            min_dist = min(min_dist, hit_dist)
        
        # Return None if out of range (mimics real Multiranger behavior)
        return min_dist if min_dist < max_range else None
    
    def _scan(self):
        """Perform sensor scan in all directions and update readings"""
        # Sensor directions relative to drone yaw (body frame)
        # Front: 0°, Right: -90°, Back: 180°, Left: +90°
        self._front_distance = self._raycast(self._drone_yaw, self._max_range)
        self._back_distance = self._raycast(self._drone_yaw + np.pi, self._max_range)
        self._left_distance = self._raycast(self._drone_yaw + np.pi/2, self._max_range)
        self._right_distance = self._raycast(self._drone_yaw - np.pi/2, self._max_range)
        
        # Up/Down not used in 2D simulation
        self._up_distance = None
        self._down_distance = None
    
    # Properties that match cflib.utils.multiranger.Multiranger API
    @property
    def front(self):
        """Distance to obstacle in front (meters) or None"""
        return self._front_distance
    
    @property
    def back(self):
        """Distance to obstacle behind (meters) or None"""
        return self._back_distance
    
    @property
    def left(self):
        """Distance to obstacle on left (meters) or None"""
        return self._left_distance
    
    @property
    def right(self):
        """Distance to obstacle on right (meters) or None"""
        return self._right_distance
    
    @property
    def up(self):
        """Distance to obstacle above (meters) or None"""
        return self._up_distance
    
    @property
    def down(self):
        """Distance to obstacle below (meters) or None"""
        return self._down_distance
    
    def check_collision(self, x, y, safety_radius=0.1):
        """
        Check if position collides with any obstacle
        
        Args:
            x, y: Position to check
            safety_radius: Extra safety margin around drone
            
        Returns:
            (collided, obstacle_info)
        """
        for obs_x, obs_y, obs_radius in self.obstacles:
            dist_to_obs = np.hypot(x - obs_x, y - obs_y)
            if dist_to_obs < (obs_radius + safety_radius):
                return True, (obs_x, obs_y, obs_radius)
        return False, None


def calculate_repulsion(sensors, yaw, danger_dist=0.7):
    """
    Calculate repulsion vector from obstacles using sensor data
    Compatible with Multiranger sensor format (Adapted from path_planning_v2.py)
    
    Args:
        sensors: Dict with sensor distances in meters (front/back/left/right)
                 Format matches Multiranger properties (distance or None)
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
        dist = sensors.get(direction)
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


def blend_target_and_avoidance(target, current, repulsion, weight=0.7):
    """
    Blend waypoint attraction with obstacle repulsion
    (Adapted from path_planning_v2.py)
    
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


def update_map(grid_map, sensors, x, y, yaw, safety_margin=0.15):
    """
    Update grid map based on sensor readings with safety margin inflation
    
    Args:
        grid_map: GridMap instance
        sensors: Sensor readings dict
        x, y: Current position
        yaw: Current yaw in radians
        safety_margin: Inflate obstacles by this amount (meters)
    """
    directions = {'front': 0, 'right': -np.pi/2, 'back': np.pi, 'left': np.pi/2}
    
    cells_updated = 0
    for direction, distance in sensors.items():
        if distance is None or distance > 2.0 or distance < 0.05:
            continue
        
        # Calculate obstacle position
        angle = yaw + directions[direction]
        obs_x = x + distance * np.cos(angle)
        obs_y = y + distance * np.sin(angle)
        
        # Mark obstacle AND surrounding cells for safety
        # This inflates obstacles to prevent flying through narrow gaps
        for dx in np.linspace(-safety_margin, safety_margin, 5):
            for dy in np.linspace(-safety_margin, safety_margin, 5):
                success = grid_map.set_value_from_xy_pos(obs_x + dx, obs_y + dy, 1.0)
                if success:
                    cells_updated += 1
    
    return cells_updated


def plan_path_with_dstar(grid_map, start_pos, goal_pos):
    """
    Plan path using D* Lite from cfpilot.planning
    
    Args:
        grid_map: GridMap instance
        start_pos: (x, y) in meters
        goal_pos: (x, y) in meters
        
    Returns:
        (path_found, waypoints_x, waypoints_y) in meters
    """
    # Create D* Lite planner
    planner = DStarLite(grid_map, obstacle_threshold=0.5, show_animation=False)
    
    # Use the run() method which takes world coordinates directly
    # and returns path in world coordinates
    path_found, pathx, pathy = planner.run(
        start=start_pos, 
        goal=goal_pos,
        simplify=True,
        smooth=True
    )
    
    if not path_found or len(pathx) == 0:
        return False, [], []
    
    return True, list(pathx), list(pathy)


def simple_simulation(show_animation=True, animation_speed=0.01):
    """Run a simple simulation demonstrating sensor-based navigation
    
    Args:
        show_animation: If True, shows real-time animation of drone movement
        animation_speed: Pause time between frames (seconds). Lower = faster
    """
    
    print("="*60)
    print("Simple Crazyflie Simulation")
    print("Using ONLY cfpilot methods")
    if show_animation:
        print("With real-time animation")
    print("="*60)
    
    # Setup environment
    print("\n📍 Setting up environment...")
    
    # Create EMPTY map - drone has NO knowledge of obstacles yet!
    # Only boundaries are known
    grid_map = GridMap(
        width=100, height=60,
        resolution=0.05,
        center_x=2.5, center_y=1.5
    )
    grid_map.occupy_boundaries(boundary_width=2)
    
    print("   ✅ Grid map created with ONLY boundaries")
    print("   ℹ️  Drone has no knowledge of obstacles yet")
    
    # Start and goal
    start = (0.5, 1.5)
    goal = (4.5, 1.0)
    
    # True obstacles exist in the environment (for sensor simulation)
    # These are NOT in the grid_map - drone will discover them via sensors!
    true_obstacles = [
        (1.5, 1.8, 0.25),
        (2.5, 0.8, 0.25),
        (2.5, 2.0, 0.25),
        (3.5, 1.8, 0.25),

    ]
    
    # Mock sensor knows about true obstacles (to simulate real sensor readings)
    # Uses EXACT same API as cflib.utils.multiranger.Multiranger
    # 
    # TO USE WITH REAL CRAZYFLIE, replace MockMultiranger with:
    #   from cflib.utils.multiranger import Multiranger
    #   sensor = Multiranger(scf, rate_ms=100)
    #   sensor.start()
    # Then use: sensor.front, sensor.back, sensor.left, sensor.right
    sensor = MockMultiranger(true_obstacles, drone_position=start, drone_yaw=0)
    
    print(f"   Start: {start}")
    print(f"   Goal: {goal}")
    print(f"   True obstacles in environment: {len(true_obstacles)} (unknown to drone)")
    
    # Setup animation if enabled
    visualizer = None
    if show_animation:
        visualizer = DroneNavigationVisualizer(
            xlim=(0, 5), 
            ylim=(0, 3), 
            figsize=(14, 9), 
            animation_speed=animation_speed
        )
        visualizer.setup(
            start=start, 
            goal=goal, 
            true_obstacles=true_obstacles, 
            grid_map=grid_map
        )
    
    # Initial path planning with EMPTY map (only boundaries known!)
    print("\n🛤️  Planning initial path with EMPTY map...")
    print("   (Drone knows only about boundaries, not obstacles)")
    path_found, waypoints_x, waypoints_y = plan_path_with_dstar(grid_map, start, goal)
    
    if not path_found:
        print("   ❌ Could not find initial path!")
        return
    
    print(f"   ✅ Initial path planned: {len(waypoints_x)} waypoints")
    print("   ℹ️  This path likely goes through obstacles (drone doesn't know yet!)")
    
    if visualizer:
        visualizer.update_path(waypoints_x, waypoints_y)
        plt.pause(0.5)
    
    # Simulate movement with real-time obstacle discovery
    print("\n🚁 Simulating flight with HYBRID navigation:")
    print("   • Global path planning with D* Lite")
    print("   • Local reactive avoidance with potential fields")
    print("   • Position filtering for smooth trajectories")
    print("   • Sensors detect obstacles → update map → replan if needed")
    
    x, y = start
    yaw = 0.0
    trajectory_x, trajectory_y = [x], [y]
    replan_positions = []
    
    # Initialize position filter for smooth commands
    pos_filter = PositionFilter(alpha=0.4)  # 40% new, 60% old (smooth but responsive)
    pos_filter.reset(x, y)
    
    waypoint_idx = 0
    replans = 0
    max_steps = 400
    replan_cooldown = 0  # Cooldown between replans
    stuck_counter = 0  # Track if drone is stuck
    last_position = (x, y)
    
    for step in range(max_steps):
        # Check if reached goal
        if np.hypot(goal[0] - x, goal[1] - y) < 0.15:
            print(f"✅ Goal reached at step {step}!")
            break
        
        # Update sensor with current pose and read distances
        sensor.update_pose(x, y, yaw)
        
        # Read sensor distances (using Multiranger API)
        sensors = {
            'front': sensor.front,
            'back': sensor.back,
            'left': sensor.left,
            'right': sensor.right
        }
        
        # Calculate repulsion forces from nearby obstacles
        rep_x, rep_y, emergency = calculate_repulsion(sensors, yaw, danger_dist=0.6)
        
        # Check if stuck (not moving)
        dist_moved = np.hypot(x - last_position[0], y - last_position[1])
        if dist_moved < 0.005:  # Less than 5mm movement
            stuck_counter += 1
        else:
            stuck_counter = 0
        last_position = (x, y)
        
        # Handle being stuck between obstacles
        if stuck_counter > 20:  # Stuck for 20 steps
            print(f"   ⚠️  Drone stuck at step {step}! Attempting escape...")
            
            # Try to escape by skipping ahead in waypoints
            waypoint_idx = min(waypoint_idx + 3, len(waypoints_x) - 1)
            stuck_counter = 0
            
            # Force replan if still stuck after trying to skip
            if replan_cooldown == 0:
                print(f"   🔄 Replanning to escape stuck situation...")
                path_found, waypoints_x, waypoints_y = plan_path_with_dstar(
                    grid_map, (x, y), goal
                )
                
                if path_found and len(waypoints_x) > 0:
                    waypoint_idx = 0
                    replans += 1
                    replan_positions.append((x, y))
                    replan_cooldown = 30  # Long cooldown to avoid oscillation
                    print(f"   ✅ Escape replan: {len(waypoints_x)} waypoints")
                    
                    if visualizer:
                        visualizer.update_path(waypoints_x, waypoints_y)
                        visualizer.mark_replan(x, y)
                        plt.pause(0.3)
            
            continue
        
        # Handle emergency: just slow down and rely on repulsion
        if emergency:
            # Don't stop completely - let repulsion guide us
            # This prevents getting truly stuck
            pass  # Continue with normal movement logic below
        
        # Decrement cooldown
        if replan_cooldown > 0:
            replan_cooldown -= 1
        
        # Update map based on sensor readings periodically (not every step)
        # This is where the drone "discovers" obstacles!
        if step % 3 == 0:  # Update map every 3 steps to reduce noise
            cells_updated = update_map(grid_map, sensors, x, y, yaw, safety_margin=0.15)
        else:
            cells_updated = 0
        
        # Update animation
        if visualizer:
            # Update drone position and trajectory
            visualizer.update_drone(x, y, yaw)
            
            # Update sensor readings
            visualizer.update_sensors(sensors)
            
            # Update repulsion force
            visualizer.update_repulsion(rep_x, rep_y)
            
            # Set status flags
            visualizer.set_status(
                stuck=(stuck_counter > 10),
                emergency=emergency,
                avoiding=(np.hypot(rep_x, rep_y) > 0.1)
            )
            
            # Render visualization
            visualizer.render()
        
        # Check if path is blocked (but only replan if cooldown expired)
        if cells_updated > 0 and waypoint_idx < len(waypoints_x) and replan_cooldown == 0:
            # Check upcoming waypoints
            blocked = False
            for i in range(waypoint_idx, min(waypoint_idx + 5, len(waypoints_x))):
                wx, wy = waypoints_x[i], waypoints_y[i]
                idx_x, idx_y = grid_map.get_xy_index_from_xy_pos(wx, wy)
                if idx_x is not None:
                    val = grid_map.get_value_from_xy_index(idx_x, idx_y)
                    if val is not None and val > 0.5:
                        blocked = True
                        break
            
            # Replan if blocked (with cooldown to reduce oscillation)
            if blocked:
                print(f"   ⚠️  Obstacle detected at step {step}! Replanning...")
                path_found, waypoints_x, waypoints_y = plan_path_with_dstar(
                    grid_map, (x, y), goal
                )
                
                if path_found and len(waypoints_x) > 0:
                    waypoint_idx = 0
                    replans += 1
                    replan_positions.append((x, y))
                    replan_cooldown = 25  # Add cooldown to prevent rapid replanning
                    print(f"   ✅ Replanned with {len(waypoints_x)} waypoints")
                    
                    if visualizer:
                        # Update path line
                        visualizer.update_path(waypoints_x, waypoints_y)
                        # Mark replan position
                        visualizer.mark_replan(x, y)
                        plt.pause(0.5)
                else:
                    print("   ❌ Replanning failed!")
                    break
        
        # Move towards next waypoint
        if waypoint_idx < len(waypoints_x):
            target_x, target_y = waypoints_x[waypoint_idx], waypoints_y[waypoint_idx]
            
            # Check if reached waypoint
            if np.hypot(target_x - x, target_y - y) < 0.12:  # Larger tolerance
                waypoint_idx += 1
                stuck_counter = 0  # Reset stuck counter
                if waypoint_idx % 5 == 0:  # Print progress every 5 waypoints
                    print(f"   ✓ Waypoint {waypoint_idx}/{len(waypoints_x)} reached")
                continue
            
            # === HYBRID NAVIGATION: Blend global path with local avoidance ===
            # If obstacles are nearby, blend waypoint following with repulsion
            repulsion_magnitude = np.hypot(rep_x, rep_y)
            
            if repulsion_magnitude > 0.08:  # Threshold for avoidance activation
                # Active avoidance: blend target attraction with obstacle repulsion
                # Adapt weight based on repulsion strength (stronger repulsion = more cautious)
                avoidance_weight = min(0.75, repulsion_magnitude * 1.8)  # Max 75% avoidance
                
                next_x, next_y = blend_target_and_avoidance(
                    target=(target_x, target_y),
                    current=(x, y),
                    repulsion=(rep_x, rep_y),
                    weight=avoidance_weight
                )
            else:
                # No obstacles nearby: normal waypoint following
                dx = target_x - x
                dy = target_y - y
                dist = np.hypot(dx, dy)
                step_size = 0.03  # 3cm per step (increased from 2cm for faster progress)
                
                if dist > 0:
                    next_x = x + step_size * dx / dist
                    next_y = y + step_size * dy / dist
                else:
                    next_x, next_y = x, y
            
            # Apply position filter for smooth trajectory
            filtered_x, filtered_y = pos_filter.update(next_x, next_y)
            
            # SAFETY CHECK: Verify no collision before moving
            collided, obs_info = sensor.check_collision(filtered_x, filtered_y, safety_radius=0.08)
            
            if collided:
                # CRITICAL: About to collide! Don't move, increase repulsion
                print(f"   ⚠️  COLLISION WARNING at step {step}! Blocking move to ({filtered_x:.2f}, {filtered_y:.2f})")
                stuck_counter += 5  # Accelerate stuck detection
                
                if show_animation:
                    ax.plot(filtered_x, filtered_y, 'rx', markersize=15, zorder=13, 
                           markeredgewidth=3, label='Blocked position')
                    plt.pause(0.1)
                
                # Don't update position - stay where we are
                continue
            
            # Update position and orientation
            dx = filtered_x - x
            dy = filtered_y - y
            if abs(dx) > 0.001 or abs(dy) > 0.001:
                yaw = np.arctan2(dy, dx)
                x, y = filtered_x, filtered_y
            
            trajectory_x.append(x)
            trajectory_y.append(y)
    
    print(f"\n📊 Results:")
    print(f"   Trajectory points: {len(trajectory_x)}")
    print(f"   Replanning events: {replans}")
    print(f"   Final position: ({x:.2f}, {y:.2f})")
    print(f"   Distance to goal: {np.hypot(goal[0] - x, goal[1] - y):.2f}m")
    
    # Check if trajectory was collision-free
    collision_free = True
    for tx, ty in zip(trajectory_x, trajectory_y):
        collided, _ = sensor.check_collision(tx, ty, safety_radius=0.0)
        if collided:
            collision_free = False
            break
    
    if collision_free:
        print(f"   ✅ Trajectory was collision-free!")
    else:
        print(f"   ⚠️  Warning: Trajectory had collisions")
    
    # Final visualization
    if visualizer:
        visualizer.finalize('simple_simulation_result.png')
    else:
        # Static visualization
        print("\n📈 Creating visualization...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot discovered map
        grid_map.plot_grid_map(ax=ax, use_world_coords=True)
        
        # Plot true obstacles (in red)
        for obs_x, obs_y, radius in true_obstacles:
            circle = Circle((obs_x, obs_y), radius, color='red', alpha=0.3, 
                           linewidth=2, edgecolor='red', label='True obstacle' if obs_x == true_obstacles[0][0] else '')
            ax.add_patch(circle)
        
        # Plot trajectory
        ax.plot(trajectory_x, trajectory_y, 'g-', linewidth=2, label='Trajectory')
        
        # Plot replan positions
        if replan_positions:
            rx, ry = zip(*replan_positions)
            ax.plot(rx, ry, 'ro', markersize=12, label=f'Replan ({replans})')
        
        # Plot start and goal
        ax.plot(start[0], start[1], 'g^', markersize=15, label='Start')
        ax.plot(goal[0], goal[1], 'm*', markersize=20, label='Goal')
        
        # Plot current path
        if len(waypoints_x) > 0:
            ax.plot(waypoints_x, waypoints_y, 'b--', alpha=0.5, label='Path')
        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(f'Simple Simulation Result ({replans} replans)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        plt.tight_layout()
        plt.savefig('simple_simulation_result.png', dpi=150)
        print("💾 Saved to simple_simulation_result.png")
        plt.show()
    
    print("\n" + "="*60)
    print("✅ Simulation complete!")
    print("="*60)


if __name__ == "__main__":
    # Run with animation (default)
    simple_simulation(show_animation=True, animation_speed=0.01)
    
    # Or run without animation for faster execution:
    # simple_simulation(show_animation=False)
