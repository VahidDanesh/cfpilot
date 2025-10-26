"""
Simple Simulation: Autonomous Landing Pad Detection Mission

Simulates the EXACT algorithm from fly_autonomous.py:
1. Navigate to goal using D* Lite with dynamic replanning
2. Search and detect landing pad at goal (sweep/spiral pattern)
3. Land on detected pad
4. Return to start
5. Search and detect landing pad at start
6. Land on detected pad at start
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time

from cfpilot.mapping import GridMap
from cfpilot.planning.DStarLite.d_star_lite import DStarLite
from cfpilot.visualization import DroneNavigationVisualizer
from cfpilot.detection import LandingPadDetector



# ============================================================================
# HELPER CLASSES (matching fly_autonomous.py)
# ============================================================================

class ExponentialFilter:
    """Exponential smoothing filter for 2D values (position, velocity)."""
    def __init__(self, alpha=0.5, initial_x=None, initial_y=None):
        self.alpha = alpha
        self.x = self._to_scalar(initial_x)
        self.y = self._to_scalar(initial_y)
    
    @staticmethod
    def _to_scalar(val):
        """Robustly convert any scalar or numpy array to float"""
        if val is None:
            return None
        arr = np.asarray(val)
        if arr.shape == ():
            return float(arr)
        elif arr.size == 1:
            return float(arr.ravel()[0])
        else:
            raise ValueError("ExponentialFilter expects a scalar or an array of size 1.")
    
    def update(self, x, y):
        x = self._to_scalar(x)
        y = self._to_scalar(y)
        
        if self.x is None:
            self.x, self.y = x, y
        else:
            self.x = self.alpha * x + (1 - self.alpha) * self.x
            self.y = self.alpha * y + (1 - self.alpha) * self.y
        return self.x, self.y
    
    def reset(self, x=0.0, y=0.0):
        self.x = self._to_scalar(x)
        self.y = self._to_scalar(y)


class MockMultiranger:
    """Mock Multiranger sensor with landing pad detection support"""
    
    def __init__(self, obstacles, drone_position=(0, 0), drone_yaw=0):
        self.obstacles = obstacles
        self._drone_x = drone_position[0]
        self._drone_y = drone_position[1]
        self._drone_z = 0.5  # Flight height
        self._drone_yaw = drone_yaw
        self._max_range = 2.0
        self._min_range = 0.01
        
        # Add landing pads with known heights
        self.landing_pads = []  # [(x, y, width, pad_height)]
        
    def add_landing_pad(self, x, y, width=0.3, pad_height=0.1):
        """Add a landing pad at position with given dimensions"""
        self.landing_pads.append((x, y, width, pad_height))
    
    def update_pose(self, x, y, z, yaw):
        """Update drone position and orientation"""
        self._drone_x = x
        self._drone_y = y
        self._drone_z = z
        self._drone_yaw = yaw
    
    def _raycast(self, direction_angle, max_range=2.0):
        """Raycast to detect obstacles with correct geometry"""
        min_dist = max_range
        
        for angle_offset in [-0.087, 0.0, 0.087]:  # -5°, 0°, +5°
            scan_angle = direction_angle + angle_offset
            dx = np.cos(scan_angle)
            dy = np.sin(scan_angle)
            ray_dir = np.array([dx, dy])
            
            for obs_x, obs_y, radius in self.obstacles:
                # Vector from drone to obstacle center
                to_obs = np.array([obs_x - self._drone_x, obs_y - self._drone_y])
                
                # Project onto ray direction
                proj = np.dot(to_obs, ray_dir)
                
                # Only check obstacles in front of the sensor
                if proj > 0:
                    # Find closest point on ray to obstacle center
                    closest_point = proj * ray_dir
                    
                    # Perpendicular distance from ray to obstacle center
                    perp_vec = to_obs - closest_point
                    perp_dist = np.linalg.norm(perp_vec)
                    
                    # Check if ray intersects the circular obstacle
                    if perp_dist <= radius:
                        # Calculate distance to entry point of circle
                        # Use max to avoid negative values due to floating point errors
                        chord_half = np.sqrt(max(0, radius**2 - perp_dist**2))
                        hit_dist = proj - chord_half
                        
                        # Only count valid hits (positive distance, within range)
                        if self._min_range < hit_dist < max_range:
                            min_dist = min(min_dist, hit_dist)
        
        return min_dist if min_dist < max_range else None
    
    @property
    def front(self):
        return self._raycast(self._drone_yaw, self._max_range)
    
    @property
    def back(self):
        return self._raycast(self._drone_yaw + np.pi, self._max_range)
    
    @property
    def left(self):
        return self._raycast(self._drone_yaw + np.pi/2, self._max_range)
    
    @property
    def right(self):
        return self._raycast(self._drone_yaw - np.pi/2, self._max_range)
    
    @property
    def down(self):
        """Simulate z-range sensor - detects landing pad height changes"""
        # Check if over any landing pad
        for pad_x, pad_y, pad_width, pad_height in self.landing_pads:
            dist_to_pad = np.hypot(self._drone_x - pad_x, self._drone_y - pad_y)
            
            # Simulate realistic edge detection with smooth transition
            pad_radius = pad_width / 2
            
            if dist_to_pad < pad_radius:
                # Over the pad - return reduced height
                return self._drone_z - pad_height
            elif dist_to_pad < pad_radius + 0.02:
                # Near edge (within 2cm) - simulate edge transition
                # This creates detectable height changes at edges
                edge_factor = (dist_to_pad - pad_radius) / 0.02
                return self._drone_z - pad_height * (1 - edge_factor)
        
        # Not over pad - return normal flight height
        return self._drone_z




# ============================================================================
# UTILITY FUNCTIONS (matching fly_autonomous.py / utils.py)
# ============================================================================

def world_to_body(vx_w, vy_w, yaw_rad):
    """Convert world-frame velocity to body frame"""
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    return cy * vx_w + sy * vy_w, -sy * vx_w + cy * vy_w


def compute_path_velocity(current_xy, target_xy, speed):
    """Compute velocity vector toward target"""
    dx, dy = target_xy[0] - current_xy[0], target_xy[1] - current_xy[1]
    dist = np.hypot(dx, dy)
    return (0.0, 0.0) if dist < 1e-6 else (speed * dx / dist, speed * dy / dist)


def compute_avoidance_velocity(sensors, yaw_rad, danger_dist=0.6, gain=0.7):
    """Compute obstacle avoidance velocity"""
    angles = {'front': 0.0, 'right': -np.pi/2, 'back': np.pi, 'left': np.pi/2}
    rx, ry = 0.0, 0.0
    
    for d in ['front', 'back', 'left', 'right']:
        dist = sensors.get(d)
        if dist is None or dist > danger_dist:
            continue
        force = ((danger_dist - dist) / danger_dist) ** 2
        ang = yaw_rad + angles[d]
        rx -= force * np.cos(ang)
        ry -= force * np.sin(ang)
    
    return gain * (0.0 if abs(rx) < 0.05 else rx), gain * (0.0 if abs(ry) < 0.05 else ry)


def update_map(grid_map, sensors, x, y, yaw, safety_margin=0.05):
    """Update map with sensor readings"""
    dirs = {'front': 0, 'right': -np.pi/2, 'back': np.pi, 'left': np.pi/2}
    cells_updated = 0
    
    for d, dist in sensors.items():
        if dist is None or not (0.02 < dist <= 2.0):
            continue
        ang = yaw + dirs[d]
        ox, oy = x + dist * np.cos(ang), y + dist * np.sin(ang)
        
        for dx in np.linspace(-safety_margin, safety_margin, 3):
            for dy in np.linspace(-safety_margin, safety_margin, 3):
                if grid_map.set_value_from_xy_pos(ox + dx, oy + dy, 1.0):
                    cells_updated += 1
    
    return cells_updated


def generate_sweep_pattern(center, width, height, spacing=0.2):
    """Generate zigzag sweep pattern"""
    cx, cy = center
    waypoints = []
    
    x_min, x_max = cx - width / 2, cx + width / 2
    y_min, y_max = cy - height / 2, cy + height / 2
    
    y = y_min
    direction = 1
    
    while y <= y_max:
        waypoints.append((x_min if direction == 1 else x_max, y))
        waypoints.append((x_max if direction == 1 else x_min, y))
        y += spacing
        direction *= -1
    
    return waypoints


def generate_spiral_pattern(center, max_radius=0.5, spacing=0.15, points_per_loop=8):
    """Generate outward spiral pattern"""
    cx, cy = center
    waypoints = [(cx, cy)]
    
    radius = spacing
    while radius <= max_radius:
        for i in range(points_per_loop):
            angle = 2 * np.pi * i / points_per_loop
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            waypoints.append((x, y))
        radius += spacing
    
    return waypoints


# ============================================================================
# SIMULATION CLASS (matching AutonomousMission structure)
# ============================================================================

class SimulatedMission:
    """Simulates the exact autonomous mission algorithm"""
    
    def __init__(self, cruise_speed=0.3, flight_height=0.5, use_spiral=True):
        self.start = (0.6, 1.3)
        self.goal = (4.0, 2.0)
        
        self.current_x, self.current_y, self.current_z = self.start[0], self.start[1], flight_height
        self.current_yaw = 0.0
        
        self.cruise_speed = cruise_speed
        self.flight_height = flight_height
        self.use_spiral = use_spiral
        
        # Create map (same size as fly_autonomous.py)
        self.grid_map = GridMap(
            width=int(5.0/0.05), height=int(3.0/0.05),
            resolution=0.05, center_x=2.5, center_y=1.5
        )
        self.grid_map.occupy_boundaries(boundary_width=2)
        
        # Planning
        self.planner = None
        self.waypoints_x, self.waypoints_y = np.array([]), np.array([])
        self.waypoint_idx = 0
        
        # Environment with obstacles and landing pads
        self.true_obstacles = [
            (1.5, 1.8, 0.25),
            (2.5, 0.8, 0.25),
            (2.5, 2.0, 0.25),
            (3.5, 1.5, 0.25),
        ]
        
        self.sensor = MockMultiranger(self.true_obstacles, drone_position=self.start, drone_yaw=0)
        
        # Add landing pads at start and goal
        self.sensor.add_landing_pad(self.start[0], self.start[1], width=0.3, pad_height=0.1)
        self.sensor.add_landing_pad(self.goal[0], self.goal[1], width=0.3, pad_height=0.1)
        
        # Detection - optimized for 0.3m x 0.3m landing pad
        self.detector = LandingPadDetector()
        self.detector.configure_detection({
            'lag': 8,  # Larger lag for more stable detection
            'threshold': 2.0,  # Lower threshold for better sensitivity
            'influence': 0.3,  # Low influence to detect all edges
            'min_peak_height': 0.025,  # Lower to catch all edges (0.1m pad height)
            'min_edge_distance': 0.12  # Minimum 12cm between edges (for 30cm pad)
        })
        
        # Filters (matching fly_autonomous.py)
        self.target_filter = ExponentialFilter(alpha=0.6)
        self.velocity_filter = ExponentialFilter(alpha=0.6, initial_x=0.0, initial_y=0.0)
        
        # Visualization
        self.visualizer = None
        self.mission_active = False
        self.step_counter = 0
        
        # Trajectory tracking
        self.trajectory_x = [self.current_x]
        self.trajectory_y = [self.current_y]
    
    def update_sensors(self, debug=False):
        """Update sensor readings"""
        self.sensor.update_pose(self.current_x, self.current_y, self.current_z, self.current_yaw)
        self.sensor_readings = {
            'front': self.sensor.front,
            'back': self.sensor.back,
            'left': self.sensor.left,
            'right': self.sensor.right
        }
        
        # Debug: print sensor readings and calculated obstacle positions
        if debug and any(d is not None for d in self.sensor_readings.values()):
            print(f"\n  Drone at ({self.current_x:.2f}, {self.current_y:.2f}), yaw={np.degrees(self.current_yaw):.1f}°")
            dirs = {'front': 0, 'right': -np.pi/2, 'back': np.pi, 'left': np.pi/2}
            for direction, dist in self.sensor_readings.items():
                if dist is not None:
                    ang = self.current_yaw + dirs[direction]
                    obs_x = self.current_x + dist * np.cos(ang)
                    obs_y = self.current_y + dist * np.sin(ang)
                    print(f"    {direction:>6s}: {dist:.3f}m -> obstacle at ({obs_x:.2f}, {obs_y:.2f})")
        
        if self.visualizer and self.visualizer.is_setup:
            self.visualizer.update_sensors(self.sensor_readings)
    
    def process_detection(self):
        """Process landing pad detection"""
        height_m = self.sensor.down
        if height_m is not None:
            self.detector.process_height_measurement(height_m, (self.current_x, self.current_y))
    
    def plan_path(self, from_pos, to_pos, create_new_planner=True):
        """Plan path using D* Lite with dynamic replanning"""
        from_pos = (float(from_pos[0]), float(from_pos[1]))
        to_pos = (float(to_pos[0]), float(to_pos[1]))
        
        if create_new_planner or self.planner is None:
            print(f"  Creating new D* Lite planner from {from_pos} to {to_pos}")
            self.planner = DStarLite(self.grid_map, obstacle_threshold=0.5, show_animation=False)
            ok, self.waypoints_x, self.waypoints_y = self.planner.run(
                start=from_pos, goal=to_pos, simplify=True, smooth=False
            )
            if not ok or len(self.waypoints_x) == 0:
                raise RuntimeError('Planning failed')
            
            self.waypoints_x = np.array(self.waypoints_x)
            self.waypoints_y = np.array(self.waypoints_y)
        else:
            # Dynamic replanning
            print("  Using D* Lite dynamic replanning...")
            start_x_ind, start_y_ind = self.planner.world_to_grid(from_pos[0], from_pos[1])
            if start_x_ind is None:
                return False
            
            self.planner.start.x = int(start_x_ind)
            self.planner.start.y = int(start_y_ind)
            
            path = self.planner.compute_current_path()
            if path is None or len(path) == 0:
                return False
            
            self.waypoints_x, self.waypoints_y = self.planner.path_to_world_coords(path)
            self.waypoints_x, self.waypoints_y = self.planner.simplify_path(
                self.waypoints_x, self.waypoints_y
            )
            
            self.waypoints_x = np.array(self.waypoints_x)
            self.waypoints_y = np.array(self.waypoints_y)
        
        self.waypoint_idx = 0
        if len(self.waypoints_x) > 0:
            self.target_filter.reset(self.waypoints_x[0].item(), self.waypoints_y[0].item())
        else:
            self.target_filter.reset(to_pos[0], to_pos[1])
        return True
    
    def replan_if_blocked(self, cells_updated, current_target):
        """Replan if obstacles detected using D* Lite dynamic replanning"""
        if not self.mission_active or cells_updated == 0 or self.planner is None:
            return
        if len(self.waypoints_x) == 0:
            return
        
        replanned = self.planner.update_map()
        
        if replanned:
            print("  Extracting new path from D* Lite...")
            success = self.plan_path((self.current_x, self.current_y), current_target,
                                    create_new_planner=False)
            if not success:
                print("  Dynamic replanning failed, creating new planner")
                self.plan_path((self.current_x, self.current_y), current_target,
                              create_new_planner=True)
            
            if self.visualizer:
                self.visualizer.update_path(self.waypoints_x, self.waypoints_y)
                self.visualizer.mark_replan(self.current_x, self.current_y)
    
    def navigate_with_avoidance(self, target, slow_mode=False, use_absolute_height=False, skip_viz=False):
        """Navigate with obstacle avoidance (matching fly_autonomous.py logic)
        
        Args:
            skip_viz: If True, skip visualization updates for speed during search
        """
        self.process_detection()
        
        # Waypoint handling
        if self.waypoint_idx < len(self.waypoints_x):
            tx, ty = self.waypoints_x[self.waypoint_idx], self.waypoints_y[self.waypoint_idx]
            if np.hypot(self.current_x - tx, self.current_y - ty) < 0.05:
                self.waypoint_idx += 1
                if self.waypoint_idx < len(self.waypoints_x):
                    target = (self.waypoints_x[self.waypoint_idx], self.waypoints_y[self.waypoint_idx])
            else:
                target = (tx, ty)
        
        ftx, fty = self.target_filter.update(target[0], target[1])
        # Slower speed during search for better edge detection (0.1 m/s)
        cruise_speed = 0.1 if slow_mode else self.cruise_speed
        
        # Compute velocities
        vx_path_w, vy_path_w = compute_path_velocity(
            (self.current_x, self.current_y), (ftx, fty), cruise_speed
        )
        vx_avoid_w, vy_avoid_w = compute_avoidance_velocity(
            self.sensor_readings, self.current_yaw, danger_dist=0.5, gain=0.7
        )
        
        vx_w, vy_w = vx_path_w + vx_avoid_w, vy_path_w + vy_avoid_w
        
        # Speed limiting near obstacles
        dists = [d for d in self.sensor_readings.values() if d is not None]
        if dists and min(dists) < 0.4:
            speed = np.hypot(vx_w, vy_w)
            if speed > cruise_speed:
                vx_w, vy_w = cruise_speed * vx_w / speed, cruise_speed * vy_w / speed
        
        vx_w_f, vy_w_f = self.velocity_filter.update(vx_w, vy_w)
        
        # Update position (simple integration)
        dt = 0.01  # 10ms steps
        self.current_x += vx_w_f * dt
        self.current_y += vy_w_f * dt
        
        # Update yaw to face movement direction
        if abs(vx_w_f) > 0.01 or abs(vy_w_f) > 0.01:
            self.current_yaw = np.arctan2(vy_w_f, vx_w_f)
        
        self.trajectory_x.append(self.current_x)
        self.trajectory_y.append(self.current_y)
        
        # Update visualizer at reasonable frequency (every 5 steps for good balance)
        # Skip entirely if skip_viz=True (can be used for speed-critical sections)
        if not skip_viz and self.visualizer and self.step_counter % 5 == 0:
            self.visualizer.update_drone(self.current_x, self.current_y, self.current_yaw)
            self.visualizer.update_repulsion(vx_avoid_w, vy_avoid_w)
            avoiding = np.hypot(vx_avoid_w, vy_avoid_w) > 0.02
            self.visualizer.set_status(stuck=False, emergency=False, avoiding=avoiding)
            self.visualizer.render(force=True)
    
    def setup_visualizer(self):
        """Setup visualization with optimized settings for speed"""
        self.visualizer = DroneNavigationVisualizer(
            xlim=(0, 5), ylim=(0, 3), 
            figsize=(15, 8),  # Smaller figure for faster rendering
            animation_speed=0.0001  # Minimal pause for speed
        )
        self.visualizer.setup(start=self.start, goal=self.goal, 
                             true_obstacles=self.true_obstacles, grid_map=self.grid_map)
        plt.ion()
        # Disable toolbar for faster rendering
        plt.rcParams['toolbar'] = 'None'
    
    def clear_map_and_visualization(self):
        """Clear map and visualization for return journey"""
        print('\n  Clearing map and visualization...')
        
        # Reset grid map
        self.grid_map = GridMap(
            width=int(5.0/0.05), height=int(3.0/0.05),
            resolution=0.05, center_x=2.5, center_y=1.5
        )
        self.grid_map.occupy_boundaries(boundary_width=2)

        # Clear and reset visualization
        if self.visualizer:
            try:
                for artist in list(self.visualizer.ax.lines):
                    try:
                        artist.remove()
                    except:
                        pass
                for artist in list(self.visualizer.ax.collections):
                    try:
                        artist.remove()
                    except:
                        pass
                legend = self.visualizer.ax.get_legend()
                if legend:
                    try:
                        legend.remove()
                    except:
                        pass
            except Exception:
                pass
            
            self.visualizer.reset()
            self.visualizer.setup(start=self.start, goal=self.goal,
                                true_obstacles=self.true_obstacles, grid_map=self.grid_map)
            plt.draw()
            plt.pause(0.01)
    
    def search_and_land(self, search_center, is_goal=True):
        """Search for landing pad and land on it"""
        location_name = "goal" if is_goal else "start"
        print(f'\n📍 Searching for landing pad at {location_name}...')
        
        self.detector.peak_positions.clear()
        self.detector.calculated_center = None
        self.detector.start_detection(baseline_height=self.flight_height)
        pad_center = None
        

        print('  Using sweep search pattern (optimized for 0.3m x 0.3m pad)')
        # Dense sweep with 8cm spacing to ensure crossing all edges of 30cm pad
        sweep = generate_sweep_pattern(search_center, width=1.0, height=1.0, spacing=0.08)
        sweep_with_dist = [(wp, np.hypot(wp[0] - self.current_x, wp[1] - self.current_y)) 
                            for wp in sweep]
        sweep_with_dist.sort(key=lambda x: x[1])
        sweep = [wp for wp, _ in sweep_with_dist]
        
        # Visualize search pattern
        if self.visualizer:
            sweep_x, sweep_y = [wp[0] for wp in sweep], [wp[1] for wp in sweep]
            pattern_name = 'Spiral' if self.use_spiral else 'Sweep'
            self.visualizer.ax.scatter(sweep_x, sweep_y, c='red', s=50, marker='o',
                                      alpha=0.7, label=f'{pattern_name} ({location_name})', zorder=5)
            self.visualizer.render(force=True)
        
        # Execute search
        for i, sweep_target in enumerate(sweep):
            if self.visualizer:
                self.visualizer.goal = sweep_target
            
            print(f'  Sweep {i+1}/{len(sweep)}: ({sweep_target[0]:.2f}, {sweep_target[1]:.2f})')
            
            # Navigate to sweep waypoint with slow mode and absolute height
            # Update visualization during search (set skip_viz=False)
            # Use tighter tolerance for better sweep following (5cm instead of 10cm)
            dist = np.hypot(self.current_x - sweep_target[0], self.current_y - sweep_target[1])
            while dist > 0.05:
                self.update_sensors()
                self.process_detection()
                self.navigate_with_avoidance(sweep_target, slow_mode=True, 
                                            use_absolute_height=True, skip_viz=False)
                self.step_counter += 1
                dist = np.hypot(self.current_x - sweep_target[0], self.current_y - sweep_target[1])
            
            # Debug: Show detection status with height info
            num_edges = len(self.detector.peak_positions)
            height_m = self.sensor.down
            if num_edges > 0 or height_m < self.flight_height - 0.02:
                print(f'    Detected {num_edges} edge(s), height={height_m:.3f}m')
            
            # If first edge detected, switch to local search
            if len(self.detector.peak_positions) == 1:
                first_edge = self.detector.peak_positions[0]['position']
                print(f'  First edge found at ({first_edge[0]:.2f}, {first_edge[1]:.2f}), searching locally...')
                # Dense local search: 0.6m x 0.6m area with 6cm spacing
                # This ensures crossing the 0.3m pad from multiple angles
                sweep = generate_sweep_pattern(first_edge, width=0.6, height=0.6, spacing=0.06)
                
                if self.visualizer:
                    sweep_x, sweep_y = [wp[0] for wp in sweep], [wp[1] for wp in sweep]
                    self.visualizer.ax.scatter(sweep_x, sweep_y, c='orange', s=50, marker='o',
                                             alpha=0.7, label='Local Sweep', zorder=5)
                    self.visualizer.render(force=True)
            
            # Check if pad fully detected
            if len(self.detector.peak_positions) >= 2:
                center = self.detector.calculate_pad_center()
                if center and self.detector.center_confidence > 0.45:
                    pad_center = center
                    conf = self.detector.center_confidence
                    print(f'  ✅ Pad detected at ({pad_center[0]:.2f}, {pad_center[1]:.2f}), confidence={conf:.2f}')
                    break
        
        self.detector.stop_detection()
        
        # Print detection summary
        print(f'\n  Detection complete: {len(self.detector.peak_positions)} edge(s) found')
        if len(self.detector.peak_positions) > 0:
            for idx, edge in enumerate(self.detector.peak_positions):
                pos = edge['position']
                print(f'    Edge {idx+1}: ({pos[0]:.3f}, {pos[1]:.3f})')
        
        # Reset goal marker
        if self.visualizer:
            self.visualizer.goal = search_center
        
        # Visualize detection results
        if self.visualizer and len(self.detector.peak_positions) > 0:
            edges_x = [edge['position'][0] for edge in self.detector.peak_positions]
            edges_y = [edge['position'][1] for edge in self.detector.peak_positions]
            self.visualizer.ax.scatter(edges_x, edges_y, c='yellow', s=100, marker='o',
                                      edgecolors='orange', linewidths=2,
                                      label=f'Edges ({location_name})', zorder=10)
            
            if pad_center:
                self.visualizer.ax.scatter([pad_center[0]], [pad_center[1]], c='lime', s=200,
                                          marker='*', edgecolors='green', linewidths=2,
                                          label=f'Center ({location_name})', zorder=11)
                self.visualizer.ax.text(pad_center[0], pad_center[1] + 0.1,
                                       f'Center\n({pad_center[0]:.2f}, {pad_center[1]:.2f})',
                                       ha='center', fontsize=8, color='green', weight='bold')
            
            self.visualizer.ax.legend(loc='upper right', fontsize=8)
            self.visualizer.render(force=True)
            plt.pause(0.5)
        
        # Navigate to pad center if found
        if pad_center:
            print(f'  🎯 Navigating to pad center...')
            dist = np.hypot(self.current_x - pad_center[0], self.current_y - pad_center[1])
            while dist > 0.05:
                self.update_sensors()
                # Enable viz during final approach to pad center
                self.navigate_with_avoidance(pad_center, slow_mode=True, 
                                            use_absolute_height=True, skip_viz=False)
                self.step_counter += 1
                dist = np.hypot(self.current_x - pad_center[0], self.current_y - pad_center[1])
        else:
            print(f'  ⚠️  No pad detected, landing at current position')
        
        print(f'  🛬 Landing at {location_name}...')
        
        return pad_center
    
    def run(self, show_animation=True):
        """Run the complete autonomous mission"""
        print("="*70)
        print("SIMULATED AUTONOMOUS LANDING PAD DETECTION MISSION")
        print("="*70)
        
        if show_animation:
            self.setup_visualizer()
        
        try:
            # PHASE 1: Navigate to goal
            print("\n🚁 PHASE 1: Navigate to goal")
            print(f"  Start: {self.start}, Goal: {self.goal}")
            
            self.plan_path(self.start, self.goal)
            print(f"  Path planned with {len(self.waypoints_x)} waypoints")
            
            if self.visualizer:
                self.visualizer.update_path(self.waypoints_x, self.waypoints_y)
            
            self.mission_active = True
            print("  Navigating to goal...")
            
            # Debug: Print obstacle positions
            print("\n  Known obstacle positions:")
            for i, (ox, oy, r) in enumerate(self.true_obstacles):
                print(f"    Obstacle {i+1}: center=({ox:.2f}, {oy:.2f}), radius={r:.2f}")
            
            debug_steps = 0  # Count for debug output
            
            while self.mission_active:
                # Enable debug for first 5 sensor readings
                debug_mode = (debug_steps < 5)
                self.update_sensors(debug=debug_mode)
                if debug_mode:
                    debug_steps += 1
                
                # Update map every 2 steps (matching fly_autonomous.py)
                if self.step_counter % 5 == 0:
                    cells_updated = update_map(self.grid_map, self.sensor_readings,
                                              self.current_x, self.current_y, self.current_yaw,
                                              safety_margin=0.05)
                    if debug_steps <= 5 and cells_updated > 0:
                        print(f"    Map updated: {cells_updated} cells marked as occupied")
                    self.replan_if_blocked(cells_updated, self.goal)
                
                dist = np.hypot(self.current_x - self.goal[0], self.current_y - self.goal[1])
                if dist < 0.2:
                    break
                
                self.navigate_with_avoidance(self.goal)
                self.step_counter += 1
            
            print(f"  ✅ Arrived at goal (distance: {dist:.3f}m)")
            
            # PHASE 2: Search and land at goal
            print("\n🔍 PHASE 2: Search and land at goal")
            goal_pad_center = self.search_and_land(self.goal, is_goal=True)
            
            # PHASE 3: Clear and prepare for return
            print("\n🧹 PHASE 3: Prepare for return journey")
            self.clear_map_and_visualization()
            self.planner = None  # Force new planner for return
            
            # PHASE 4: Return to start
            print("\n🔙 PHASE 4: Return to start")
            
            # Use pad center if available, otherwise current position
            if goal_pad_center:
                start_pos = (np.clip(goal_pad_center[0], 0.5, 5.5),
                           np.clip(goal_pad_center[1], 0.5, 2.5))
            else:
                start_pos = (self.current_x, self.current_y)
            
            print(f"  Planning return from {start_pos} to {self.start}")
            self.plan_path(start_pos, self.start)
            
            if self.visualizer:
                self.visualizer.goal = self.start
                self.visualizer.update_path(self.waypoints_x, self.waypoints_y)
            
            self.mission_active = True
            print("  Returning to start...")
            
            while self.mission_active:
                self.update_sensors()
                
                if self.step_counter % 5 == 0:
                    cells_updated = update_map(self.grid_map, self.sensor_readings,
                                              self.current_x, self.current_y, self.current_yaw,
                                              safety_margin=0.05)
                    self.replan_if_blocked(cells_updated, self.start)
                
                dist = np.hypot(self.current_x - self.start[0], self.current_y - self.start[1])
                if dist < 0.2:
                    break
                
                self.navigate_with_avoidance(self.start)
                self.step_counter += 1
            
            print(f"  ✅ Arrived at start (distance: {dist:.3f}m)")
            
            # PHASE 5: Search and land at start
            print("\n🔍 PHASE 5: Search and land at start")
            start_pad_center = self.search_and_land(self.start, is_goal=False)
            
            print("\n" + "="*70)
            print("✅ MISSION COMPLETE!")
            print("="*70)
            print(f"\nTrajectory points: {len(self.trajectory_x)}")
            print(f"Total steps: {self.step_counter}")
            print(f"Final position: ({self.current_x:.2f}, {self.current_y:.2f})")
            
            # Final visualization
            if self.visualizer:
                # Plot full trajectory
                self.visualizer.ax.plot(self.trajectory_x, self.trajectory_y, 'b-',
                                       linewidth=1, alpha=0.5, label='Full Trajectory')
                self.visualizer.render(force=True)
                plt.ioff()
                plt.show()
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.mission_active = False


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Run simulated autonomous mission"""
    mission = SimulatedMission(
        cruise_speed=1.0,
        flight_height=0.5,
        use_spiral=False  # Set to False for sweep pattern
    )
    mission.run(show_animation=True)


if __name__ == "__main__":
    main()