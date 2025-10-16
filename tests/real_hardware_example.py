"""
Real Hardware Example: Autonomous Navigation with Obstacle Avoidance

This example demonstrates autonomous navigation on real Crazyflie hardware
using the SAME approach as simulation:
- D* Lite path planning with dynamic replanning
- Reactive obstacle avoidance with potential fields
- Real-time visualization with DroneNavigationVisualizer
- Low-level Commander for trajectory following

REQUIREMENTS:
1. Connected Crazyflie with Multiranger deck
2. Lighthouse/Loco positioning system for position estimates
3. cflib installed and configured

USAGE:
    python real_hardware_example.py
"""

import time
import numpy as np
from pathlib import Path
import sys


import cflib.crtp
from cflib.utils import uri_helper
from cflib.utils.multiranger import Multiranger

try:
    import cfpilot
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))


from cfpilot.controller import CrazyflieController
from cfpilot.visualization import DroneNavigationVisualizer
from cfpilot.mapping import GridMap
from cfpilot.planning.DStarLite.d_star_lite import DStarLite


# Import helper functions from simulation
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


def calculate_repulsion(sensors, yaw, danger_dist=0.7):
    """
    Calculate repulsion vector from obstacles using sensor data
    Compatible with Multiranger sensor format
    
    Args:
        sensors: Dict with sensor distances in meters (front/back/left/right)
        yaw: Current yaw in radians
        danger_dist: Distance threshold for repulsion
        
    Returns:
        (repulsion_x, repulsion_y, emergency_stop)
    """
    angles = {'front': 0.0, 'right': -np.pi/2, 'back': np.pi, 'left': np.pi/2}
    
    repulsion_x, repulsion_y = 0.0, 0.0
    emergency = False
    
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


def blend_target_and_avoidance(target, current, repulsion, weight=0.7):
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
    to_target = (target[0] - current[0], target[1] - current[1])
    target_dist = np.hypot(*to_target)
    
    if target_dist < 0.03:
        return target
    
    to_target = (to_target[0] / target_dist, to_target[1] / target_dist)
    
    blend_x = (1 - weight) * to_target[0] + weight * repulsion[0]
    blend_y = (1 - weight) * to_target[1] + weight * repulsion[1]
    
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
    
    Returns:
        Number of cells updated
    """
    directions = {'front': 0, 'right': -np.pi/2, 'back': np.pi, 'left': np.pi/2}
    
    cells_updated = 0
    for direction, distance in sensors.items():
        if distance is None or distance > 2.0 or distance < 0.05:
            continue
        
        angle = yaw + directions[direction]
        obs_x = x + distance * np.cos(angle)
        obs_y = y + distance * np.sin(angle)
        
        # Mark obstacle AND surrounding cells for safety
        for dx in np.linspace(-safety_margin, safety_margin, 5):
            for dy in np.linspace(-safety_margin, safety_margin, 5):
                grid_map.set_value_from_xy_pos(obs_x + dx, obs_y + dy, 1.0)
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
    planner = DStarLite(grid_map, obstacle_threshold=0.5, show_animation=False)
    
    path_found, pathx, pathy = planner.run(
        start=start_pos, 
        goal=goal_pos,
        simplify=True,
        smooth=True
    )
    
    if not path_found or len(pathx) == 0:
        return False, [], []
    
    return True, list(pathx), list(pathy)


class RealHardwareNavigationDemo:
    """
    Real-time autonomous navigation with Crazyflie hardware
    
    This class demonstrates the SAME navigation approach as simulation:
    1. D* Lite path planning with dynamic replanning
    2. Hybrid navigation (global + local reactive avoidance)
    3. Real-time sensor-based obstacle discovery
    4. Position filtering for smooth trajectories
    5. Low-level Commander for setpoint control
    """
    
    def __init__(self, cruise_speed=0.08, goal_speed=0.03, target_yaw=0.0, filter_alpha=0.6):
        """
        Initialize demo
        
        Args:
            cruise_speed: Speed when far from goal (m/step), default 0.08
            goal_speed: Speed when near goal (m/step), default 0.03
            target_yaw: Target yaw angle to maintain in degrees, default 0.0
            filter_alpha: Position filter smoothing (0-1), default 0.6
        """
        
        
        # Controller (uses existing implementation from controller.py)
        self.controller = CrazyflieController()
        self.uri = self.controller.config['connection']['uri']

        # Navigation state
        self.start = (0.0, 0.0)
        self.goal = (2.0, 0.0)
        self.flight_height = 0.5  # meters
        
        # Navigation parameters (configurable)
        self.cruise_speed = cruise_speed  # Speed when far from goal
        self.goal_speed = goal_speed      # Speed when near goal
        self.target_yaw = target_yaw      # Target yaw angle (degrees)
        self.yaw_tolerance = 5.0          # Acceptable yaw error (degrees)
        self.goal_distance_threshold = 0.3  # Distance to switch to goal_speed
        
        # Current state (updated via controller callbacks)
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.current_yaw = 0.0  # radians
        
        # Visualizer
        self.visualizer = None
        
        # Grid map for obstacle discovery
        self.grid_map = GridMap(
            width=80, height=80,
            resolution=0.05,
            center_x=0.0, center_y=0.0
        )
        self.grid_map.occupy_boundaries(boundary_width=2)
        
        # Sensor data
        self.sensor_readings = {}
        self.multiranger = None
        
        # Path planning
        self.waypoints_x = []
        self.waypoints_y = []
        self.waypoint_idx = 0
        
        # Position filter for smooth commands
        self.pos_filter = PositionFilter(alpha=filter_alpha)
        
        # Control state
        self.replans = 0
        self.replan_cooldown = 0
        self.stuck_counter = 0
        self.last_position = (0.0, 0.0)
        self.mission_active = False
        
        print("="*60)
        print("Real Hardware Autonomous Navigation")
        print("Using CrazyflieController + Low-level Commander")
        print("="*60)
        print(f"\nNavigation Parameters:")
        print(f"  Cruise speed: {self.cruise_speed} m/step")
        print(f"  Goal speed: {self.goal_speed} m/step")
        print(f"  Target yaw: {self.target_yaw}°")
        print(f"  Filter alpha: {filter_alpha}")
    
    def position_callback(self, timestamp, data, logconf_name):
        """
        Callback for position data from controller
        
        This is called automatically when new position data arrives.
        """
        # Update current position
        self.current_x = data.get('stateEstimate.x', 0.0)
        self.current_y = data.get('stateEstimate.y', 0.0)
        self.current_z = data.get('stateEstimate.z', 0.0)
        yaw_deg = data.get('stabilizer.yaw', 0.0)
        self.current_yaw = np.deg2rad(yaw_deg)
        
        # Debug: Print position occasionally during navigation
        # if hasattr(self, 'mission_active') and self.mission_active:
        #     if int(timestamp) % 1000 == 0:  # Every second
        #         print(f"   [Position] ({self.current_x:.2f}, {self.current_y:.2f}, {self.current_z:.2f})")
        
        # Update visualizer if active
        if self.visualizer and self.visualizer.is_setup:
            self.visualizer.update_drone(self.current_x, self.current_y, self.current_yaw)
    
    def update_sensors(self):
        """Update sensor readings from Multiranger"""
        if not self.multiranger:
            return
        
        # Read sensors using EXACT same API as simulation MockMultiranger
        self.sensor_readings = {
            'front': self.multiranger.front,
            'back': self.multiranger.back,
            'left': self.multiranger.left,
            'right': self.multiranger.right
        }
        
        # Update visualizer
        if self.visualizer and self.visualizer.is_setup:
            self.visualizer.update_sensors(self.sensor_readings)
    
    def run_autonomous_mission(self):
        """
        Run autonomous navigation mission
        
        This follows the SAME logic as simulation:
        1. Take off
        2. Plan initial path
        3. Navigate with hybrid control (D* Lite + potential fields)
        4. Replan when obstacles detected
        5. Land at goal
        """
        print(f"\n📡 Connecting to Crazyflie at {self.uri}...")
        
        # Connect using controller
        self.controller.connect(self.uri, x=self.start[0], y=self.start[1], z=0.0, yaw=0.0)
        
        if not self.controller.wait_for_connection(timeout=10.0):
            print("❌ Connection failed!")
            return
        
        print("✅ Connected!")
        
        try:
            # Add position callback
            self.controller.add_data_callback(self.position_callback)
            
            # Initialize Multiranger
            print("🔍 Initializing Multiranger sensors...")
            self.multiranger = Multiranger(self.controller.cf, rate_ms=100)
            self.multiranger.start()
            time.sleep(2.0)  # Wait for sensors to stabilize
            
            # Setup visualizer
            print("\n📊 Setting up visualizer...")
            self.visualizer = DroneNavigationVisualizer(
                xlim=(-2, 2),
                ylim=(-2, 2),
                figsize=(12, 9),
                animation_speed=0.05
            )
            self.visualizer.setup(
                start=self.start,
                goal=self.goal,
                grid_map=self.grid_map
            )
            
            # Take off
            print("\n🚁 Taking off...")
            self._takeoff()
            
            # Plan initial path
            print("\n🛤️  Planning initial path...")
            path_found, self.waypoints_x, self.waypoints_y = plan_path_with_dstar(
                self.grid_map, self.start, self.goal
            )
            
            if not path_found:
                print("❌ Could not find initial path!")
                self._land()
                return
            
            print(f"✅ Initial path planned: {len(self.waypoints_x)} waypoints")
            
            if self.visualizer:
                self.visualizer.update_path(self.waypoints_x, self.waypoints_y)
            
            # Navigate to goal
            print("\n🚁 Starting autonomous navigation...")
            print("   • Global path planning with D* Lite")
            print("   • Local reactive avoidance with potential fields")
            print("   • Dynamic replanning when obstacles detected")
            
            self.mission_active = True
            self.waypoint_idx = 0
            
            control_rate = 20  # Hz
            dt = 1.0 / control_rate
            max_iterations = 2000  # 100 seconds max
            
            # Wait a moment for position to update after takeoff
            time.sleep(0.5)
            
            # Initialize filter and tracking with current position
            self.pos_filter.reset(self.current_x, self.current_y)
            self.last_position = (self.current_x, self.current_y)
            self.stuck_counter = 0
            
            print(f"   Starting from: ({self.current_x:.2f}, {self.current_y:.2f})")
            print(f"   Goal: ({self.goal[0]:.2f}, {self.goal[1]:.2f})")
            
            for iteration in range(max_iterations):
                if not self.mission_active:
                    break
                
                # Check if reached goal
                dist_to_goal = np.hypot(self.goal[0] - self.current_x, self.goal[1] - self.current_y)
                if dist_to_goal < 0.05:
                    print(f"✅ Goal reached at iteration {iteration}!")
                    break
                
                # Update sensors
                self.update_sensors()
                
                # Calculate repulsion forces
                rep_x, rep_y, emergency = calculate_repulsion(
                    self.sensor_readings, self.current_yaw, danger_dist=0.6
                )
                
                # Check if stuck (only after some iterations to allow position updates)
                if iteration > 20:  # Allow initial settling
                    dist_moved = np.hypot(self.current_x - self.last_position[0], 
                                         self.current_y - self.last_position[1])
                    if dist_moved < 0.005:
                        self.stuck_counter += 1
                    else:
                        self.stuck_counter = 0
                self.last_position = (self.current_x, self.current_y)
                
                # Handle stuck situation
                if self.stuck_counter > 40:  # Stuck for 40 steps (2 seconds)
                    print(f"   ⚠️  Drone stuck at ({self.current_x:.2f}, {self.current_y:.2f})! Attempting escape...")
                    print(f"       Current waypoint: ({self.waypoints_x[self.waypoint_idx]:.2f}, {self.waypoints_y[self.waypoint_idx]:.2f})")
                    
                    # Skip ahead in waypoints
                    self.waypoint_idx = min(self.waypoint_idx + 3, len(self.waypoints_x) - 1)
                    self.stuck_counter = 0
                    
                    if self.replan_cooldown == 0:
                        print(f"   🔄 Escape replan...")
                        path_found, self.waypoints_x, self.waypoints_y = plan_path_with_dstar(
                            self.grid_map, (self.current_x, self.current_y), self.goal
                        )
                        if path_found and len(self.waypoints_x) > 0:
                            self.waypoint_idx = 0
                            self.replans += 1
                            self.replan_cooldown = 30
                            if self.visualizer:
                                self.visualizer.update_path(self.waypoints_x, self.waypoints_y)
                                self.visualizer.mark_replan()
                        else:
                            print("   ❌ Escape replan failed!")
                    continue
                
                # Decrement cooldown
                if self.replan_cooldown > 0:
                    self.replan_cooldown -= 1
                
                # Update map periodically
                if iteration % 3 == 0:
                    cells_updated = update_map(self.grid_map, self.sensor_readings,
                                              self.current_x, self.current_y, self.current_yaw,
                                              safety_margin=0.15)
                else:
                    cells_updated = 0
                
                # Check if path is blocked and replan
                if cells_updated > 0 and self.waypoint_idx < len(self.waypoints_x) and self.replan_cooldown == 0:
                    blocked = False
                    for i in range(self.waypoint_idx, min(self.waypoint_idx + 5, len(self.waypoints_x))):
                        wx, wy = self.waypoints_x[i], self.waypoints_y[i]
                        idx_x, idx_y = self.grid_map.get_xy_index_from_xy_pos(wx, wy)
                        if idx_x is not None:
                            val = self.grid_map.get_value_from_xy_index(idx_x, idx_y)
                            if val is not None and val > 0.5:
                                blocked = True
                                break
                    
                    if blocked:
                        print(f"   ⚠️  Obstacle detected! Replanning...")
                        path_found, self.waypoints_x, self.waypoints_y = plan_path_with_dstar(
                            self.grid_map, (self.current_x, self.current_y), self.goal
                        )
                        
                        if path_found and len(self.waypoints_x) > 0:
                            self.waypoint_idx = 0
                            self.replans += 1
                            self.replan_cooldown = 25
                            print(f"   ✅ Replanned with {len(self.waypoints_x)} waypoints")
                            
                            if self.visualizer:
                                self.visualizer.update_path(self.waypoints_x, self.waypoints_y)
                                self.visualizer.mark_replan()
                
                # Navigate to next waypoint
                if self.waypoint_idx < len(self.waypoints_x):
                    target_x = self.waypoints_x[self.waypoint_idx]
                    target_y = self.waypoints_y[self.waypoint_idx]
                    
                    # Check if reached waypoint
                    dist_to_waypoint = np.hypot(target_x - self.current_x, target_y - self.current_y)
                    if dist_to_waypoint < 0.12:
                        self.waypoint_idx += 1
                        self.stuck_counter = 0
                        if self.waypoint_idx % 5 == 0:
                            print(f"   ✓ Waypoint {self.waypoint_idx}/{len(self.waypoints_x)} reached")
                        continue
                    
                    # Hybrid navigation: blend global path with local avoidance
                    repulsion_magnitude = np.hypot(rep_x, rep_y)
                    
                    if repulsion_magnitude > 0.08:
                        # Active avoidance
                        avoidance_weight = min(0.75, repulsion_magnitude * 1.8)
                        next_x, next_y = blend_target_and_avoidance(
                            target=(target_x, target_y),
                            current=(self.current_x, self.current_y),
                            repulsion=(rep_x, rep_y),
                            weight=avoidance_weight
                        )
                    else:
                        # Normal waypoint following with adaptive speed
                        dx = target_x - self.current_x
                        dy = target_y - self.current_y
                        dist = np.hypot(dx, dy)
                        
                        # Adaptive step size based on distance to goal
                        if dist_to_goal < self.goal_distance_threshold:
                            step_size = self.goal_speed  # Slower near goal
                        else:
                            step_size = self.cruise_speed  # Faster when far from goal
                        
                        if dist > 0.001:  # Avoid division by zero
                            next_x = self.current_x + step_size * dx / dist
                            next_y = self.current_y + step_size * dy / dist
                        else:
                            next_x, next_y = self.current_x, self.current_y
                    
                    # Apply position filter
                    filtered_x, filtered_y = self.pos_filter.update(next_x, next_y)
                    
                    # Yaw control: maintain target yaw with smooth adjustment
                    yaw_error = self.target_yaw - np.rad2deg(self.current_yaw)
                    # Normalize to [-180, 180]
                    while yaw_error > 180:
                        yaw_error -= 360
                    while yaw_error < -180:
                        yaw_error += 360
                    
                    # Apply yaw correction if needed
                    if abs(yaw_error) > self.yaw_tolerance:
                        # Smooth yaw adjustment (max 5 degrees per step)
                        yaw_correction = np.clip(yaw_error * 0.3, -5.0, 5.0)
                        command_yaw = np.rad2deg(self.current_yaw) + yaw_correction
                    else:
                        # Hold target yaw
                        command_yaw = self.target_yaw
                    
                    # Send position setpoint with controlled yaw
                    self.controller.cf.commander.send_position_setpoint(
                        filtered_x, filtered_y, self.flight_height, command_yaw
                    )
                else:
                    # All waypoints reached, hold position with target yaw
                    self.controller.cf.commander.send_position_setpoint(
                        self.current_x, self.current_y, self.flight_height, self.target_yaw
                    )
                
                # Update visualizer
                if self.visualizer and iteration % 5 == 0:
                    self.visualizer.update_repulsion(rep_x, rep_y)
                    self.visualizer.set_status(
                        stuck=(self.stuck_counter > 10),
                        emergency=emergency,
                        avoiding=(np.hypot(rep_x, rep_y) > 0.1)
                    )
                    self.visualizer.render()
                
                # Debug output every 2 seconds
                if iteration % 40 == 0:
                    print(f"   [Debug] Pos: ({self.current_x:.2f}, {self.current_y:.2f}), "
                          f"Waypoint: {self.waypoint_idx}/{len(self.waypoints_x)}, "
                          f"Dist to goal: {dist_to_goal:.2f}m, "
                          f"Stuck: {self.stuck_counter}")
                
                time.sleep(dt)
            
            # Land
            print("\n⬇️  Landing...")
            self._land()
            
            print(f"\n📊 Results:")
            print(f"   Replanning events: {self.replans}")
            print(f"   Final position: ({self.current_x:.2f}, {self.current_y:.2f})")
            print(f"   Distance to goal: {dist_to_goal:.2f}m")
            
        finally:
            # Cleanup
            print("\n🛑 Stopping mission...")
            self.mission_active = False
            
            # Stop motors immediately
            if self.controller.cf and self.controller.is_connected:
                print("   Stopping motors...")
                try:
                    self.controller.cf.commander.send_stop_setpoint()
                    time.sleep(0.1)
                except Exception as e:
                    print(f"   Warning: Could not stop motors: {e}")
            
            # Stop sensors
            if self.multiranger:
                print("   Stopping Multiranger...")
                try:
                    self.multiranger.stop()
                except Exception as e:
                    print(f"   Warning: Could not stop Multiranger: {e}")
            
            # Finalize visualizer
            if self.visualizer:
                print("   Finalizing visualization...")
                try:
                    self.visualizer.finalize('real_hardware_result.png')
                except Exception as e:
                    print(f"   Warning: Could not finalize visualizer: {e}")
            
            # Disconnect controller
            print("   Disconnecting...")
            try:
                self.controller.disconnect()
            except Exception as e:
                print(f"   Warning: Could not disconnect: {e}")
            
            print("✅ Cleanup complete!")
    
    def _takeoff(self):
        """Take off to flight height"""
        # Unlock thrust protection
        self.controller.cf.commander.send_setpoint(0, 0, 0, 0)
        time.sleep(0.1)
        
        # Gradual takeoff
        takeoff_time = 3.0  # seconds
        control_rate = 20  # Hz
        dt = 1.0 / control_rate
        steps = int(takeoff_time * control_rate)
        
        for i in range(steps):
            z = (i / steps) * self.flight_height
            # Use target yaw throughout takeoff
            self.controller.cf.commander.send_position_setpoint(
                self.start[0], self.start[1], z, self.target_yaw
            )
            time.sleep(dt)
        
        # Hold position at target yaw
        for _ in range(20):
            self.controller.cf.commander.send_position_setpoint(
                self.current_x, self.current_y, self.flight_height, self.target_yaw
            )
            time.sleep(dt)
        
        print(f"✅ Takeoff complete at height {self.flight_height}m, yaw={self.target_yaw}°")
    
    def _land(self):
        """Land at current position"""
        landing_time = 2.0  # seconds
        control_rate = 20  # Hz
        dt = 1.0 / control_rate
        steps = int(landing_time * control_rate)
        
        start_z = self.current_z
        
        for i in range(steps):
            z = start_z * (1 - i / steps)
            # Maintain target yaw during landing
            self.controller.cf.commander.send_position_setpoint(
                self.current_x, self.current_y, max(0.0, z), self.target_yaw
            )
            time.sleep(dt)
        
        # Stop motors
        self.controller.cf.commander.send_stop_setpoint()
        time.sleep(0.1)
        self.controller.cf.commander.send_notify_setpoint_stop()
        time.sleep(0.)
        print("✅ Landing complete")


def main():
    """Main entry point"""
    # Initialize cflib drivers
    cflib.crtp.init_drivers()

    # Create demo with configurable parameters
    # Adjust these values as needed:
    demo = RealHardwareNavigationDemo(
        cruise_speed=0.12,   # Speed when far from goal (m/step) - increase for faster movement
        goal_speed=0.03,     # Speed when near goal (m/step) - keep slow for precision
        target_yaw=0.0,      # Target yaw angle to maintain (degrees) - 0=forward
        filter_alpha=0.6     # Position filter smoothing (0-1, higher=less smooth, faster response)
    )
    
    try:
        demo.run_autonomous_mission()
    except KeyboardInterrupt:
        print("\n\n⚠️  Mission interrupted by user (Ctrl+C)")
        demo.mission_active = False
        
        # Emergency stop
        if demo.controller.cf and demo.controller.is_connected:
            try:
                demo.controller.cf.commander.send_stop_setpoint()
                time.sleep(0.1)
            except:
                pass
        
        # Stop sensors
        if demo.multiranger:
            try:
                demo.multiranger.stop()
            except:
                pass
        
        # Disconnect
        try:
            demo.controller.disconnect()
        except:
            pass
        
        print("✅ Emergency stop complete")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        demo.mission_active = False
        
        # Emergency cleanup
        if demo.controller.cf and demo.controller.is_connected:
            try:
                demo.controller.cf.commander.send_stop_setpoint()
            except:
                pass
        
        if demo.multiranger:
            try:
                demo.multiranger.stop()
            except:
                pass
        
        try:
            demo.controller.disconnect()
        except:
            pass


if __name__ == "__main__":
    main()

