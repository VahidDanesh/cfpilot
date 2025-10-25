"""
Real Hardware Hover Navigation Example (Hover mode)

Round-trip mission with obstacle avoidance:
- Takeoff -> Navigate to goal -> Land (motors off) -> Takeoff -> Return to start -> Land (motors off)

Features:
- D* Lite path planning with dynamic replanning
- Reactive obstacle avoidance with sensor fusion
- Velocity and position filtering for smooth flight
- Hysteresis-based avoidance to prevent oscillations
"""

import time
import threading
import numpy as np
from pathlib import Path
import sys

import cflib.crtp
from cflib.utils.multiranger import Multiranger

try:
    import cfpilot
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from cfpilot.controller import CrazyflieController
from cfpilot.visualization import DroneNavigationVisualizer
from cfpilot.mapping import GridMap
from cfpilot.planning.DStarLite.d_star_lite import DStarLite


class ExponentialFilter:
    """Exponential smoothing filter for 2D values (position, velocity, etc.)."""
    def __init__(self, alpha=0.5, initial_x=None, initial_y=None):
        self.alpha = alpha
        self.x = self._to_scalar(initial_x)
        self.y = self._to_scalar(initial_y)

    @staticmethod
    def _to_scalar(val):
        # Robustly convert any scalar or 0-dim/1-element numpy array to float
        if val is None:
            return None
        arr = np.asarray(val)
        if arr.shape == ():
            return float(arr)
        elif arr.size == 1:
            return float(arr.ravel()[0])
        else:
            raise ValueError("ExponentialFilter expects a scalar or an array of shape () or size 1.")
    
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

def world_to_body(vx_w, vy_w, yaw_rad):
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    vx_b =  cy * vx_w + sy * vy_w  # body x forward
    vy_b = -sy * vx_w + cy * vy_w  # body y left
    return vx_b, vy_b


def compute_path_velocity(current_xy, target_xy, speed):
    dx = target_xy[0] - current_xy[0]
    dy = target_xy[1] - current_xy[1]
    dist = np.hypot(dx, dy)
    if dist < 1e-6:
        return 0.0, 0.0
    return speed * dx / dist, speed * dy / dist


def compute_avoidance_velocity(sensors, yaw_rad, danger_dist=0.6, gain=0.3, hysteresis=0.05):
    """
    Compute avoidance velocity with hysteresis to prevent toggling.
    
    Args:
        hysteresis: Additional margin before disabling avoidance (prevents on/off oscillation)
    """
    angles = {'front': 0.0, 'right': -np.pi/2, 'back': np.pi, 'left': np.pi/2}
    rx, ry = 0.0, 0.0
    effective_danger_dist = danger_dist + hysteresis  # Extended zone for smoother transitions
    
    for d in ['front', 'back', 'left', 'right']:
        dist = sensors.get(d)
        if dist is None or dist > effective_danger_dist:
            continue
        # Smooth force falloff
        force = (effective_danger_dist - dist) / effective_danger_dist
        force = force * force
        ang = yaw_rad + angles[d]
        rx -= force * np.cos(ang)
        ry -= force * np.sin(ang)
    
    # Larger deadzone to prevent micro-movements
    if abs(rx) < 0.05:
        rx = 0.0
    if abs(ry) < 0.05:
        ry = 0.0
    return gain * rx, gain * ry


def update_map(grid_map, sensors, x, y, yaw, safety_margin=0.15, drone_clearance=0.05):
    """
    Update map with sensor readings.
    
    First clears drone's position to prevent self-blocking, then adds detected obstacles.
    
    Args:
        safety_margin: Margin around detected obstacles
        drone_clearance: Radius around drone to keep clear (prevents self-blocking)
    """
    # First, ensure drone's current position and immediate area is marked as free
    # This prevents the drone from marking itself as an obstacle
    for dx in np.linspace(-drone_clearance, drone_clearance, 4):
        for dy in np.linspace(-drone_clearance, drone_clearance, 4):
            if np.hypot(dx, dy) <= drone_clearance:  # Circular clearance
                grid_map.set_value_from_xy_pos(x + dx, y + dy, 0.0)
    
    # Now update obstacles from sensor readings
    dirs = {'front': 0, 'right': -np.pi/2, 'back': np.pi, 'left': np.pi/2}
    cells_updated = 0
    
    for d, dist in sensors.items():
        if dist is None or dist > 2.0 or dist < 0.02:
            continue
        
        ang = yaw + dirs[d]
        ox = x + dist * np.cos(ang)
        oy = y + dist * np.sin(ang)
        
        # Add obstacle with safety margin
        for dx in np.linspace(-safety_margin, safety_margin, 5):
            for dy in np.linspace(-safety_margin, safety_margin, 5):
                obs_x = ox + dx
                obs_y = oy + dy
                ok = grid_map.set_value_from_xy_pos(obs_x, obs_y, 1.0)
                if ok:
                    cells_updated += 1
    
    return cells_updated


class HoverTimer:
    """Fixed-rate sender for hover setpoints using self.hover state."""

    def __init__(self, cf, rate_hz=50.0, hover_state=None, lock=None, max_speed=None):
        self.cf = cf
        self.period = 1.0 / rate_hz
        self.hover = hover_state
        self._lock = lock
        self._running = False
        self._thread = None
        self._max_speed = max_speed  # m/s (None means no clamp)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            self.cf.commander.send_stop_setpoint()
            time.sleep(0.1)
            self.cf.commander.send_notify_setpoint_stop()
        except Exception:
            pass

    def _run(self):
        next_t = time.perf_counter()
        while self._running:
            # Read hover state with thread safety
            with self._lock if self._lock else NoOpContext():
                vx = self.hover.get('x', 0.0)
                vy = self.hover.get('y', 0.0)
                yaw = self.hover.get('yaw', 0.0)
                h = self.hover.get('height', 0.0)
            
            # Clamp speed if configured
            if self._max_speed is not None:
                speed = np.hypot(vx, vy)
                if speed > 1e-6 and speed > self._max_speed:
                    scale = self._max_speed / speed
                    vx *= scale
                    vy *= scale
            
            # Send command
            try:
                self.cf.commander.send_hover_setpoint(vx, vy, yaw, h)
            except Exception:
                pass
            
            # Maintain fixed rate
            next_t += self.period
            sleep_t = next_t - time.perf_counter()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                next_t = time.perf_counter()


class NoOpContext:
    """No-op context manager for when lock is None."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class RealHardwareHoverNav:
    # Constants
    WAYPOINT_ACCEPTANCE_RADIUS = 0.05  # m
    TARGET_ARRIVAL_THRESHOLD = 0.5    # m - threshold to start holding position at goal/start
    MIN_HEIGHT = 0.2                   # m - minimum safe height
    MAP_UPDATE_INTERVAL = 2            # steps between map updates (~10 Hz at 50 Hz loop)
    
    def __init__(self,
                 cruise_speed=0.4,     # m/s along path
                 flight_height=0.5,
                 control_rate_hz=50.0):
        self.controller = CrazyflieController()
        self.uri = self.controller.config['connection']['uri']

        self.start = (0.6, 1.3)
        self.goal = (4.5, 2.0)
        self.current_x = self.start[0]
        self.current_y = self.start[1]
        self.current_z = 0.1
        self.current_yaw = 0.0  # radians

        # Mission state tracking
        # States: GOING_TO_GOAL, AT_GOAL_HOLDING, AT_GOAL_LANDED, RETURNING_TO_START, AT_START_HOLDING, AT_START_LANDED, COMPLETED
        self.mission_phase = 'GOING_TO_GOAL'
        self.current_target = self.goal
        self.holding_position = False  # True when holding at goal/start

        self.cruise_speed = cruise_speed
        self.flight_height = flight_height
        self.control_rate_hz = control_rate_hz
        self.safety_margin = 0.05              # m - extra obstacle margin
        self.near_obstacle_dist = 0.4          # m - start repulsion distance
        self.max_speed_near_obstacle = 0.2    # m/s - max speed near obstacles

        self.grid_map = GridMap(
            width=int(5.4 / 0.05), height=int(2.6 / 0.05),
            resolution=0.05, center_x=2.7, center_y=1.3
        )
        self.grid_map.occupy_boundaries(boundary_width=2)

        self.waypoints_x = np.array([])
        self.waypoints_y = np.array([])
        self.waypoint_idx = 0

        self.multiranger = None
        self.sensor_readings = {}
        
        # D* Lite planner for dynamic replanning
        self.planner = None

        self.visualizer = None
        self.position_text = None  # Initialize for position display

        # Hover state and timer (names like multiranger_pointcloud.py)
        self.hover = {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'height': self.flight_height}
        self._hover_lock = threading.Lock()
        self.hover_timer = None

        self.mission_active = False
        self.target_filter = ExponentialFilter(alpha=0.6)  # Smooth target tracking
        self.velocity_filter = ExponentialFilter(alpha=0.7, initial_x=0.0, initial_y=0.0)  # Smooth velocity commands
        
        # Keyboard control state
        self.keyboard_vx = 0.0  # body frame forward/backward
        self.keyboard_vy = 0.0  # body frame left/right
        self.keyboard_height_delta = 0.0
        self.keyboard_velocity = 0.2  # m/s for keyboard control
        self.keyboard_height_rate = 0.1  # m/s for height change
        self._keyboard_lock = threading.Lock()
        
        # Command flags from keyboard
        self.landing_requested = False
        self.takeoff_requested = False

    # ---- Callbacks / IO ----

    def position_callback(self, timestamp, data, logconf_name):
        self.current_x = data.get('stateEstimate.x', 0.0)
        self.current_y = data.get('stateEstimate.y', 0.0)
        self.current_z = data.get('stateEstimate.z', 0.0)
        yaw_deg = data.get('stabilizer.yaw', 0.0)
        self.current_yaw = np.deg2rad(yaw_deg)
        if self.visualizer and self.visualizer.is_setup:
            self.visualizer.update_drone(self.current_x, self.current_y, self.current_yaw)

    def update_hover(self, vx=None, vy=None, yaw=None, height=None):
        """Thread-safe update of hover state."""
        with self._hover_lock:
            if vx is not None:
                self.hover['x'] = vx
            if vy is not None:
                self.hover['y'] = vy
            if yaw is not None:
                self.hover['yaw'] = yaw
            if height is not None:
                self.hover['height'] = height
    
    def on_key_press(self, event):
        """Handle keyboard press events for manual control."""
        with self._keyboard_lock:
            if event.key == 'up':
                self.keyboard_vx = self.keyboard_velocity
                print('Key: UP - Moving forward')
            elif event.key == 'down':
                self.keyboard_vx = -self.keyboard_velocity
                print('Key: DOWN - Moving backward')
            elif event.key == 'left':
                self.keyboard_vy = self.keyboard_velocity
                print('Key: LEFT - Moving left')
            elif event.key == 'right':
                self.keyboard_vy = -self.keyboard_velocity
                print('Key: RIGHT - Moving right')
            elif event.key == 'q':
                self.keyboard_height_delta = self.keyboard_height_rate
                print('Key: Q - Increasing height')
            elif event.key == 'a':
                self.keyboard_height_delta = -self.keyboard_height_rate
                print('Key: A - Decreasing height')
            elif event.key == 'l':
                print('Key: L - Landing requested')
                self.landing_requested = True
            elif event.key == 't':
                print('Key: T - Takeoff requested')
                self.takeoff_requested = True
            elif event.key == 'escape':
                print('Key: ESC - Exiting program...')
                self.mission_active = False
                self.mission_phase = 'COMPLETED'
    
    def on_key_release(self, event):
        """Handle keyboard release events for manual control."""
        with self._keyboard_lock:
            if event.key in ['up', 'down']:
                self.keyboard_vx = 0.0
                print(f'Key released: {event.key.upper()}')
            elif event.key in ['left', 'right']:
                self.keyboard_vy = 0.0
                print(f'Key released: {event.key.upper()}')
            elif event.key == 'q':
                self.keyboard_height_delta = 0.0
                print('Key released: Q')
            elif event.key == 'a':
                self.keyboard_height_delta = 0.0
                print('Key released: A')

    def send_position_setpoint(self, x, y, z, yaw, duration_s=1.0):
        """Control mode where position is sent as absolute (world) value.
        
        Args:
            x, y, z: Position in meters
            yaw: Yaw angle in degrees
            duration_s: Duration to send setpoints
        """
        dt = 1.0 / self.control_rate_hz
        steps = max(1, int(duration_s * self.control_rate_hz))
        for _ in range(steps):
            self.controller.cf.commander.send_position_setpoint(x, y, z, yaw)
            time.sleep(dt)
        
    # ---- Setup / Teardown ----

    def connect_cf(self):
        cflib.crtp.init_drivers()
        self.controller.connect(self.uri, x=self.start[0], y=self.start[1], z=self.current_z, yaw=self.current_yaw)
        if not self.controller.wait_for_connection(timeout=10.0):
            raise RuntimeError('Connection failed')
        self.controller.add_data_callback(self.position_callback)

    def setup_sensors(self):
        self.multiranger = Multiranger(self.controller.cf, rate_ms=100)
        self.multiranger.start()

    def setup_visualizer(self):
        self.visualizer = DroneNavigationVisualizer(
            xlim=(0, 6), ylim=(0, 3), figsize=(12, 9), animation_speed=0.05
        )
        self.visualizer.setup(start=self.start, goal=self.goal, grid_map=self.grid_map)
        
        # Connect keyboard event handlers
        self.visualizer.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.visualizer.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        
        # Add text display for keyboard controls
        self.visualizer.ax.text(
            0.02, 0.98, 
            'Keys: Arrows=Move, Q/A=Height, L=Land, T=Takeoff, ESC=Exit',
            transform=self.visualizer.ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        # Add position text display
        self.position_text = self.visualizer.ax.text(
            0.02, 0.90,
            '',
            transform=self.visualizer.ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        )

    def start_hover_timer(self):
        self.hover_timer = HoverTimer(
            self.controller.cf,
            rate_hz=self.control_rate_hz,  # Fixed rate for smooth hover commands
            hover_state=self.hover,
            lock=self._hover_lock,
            max_speed=self.max_speed_near_obstacle  # global safety clamp
        )
        self.hover_timer.start()

    def stop_hover_timer(self):
        if self.hover_timer:
            self.hover_timer.stop()

    # ---- Planning / Mapping ----

    def plan_path_to_target(self, from_pos, to_pos, create_new_planner=False):
        """
        Plan path from from_pos to to_pos and update waypoints.
        Uses D* Lite for efficient incremental replanning.
        
        Args:
            from_pos: (x, y) start position in world coords
            to_pos: (x, y) goal position in world coords
            create_new_planner: Force creation of new planner (for new goals)
        """
        # Create new planner if needed (initial planning or goal changed)
        if create_new_planner or self.planner is None:
            print(f"Creating new D* Lite planner for path from {from_pos} to {to_pos}")
            self.planner = DStarLite(self.grid_map, obstacle_threshold=0.5, show_animation=False)
            ok, self.waypoints_x, self.waypoints_y = self.planner.run(
                start=from_pos, goal=to_pos, simplify=True, smooth=False
            )
            if not ok or len(self.waypoints_x) == 0:
                print(f"Initial planning failed from {from_pos} to {to_pos}")
                return False
        else:
            # Use D* Lite dynamic replanning (much faster!)
            print("Using D* Lite dynamic replanning...")
            try:
                # Update planner's start position for replanning
                start_x_ind, start_y_ind = self.planner.world_to_grid(from_pos[0], from_pos[1])
                if start_x_ind is None:
                    print("Start position out of bounds")
                    return False
                
                # Update start position in planner
                self.planner.start.x = start_x_ind
                self.planner.start.y = start_y_ind
                
                # Compute path from current position
                path = self.planner.compute_current_path()
                if path is None or len(path) == 0:
                    print("No path found during replanning")
                    return False
                
                # Convert to world coordinates and simplify/smooth
                self.waypoints_x, self.waypoints_y = self.planner.path_to_world_coords(path)
                self.waypoints_x, self.waypoints_y = self.planner.simplify_path(
                    self.waypoints_x, self.waypoints_y
                )
                # self.waypoints_x, self.waypoints_y = self.planner.smooth_path(
                #     self.waypoints_x, self.waypoints_y
                # )
                
                print(f"Replanned path with {len(self.waypoints_x)} waypoints")
            except Exception as e:
                print(f"Error during dynamic replanning: {e}")
                # Fall back to creating new planner
                return self.plan_path_to_target(from_pos, to_pos, create_new_planner=True)
        
        self.waypoint_idx = 0
        if self.visualizer:
            self.visualizer.update_path(self.waypoints_x, self.waypoints_y)
        if len(self.waypoints_x) > 0:
            self.target_filter.reset(self.waypoints_x[0], self.waypoints_y[0])
        else:
            self.target_filter.reset(to_pos[0], to_pos[1])
        return True

    def plan_initial_path(self):
        """Plan initial path to goal."""
        if not self.plan_path_to_target(self.start, self.goal, create_new_planner=True):
            raise RuntimeError('Initial planning failed')

    def update_sensors(self):
        if not self.multiranger:
            return
        self.sensor_readings = {
            'front': self.multiranger.front,
            'back': self.multiranger.back,
            'left': self.multiranger.left,
            'right': self.multiranger.right
        }
        if self.visualizer and self.visualizer.is_setup:
            self.visualizer.update_sensors(self.sensor_readings)

    def update_map_if_needed(self, step):
        """
        Update map less frequently to reduce replanning oscillations.
        """
        if step % self.MAP_UPDATE_INTERVAL != 0:
            return 0
        
        cells_updated = update_map(
            self.grid_map, 
            self.sensor_readings, 
            self.current_x, 
            self.current_y, 
            self.current_yaw, 
            safety_margin=self.safety_margin,
            drone_clearance=0.03  # Circular clearance around drone
        )
        
        if cells_updated > 0:
            print(f"Map updated with {cells_updated} new obstacle cells")
        
        return cells_updated

    def replan_if_blocked(self, cells_updated):
        """
        Replan path when new obstacles are detected.
        Uses D* Lite's efficient incremental replanning.
        """
        if cells_updated == 0:
            return
        if self.planner is None:
            return
        if len(self.waypoints_x) == 0:
            return
        
        # Obstacles detected - trigger D* Lite replanning
        print(f"Obstacles detected, triggering D* Lite replanning...")
        
        # D* Lite's update_map() will detect changes and replan
        replanned = self.planner.update_map()
        
        if replanned:
            # Extract new path using dynamic replanning
            print("Extracting new path from D* Lite...")
            success = self.plan_path_to_target(
                (self.current_x, self.current_y), 
                self.current_target,
                create_new_planner=False  # Use incremental replanning
            )
            if not success:
                print("Dynamic replanning failed, creating new planner")
                self.plan_path_to_target(
                    (self.current_x, self.current_y), 
                    self.current_target,
                    create_new_planner=True
                )
        else:
            print("D* Lite reports no path changes needed")

    # ---- Command Computation ----

    def get_distance(self, x1, y1, x2, y2):
        """Calculate Euclidean distance between two points."""
        return np.hypot(x2 - x1, y2 - y1)
    
    def get_min_sensor_distance(self):
        """Get minimum distance from all active sensors."""
        dists = [d for d in self.sensor_readings.values() if d is not None]
        return min(dists) if dists else None

    def compute_and_update_hover(self):
        # If holding position, only use keyboard inputs (no autonomous navigation)
        if self.holding_position:
            with self._keyboard_lock:
                vx_b = self.keyboard_vx
                vy_b = self.keyboard_vy
                height_cmd = self.flight_height + self.keyboard_height_delta
            
            # Update hover commands
            self.update_hover(vx=vx_b, vy=vy_b, yaw=0.0, height=height_cmd)
            
            # Update visualizer - map still updates in background
            if self.visualizer:
                self.visualizer.update_repulsion(0.0, 0.0)
                self.visualizer.set_status(stuck=False, emergency=False, avoiding=False)
                self._update_position_display("MANUAL CONTROL")
                self.visualizer.render()
            return
        
        # Normal autonomous navigation with obstacle avoidance
        # Waypoint handling and target selection
        if self.waypoint_idx < len(self.waypoints_x):
            tx, ty = self.waypoints_x[self.waypoint_idx], self.waypoints_y[self.waypoint_idx]
            # Advance if close enough to current waypoint
            if self.get_distance(self.current_x, self.current_y, tx, ty) < self.WAYPOINT_ACCEPTANCE_RADIUS:
                self.waypoint_idx += 1
                # Select next waypoint or final target
                if self.waypoint_idx < len(self.waypoints_x):
                    target = (self.waypoints_x[self.waypoint_idx], self.waypoints_y[self.waypoint_idx])
                else:
                    target = self.current_target
            else:
                target = (tx, ty)
        else:
            target = self.current_target

        # Filtered target to avoid sudden changes
        ftx, fty = self.target_filter.update(target[0], target[1])

        # World-frame velocities
        vx_path_w, vy_path_w = compute_path_velocity((self.current_x, self.current_y), (ftx, fty), self.cruise_speed)

        # Avoidance with hysteresis
        vx_avoid_w, vy_avoid_w = compute_avoidance_velocity(
            self.sensor_readings, 
            self.current_yaw, 
            danger_dist=self.near_obstacle_dist, 
            gain=0.5, 
            hysteresis=0.1  # Add hysteresis zone
        )
        
        # Combined velocity
        vx_w = vx_path_w + vx_avoid_w
        vy_w = vy_path_w + vy_avoid_w

        # Limit velocity near obstacles with smoother transition
        min_dist = self.get_min_sensor_distance()
        if min_dist is not None and min_dist < self.near_obstacle_dist:
            speed = np.hypot(vx_w, vy_w)
            if speed > 1e-6 and speed > self.max_speed_near_obstacle:
                scale = self.max_speed_near_obstacle / speed
                vx_w *= scale
                vy_w *= scale

        # Apply velocity filter for smooth commands
        vx_w_filtered, vy_w_filtered = self.velocity_filter.update(vx_w, vy_w)

        # Convert to body frame
        vx_b, vy_b = world_to_body(vx_w_filtered, vy_w_filtered, self.current_yaw)
        
        # Add keyboard control (already in body frame)
        with self._keyboard_lock:
            vx_b += self.keyboard_vx
            vy_b += self.keyboard_vy
            height_cmd = self.flight_height + self.keyboard_height_delta
        
        # Update hover commands
        self.update_hover(vx=vx_b, vy=vy_b, yaw=0.0, height=height_cmd)

        # Visualizer updates (repulsion shown using avoidance vector)
        if self.visualizer:
            self.visualizer.update_repulsion(vx_avoid_w, vy_avoid_w)
            self.visualizer.set_status(stuck=False, emergency=False, avoiding=(np.hypot(vx_avoid_w, vy_avoid_w) > 0.02))
            self._update_position_display()
            self.visualizer.render()

    def _reset_hover_state(self):
        """Reset hover state to zero velocity at flight height."""
        with self._hover_lock:
            self.hover = {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'height': self.flight_height}
    
    def _set_hover_zero_velocity(self, height):
        """Helper to set hover state with zero velocity."""
        self.update_hover(vx=0.0, vy=0.0, yaw=0.0, height=height)
    
    def _get_control_timing(self, duration_s):
        """Calculate dt and steps for a given duration."""
        dt = 1.0 / self.control_rate_hz
        steps = max(1, int(duration_s * self.control_rate_hz))
        return dt, steps
    
    def _update_position_display(self, status=""):
        """Update position text in visualizer."""
        if self.visualizer and hasattr(self, 'position_text'):
            pos_str = f'Pos: ({self.current_x:.2f}, {self.current_y:.2f}, {self.current_z:.2f})m'
            if status:
                pos_str += f' [{status}]'
            self.position_text.set_text(pos_str)

    def hold_position(self, duration_s=1.0):
        """Hold current position for specified duration."""
        dt, steps = self._get_control_timing(duration_s)
        for _ in range(steps):
            self._set_hover_zero_velocity(self.flight_height)
            time.sleep(dt)

    def takeoff(self, duration_s=3.0):
        """Gradual takeoff to flight height."""
        dt, steps = self._get_control_timing(duration_s)
        for i in range(steps):
            z = max(self.MIN_HEIGHT, self.current_z + (self.flight_height - self.current_z) * (i + 1) / steps)
            print(f'Takeoff step {i+1}/{steps}, height={z:.2f} m')
            self._set_hover_zero_velocity(z)
            time.sleep(dt)

    def gradual_land(self, duration_s=3.0):
        """Gradual landing to ground."""
        dt, steps = self._get_control_timing(duration_s)
        for i in range(steps):
            z = max(self.MIN_HEIGHT, self.current_z * (1.0 - (i + 1) / steps))
            self._set_hover_zero_velocity(z)
            time.sleep(dt)

    def complete_landing_with_motor_off(self, duration_s=2.0):
        """Land completely and turn motors off."""
        print('Landing...')
        self.gradual_land(duration_s=duration_s)
        
        # Stop hover timer
        self.stop_hover_timer()
        
        # Send stop commands to turn off motors
        print('Stopping motors...')
        try:
            self.controller.cf.commander.send_stop_setpoint()
            time.sleep(0.1)
            self.controller.cf.commander.send_notify_setpoint_stop()
            time.sleep(0.5)
        except Exception as e:
            print(f'Warning: Error stopping motors: {e}')
        
        print('Motors stopped')
    
    def handle_landing_command(self):
        """Process landing command from keyboard."""
        if self.mission_phase == 'AT_GOAL_HOLDING':
            print('\n=== Landing at goal ===')
            self.complete_landing_with_motor_off(duration_s=2.0)
            self.mission_phase = 'AT_GOAL_LANDED'
            print('Landed at goal. Press T to takeoff and return to start.')
            
        elif self.mission_phase == 'AT_START_HOLDING':
            print('\n=== Landing at start ===')
            self.complete_landing_with_motor_off(duration_s=2.0)
            self.mission_phase = 'AT_START_LANDED'
            print('Landed at start. Press ESC to exit program.')
    
    def clear_area_around_position(self, x, y, radius=0.3):
        """
        Clear obstacles in a circular area around a position.
        
        This is useful to clear the goal/start area before planning a return path,
        preventing false obstacles from blocking the planner.
        
        Args:
            x, y: Position in world coordinates (m)
            radius: Radius of the circular area to clear (m)
        """
        # Number of points to sample in the circular area
        num_samples = int((radius / self.grid_map.resolution) ** 2 * 3.14 * 2)  # Approximate
        num_samples = max(20, min(num_samples, 200))  # Limit between 20 and 200
        
        cells_cleared = 0
        
        # Clear in a grid pattern within the radius
        steps = int(radius / self.grid_map.resolution) + 1
        for dx in np.linspace(-radius, radius, steps * 2):
            for dy in np.linspace(-radius, radius, steps * 2):
                if np.hypot(dx, dy) <= radius:  # Check if inside circle
                    clear_x = x + dx
                    clear_y = y + dy
                    ok = self.grid_map.set_value_from_xy_pos(clear_x, clear_y, 0.0)
                    if ok:
                        cells_cleared += 1
        
        if cells_cleared > 0:
            print(f"Cleared {cells_cleared} cells in {radius}m radius around ({x:.2f}, {y:.2f})")
        
        return cells_cleared
    
    def handle_takeoff_command(self):
        """Process takeoff command from keyboard."""
        if self.mission_phase == 'AT_GOAL_LANDED':
            print('\n=== Taking off from goal, returning to start ===')
            
            # Clear obstacles around current position (goal) and destination (start)
            # This prevents false obstacles from blocking the return path
            print('Clearing obstacles around goal and start positions...')
            self.clear_area_around_position(self.current_x, self.current_y, radius=0.5)
            self.clear_area_around_position(self.goal[0], self.goal[1], radius=0.5)
            self.clear_area_around_position(self.start[0], self.start[1], radius=0.5)
            
            # Restart hover timer before takeoff
            self._reset_hover_state()
            self.start_hover_timer()
            time.sleep(0.3)
            
            # Takeoff
            print('Taking off for return journey...')
            self.takeoff(duration_s=2.0)
            
            # Reset state
            self.mission_phase = 'RETURNING_TO_START'
            self.current_target = self.start
            self.holding_position = False
            self.velocity_filter.reset(0.0, 0.0)
            
            # Plan path back to start (create new planner for different goal)
            if not self.plan_path_to_target((self.current_x, self.current_y), self.start, create_new_planner=True):
                print('Warning: Could not plan return path')
            
            # Update visualizer goal marker
            if self.visualizer:
                self.visualizer.goal = self.start
            
            print('Returning to start...')
        else:
            print('Takeoff only available when landed at goal')

    # ---- Main ----

    def run(self, show_visualizer=True):
        print(f"\nConnecting to Crazyflie at {self.uri} ...")
        self.connect_cf()
        print('Connected')

        try:
            self.setup_sensors()
            time.sleep(0.3)
            if show_visualizer:
                self.setup_visualizer()

            # Start hover timer immediately; height controls takeoff/hold/land implicitly
            self.start_hover_timer()

            # Takeoff
            self.takeoff(duration_s=1.0)

            # Initial planning
            self.plan_initial_path()

            # Main control loop
            self.mission_active = True
            step = 0

            while self.mission_active and self.mission_phase != 'COMPLETED':
                # Process keyboard commands
                if self.landing_requested:
                    self.landing_requested = False
                    self.handle_landing_command()
                    continue
                
                if self.takeoff_requested:
                    self.takeoff_requested = False
                    self.handle_takeoff_command()
                    step = 0  # Reset step counter
                    continue
                
                # Check arrival at current target
                dist_to_target = self.get_distance(self.current_x, self.current_y, 
                                                   self.current_target[0], self.current_target[1])
                
                # Handle arrival at goal or start
                if dist_to_target < self.TARGET_ARRIVAL_THRESHOLD:
                    if self.mission_phase == 'GOING_TO_GOAL':
                        print('\n=== Arrived at goal - Holding position ===')
                        print('Use arrow keys to position above landing pad')
                        print('Use Q/A to adjust height')
                        print('Press L to land, then T to takeoff and return')
                        self.mission_phase = 'AT_GOAL_HOLDING'
                        self.holding_position = True
                        # Optionally clear area around goal now (helps with return planning later)
                        # self.clear_area_around_position(self.current_x, self.current_y, radius=0.25)
                    elif self.mission_phase == 'RETURNING_TO_START':
                        print('\n=== Arrived at start - Holding position ===')
                        print('Use arrow keys to position above landing pad')
                        print('Use Q/A to adjust height')
                        print('Press L to land, then ESC to exit')
                        self.mission_phase = 'AT_START_HOLDING'
                        self.holding_position = True
                elif dist_to_target > self.TARGET_ARRIVAL_THRESHOLD * 10:
                    self.holding_position = False

                # Wait in landed states (keep visualizer alive)
                if self.mission_phase in ['AT_GOAL_LANDED', 'AT_START_LANDED']:
                    if self.visualizer:
                        self._update_position_display("LANDED")
                        self.visualizer.render()
                    time.sleep(0.05)  # Small sleep to keep GUI responsive
                    continue

                # Sensors - always update to keep map responsive
                self.update_sensors()

                # Map updates - always run to keep map current
                # This allows manual positioning using keyboard at goal/start
                cells_updated = self.update_map_if_needed(step)
                
                # Replanning - only when autonomously navigating (not holding)
                if not self.holding_position:
                    self.replan_if_blocked(cells_updated)

                # Compute and update hover velocities
                self.compute_and_update_hover()

                step += 1
                # Small sleep for CPU
                time.sleep(0.05)

        except KeyboardInterrupt:
            print('Interrupted by user')
        except Exception as e:
            print(f'Error: {e}')
            import traceback
            traceback.print_exc()
        finally:
            self.mission_active = False
            # Stop hover timer if still running
            if self.hover_timer and self.hover_timer._running:
                self.stop_hover_timer()
            self.controller.cleanup()
            if self.multiranger:
                try:
                    self.multiranger.stop()
                except Exception:
                    pass
            try:
                self.controller.disconnect()
            except Exception:
                pass
            print('Cleanup complete')


def main():
    demo = RealHardwareHoverNav(
        cruise_speed=0.3,  # Reduced from 0.5 for smoother, more stable flight
        flight_height=0.5,
        control_rate_hz=20.0
    )
    demo.run(show_visualizer=True)


if __name__ == "__main__":
    main()