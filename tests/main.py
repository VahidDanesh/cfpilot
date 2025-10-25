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
        self.x = initial_x
        self.y = initial_y

    def update(self, x, y):
        if self.x is None:
            self.x, self.y = float(x), float(y)
        else:
            self.x = self.alpha * float(x) + (1 - self.alpha) * self.x
            self.y = self.alpha * float(y) + (1 - self.alpha) * self.y
        return self.x, self.y

    def reset(self, x=0.0, y=0.0):
        self.x, self.y = float(x), float(y)

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
        ang = yaw_rad + angles[d]
        rx -= force * np.cos(ang)
        ry -= force * np.sin(ang)
    
    # Larger deadzone to prevent micro-movements
    if abs(rx) < 0.05:
        rx = 0.0
    if abs(ry) < 0.05:
        ry = 0.0
    return gain * rx, gain * ry


def plan_path_with_dstar(grid_map, start_pos, goal_pos):
    planner = DStarLite(grid_map, obstacle_threshold=0.5, show_animation=False)
    ok, xs, ys = planner.run(start=start_pos, goal=goal_pos, simplify=True, smooth=True)
    if not ok or len(xs) == 0:
        return False, [], []
    return True, list(xs), list(ys)


def update_map(grid_map, sensors, x, y, yaw, safety_margin=0.15):
    dirs = {'front': 0, 'right': -np.pi/2, 'back': np.pi, 'left': np.pi/2}
    cells_updated = 0
    for d, dist in sensors.items():
        if dist is None or dist > 2.0 or dist < 0.02:
            continue
        ang = yaw + dirs[d]
        ox = x + dist * np.cos(ang)
        oy = y + dist * np.sin(ang)
        for dx in np.linspace(-safety_margin, safety_margin, 5):
            for dy in np.linspace(-safety_margin, safety_margin, 5):
                ok = grid_map.set_value_from_xy_pos(ox + dx, oy + dy, 1.0)
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
    WAYPOINT_ACCEPTANCE_RADIUS = 0.10  # m
    TARGET_ARRIVAL_THRESHOLD = 0.02    # m
    MIN_HEIGHT = 0.2                   # m - minimum safe height
    MAP_UPDATE_INTERVAL = 5           # steps between map updates
    
    def __init__(self,
                 cruise_speed=0.2,     # m/s along path
                 flight_height=0.5,
                 control_rate_hz=50.0):
        self.controller = CrazyflieController()
        self.uri = self.controller.config['connection']['uri']

        self.start = (0.6, 1.25)
        self.goal = (5.45, 0.75)
        self.current_x = self.start[0]
        self.current_y = self.start[1]
        self.current_z = 0.1
        self.current_yaw = 0.0  # radians

        # Mission state tracking
        self.mission_phase = 'GOING_TO_GOAL'  # GOING_TO_GOAL, RETURNING_TO_START, COMPLETED
        self.current_target = self.goal

        self.cruise_speed = cruise_speed
        self.flight_height = flight_height
        self.control_rate_hz = control_rate_hz
        self.safety_margin = 0.05              # m - extra obstacle margin
        self.near_obstacle_dist = 0.2          # m - start repulsion distance
        self.max_speed_near_obstacle = 0.15    # m/s - max speed near obstacles

        self.grid_map = GridMap(
            width=int(5.8 / 0.05), height=int(2.7 / 0.05),
            resolution=0.05, center_x=2.9, center_y=1.35
        )
        self.grid_map.occupy_boundaries(boundary_width=2)

        self.waypoints_x = []
        self.waypoints_y = []
        self.waypoint_idx = 0

        self.multiranger = None
        self.sensor_readings = {}

        self.visualizer = None

        # Hover state and timer (names like multiranger_pointcloud.py)
        self.hover = {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'height': self.flight_height}
        self._hover_lock = threading.Lock()
        self.hover_timer = None

        self.mission_active = False
        self.target_filter = ExponentialFilter(alpha=0.9)  # Smooth target tracking
        self.velocity_filter = ExponentialFilter(alpha=0.9, initial_x=0.0, initial_y=0.0)  # Smooth velocity commands

    # ---- Callbacks / IO ----

    def position_callback(self, timestamp, data, logconf_name):
        self.current_x = data.get('stateEstimate.x', 0.0)
        self.current_y = data.get('stateEstimate.y', 0.0)
        self.current_z = data.get('stateEstimate.z', 0.0)
        yaw_deg = data.get('stabilizer.yaw', 0.0)
        self.current_yaw = np.deg2rad(yaw_deg)
        if self.visualizer and self.visualizer.is_setup:
            self.visualizer.update_drone(self.current_x, self.current_y, self.current_yaw)

    def update_hover(self, k, v):
        """Thread-safe update of hover state."""
        with self._hover_lock:
            self.hover[k] = v

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
            xlim=(0, 5), ylim=(0, 3), figsize=(12, 9), animation_speed=0.05
        )
        self.visualizer.setup(start=self.start, goal=self.goal, grid_map=self.grid_map)

    def start_hover_timer(self):
        self.hover_timer = HoverTimer(
            self.controller.cf,
            rate_hz=self.control_rate_hz,
            hover_state=self.hover,
            lock=self._hover_lock,
            max_speed=self.max_speed_near_obstacle  # global safety clamp
        )
        self.hover_timer.start()

    def stop_hover_timer(self):
        if self.hover_timer:
            self.hover_timer.stop()

    # ---- Planning / Mapping ----

    def plan_path_to_target(self, from_pos, to_pos):
        """Plan path from from_pos to to_pos and update waypoints."""
        ok, self.waypoints_x, self.waypoints_y = plan_path_with_dstar(self.grid_map, from_pos, to_pos)
        if not ok:
            return False
        self.waypoint_idx = 0
        if self.visualizer:
            self.visualizer.update_path(self.waypoints_x, self.waypoints_y)
        if self.waypoints_x:
            self.target_filter.reset(self.waypoints_x[0], self.waypoints_y[0])
        else:
            self.target_filter.reset(to_pos[0], to_pos[1])
        return True

    def plan_initial_path(self):
        """Plan initial path to goal."""
        if not self.plan_path_to_target(self.start, self.goal):
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
        """Update map less frequently to reduce replanning oscillations."""
        if step % self.MAP_UPDATE_INTERVAL != 0:
            return 0
        return update_map(self.grid_map, 
                          self.sensor_readings, 
                          self.current_x, 
                          self.current_y, 
                          self.current_yaw, 
                          safety_margin=self.safety_margin)

    def replan_if_blocked(self, cells_updated):
        """Replan path if waypoint is blocked by new obstacles."""
        if cells_updated == 0:
            return
        if self.waypoint_idx >= len(self.waypoints_x):
            return
        wx, wy = self.waypoints_x[self.waypoint_idx], self.waypoints_y[self.waypoint_idx]
        ix, iy = self.grid_map.get_xy_index_from_xy_pos(wx, wy)
        if ix is None:
            return
        val = self.grid_map.get_value_from_xy_index(ix, iy)
        if val is not None and val > 0.5:
            self.plan_path_to_target((self.current_x, self.current_y), self.current_target)

    # ---- Command Computation ----

    def get_distance(self, x1, y1, x2, y2):
        """Calculate Euclidean distance between two points."""
        return np.hypot(x2 - x1, y2 - y1)
    
    def get_min_sensor_distance(self):
        """Get minimum distance from all active sensors."""
        dists = [d for d in self.sensor_readings.values() if d is not None]
        return min(dists) if dists else None

    def compute_and_update_hover(self):
        # Waypoint handling with slightly larger acceptance radius for smoother transitions
        if self.waypoint_idx < len(self.waypoints_x):
            tx, ty = self.waypoints_x[self.waypoint_idx], self.waypoints_y[self.waypoint_idx]
            if self.get_distance(self.current_x, self.current_y, tx, ty) < self.WAYPOINT_ACCEPTANCE_RADIUS:
                self.waypoint_idx += 1

        # Select target
        if self.waypoint_idx < len(self.waypoints_x):
            target = (self.waypoints_x[self.waypoint_idx], self.waypoints_y[self.waypoint_idx])
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
            gain=0.4,  # Reduced from 0.5 for gentler avoidance
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
        
        # Update hover commands
        self.update_hover('x', vx_b)
        self.update_hover('y', vy_b)
        self.update_hover('yaw', 0.0)
        self.update_hover('height', self.flight_height)

        # Visualizer updates (repulsion shown using avoidance vector)
        if self.visualizer:
            self.visualizer.update_repulsion(vx_avoid_w, vy_avoid_w)
            self.visualizer.set_status(stuck=False, emergency=False, avoiding=(np.hypot(vx_avoid_w, vy_avoid_w) > 0.02))
            self.visualizer.render()

    def _set_hover_zero_velocity(self, height):
        """Helper to set hover state with zero velocity."""
        self.update_hover('x', 0.0)
        self.update_hover('y', 0.0)
        self.update_hover('yaw', 0.0)
        self.update_hover('height', height)

    def hold_position(self, duration_s=1.0):
        """Hold current position for specified duration."""
        dt = 1.0 / self.control_rate_hz
        steps = max(1, int(duration_s * self.control_rate_hz))
        for _ in range(steps):
            self._set_hover_zero_velocity(self.flight_height)
            time.sleep(dt)

    def takeoff(self, duration_s=3.0):
        """Gradual takeoff to flight height."""
        dt = 1.0 / self.control_rate_hz
        steps = max(1, int(duration_s * self.control_rate_hz))
        for i in range(steps):
            z = max(self.MIN_HEIGHT, self.current_z + (self.flight_height - self.current_z) * (i + 1) / steps)
            print(f'Takeoff step {i+1}/{steps}, height={z:.2f} m')
            self._set_hover_zero_velocity(z)
            time.sleep(dt)

    def gradual_land(self, duration_s=3.0):
        """Gradual landing to ground."""
        dt = 1.0 / self.control_rate_hz
        steps = max(1, int(duration_s * self.control_rate_hz))
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

    def handle_landing_and_phase_transition(self):
        """Handle landing, phase transition, and takeoff for next leg."""
        if self.mission_phase == 'GOING_TO_GOAL':
            print('\n=== Arrived at goal, landing ===')
            self.send_position_setpoint(self.goal[0], self.goal[1], self.flight_height, 0.0, duration_s=1.0)
            
            # Complete landing with motors off
            self.complete_landing_with_motor_off(duration_s=2.0)
            
            # Pause with motors off
            print('Resting on goal pad (motors off)...')
            time.sleep(1.0)
            
            # Switch to return phase
            print('\n=== Starting return journey ===')
            self.mission_phase = 'RETURNING_TO_START'
            self.current_target = self.start
            
            # Restart hover timer before takeoff
            print('Restarting hover timer...')
            with self._hover_lock:
                self.hover = {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'height': self.flight_height}
            self.start_hover_timer()
            time.sleep(0.3)
            
            # Takeoff again
            print('Taking off for return journey...')
            self.takeoff(duration_s=2.0)
            
            # Reset velocity filter for clean start on return journey
            self.velocity_filter.reset(0.0, 0.0)
            
            # Plan path back to start
            if not self.plan_path_to_target((self.current_x, self.current_y), self.start):
                print('Warning: Could not plan return path')
            
            # Update visualizer goal marker
            if self.visualizer:
                self.visualizer.goal = self.start
                
        elif self.mission_phase == 'RETURNING_TO_START':
            print('\n=== Returned to start, landing ===')
            self.send_position_setpoint(self.start[0], self.start[1], self.flight_height, 0.0, duration_s=1.0)
            
            # Complete landing with motors off
            self.complete_landing_with_motor_off(duration_s=2.0)
            
            print('Mission complete!')
            self.mission_phase = 'COMPLETED'

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
            control_dt = 1.0 / self.control_rate_hz
            step = 0

            while self.mission_active and self.mission_phase != 'COMPLETED':
                # Check arrival at current target
                dist_to_target = self.get_distance(self.current_x, self.current_y, 
                                                   self.current_target[0], self.current_target[1])
                
                if dist_to_target < self.TARGET_ARRIVAL_THRESHOLD:
                    self.handle_landing_and_phase_transition()
                    if self.mission_phase == 'COMPLETED':
                        break
                    # Reset step counter for new phase
                    step = 0
                    continue

                # Sensors
                self.update_sensors()

                # Map + optional simple replan
                cells_updated = self.update_map_if_needed(step)
                self.replan_if_blocked(cells_updated)

                # Compute and update hover velocities
                self.compute_and_update_hover()

                step += 1
                time.sleep(control_dt)

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
        control_rate_hz=50.0
    )
    demo.run(show_visualizer=True)


if __name__ == "__main__":
    main()