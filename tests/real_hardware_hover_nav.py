"""
Real Hardware Hover Navigation Example (Hover mode)
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
from cfpilot.detection import LandingPadDetector, SearchPattern


class PositionFilter:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.fx = None
        self.fy = None

    def update(self, x, y):
        if self.fx is None:
            self.fx, self.fy = float(x), float(y)
        else:
            self.fx = self.alpha * float(x) + (1 - self.alpha) * self.fx
            self.fy = self.alpha * float(y) + (1 - self.alpha) * self.fy
        return self.fx, self.fy

    def reset(self, x, y):
        self.fx, self.fy = float(x), float(y)

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


def compute_avoidance_velocity(sensors, yaw_rad, danger_dist=0.6, gain=0.3):
    angles = {'front': 0.0, 'right': -np.pi/2, 'back': np.pi, 'left': np.pi/2}
    rx, ry = 0.0, 0.0
    for d in ['front', 'back', 'left', 'right']:
        dist = sensors.get(d)
        if dist is None or dist > danger_dist:
            continue
        force = (danger_dist - dist) / danger_dist
        force = force * force
        ang = yaw_rad + angles[d] + np.pi  # away from obstacle
        rx += force * np.cos(ang)
        ry += force * np.sin(ang)
    if abs(rx) < 0.01:
        rx = 0.0
    if abs(ry) < 0.01:
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
        if dist is None or dist > 2.0 or dist < 0.05:
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
            if self._lock:
                self._lock.acquire()
            try:
                vx = self.hover.get('x', 0.0)
                vy = self.hover.get('y', 0.0)
                yaw = self.hover.get('yaw', 0.0)
                h  = self.hover.get('height', 0.5)
            finally:
                if self._lock:
                    self._lock.release()
            # Clamp speed if configured
            if self._max_speed is not None:
                speed = (vx * vx + vy * vy) ** 0.5
                if speed > 1e-6 and speed > self._max_speed:
                    scale = self._max_speed / speed
                    vx *= scale
                    vy *= scale
            try:
                self.cf.commander.send_hover_setpoint(vx, vy, yaw, h)
            except Exception:
                pass
            next_t += self.period
            sleep_t = next_t - time.perf_counter()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                next_t = time.perf_counter()


class RealHardwareHoverNav:
    def __init__(self,
                 cruise_speed=0.2,     # m/s along path
                 searching_speed=0.1,  # m/s during landing pad search
                 flight_height=0.5,
                 control_rate_hz=50.0):
        self.controller = CrazyflieController()
        self.uri = self.controller.config['connection']['uri']

        self.start = (0.7, 1.5)
        self.goal = (2.8, 1.5)
        self.current_x = self.start[0]
        self.current_y = self.start[1]
        self.current_z = 0.1
        self.current_yaw = 0.0  # radians
        self.current_z_range = None  # z-range sensor reading


        self.cruise_speed = cruise_speed
        self.searching_speed = searching_speed
        self.flight_height = flight_height
        self.control_rate_hz = control_rate_hz
        self.safety_margin = 0.05 # This will add extra obstacle margin in meters
        self.near_obstacle_dist = 0.5 # m to start repulsion from obstacles
        self.max_speed_near_obstacle = 0.15 # m/s max speed when near obstacles

        # Landing pad detection parameters (TUNABLE)
        self.landing_zone_radius = 0.1  # Distance to goal to trigger search (m)
        self.search_area_size = 0.6     # Cross pattern search extent (m)
        self.search_step_size = 0.3    # Distance to move in each direction (m)
        self.pad_center_tolerance = 0.05  # How close to get to calculated center (m)
        self.search_timeout = 60.0      # Max search time (seconds)

        self.grid_map = GridMap(
            width=int(5.0 / 0.05), height=int(3.0 / 0.05),
            resolution=0.05, center_x=2.5, center_y=1.5
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
        self.target_filter = PositionFilter(alpha=0.4)

        # Landing pad detection
        self.detector = LandingPadDetector()
        self.search_pattern = SearchPattern()
        
        # Configure detection for 0.3x0.3m pad at 0.1m height
        self.detector.configure_detection({
            'lag': 8,
            'threshold': 1.5,      # TUNABLE: 1.0-2.5
            'influence': 0.2,      # TUNABLE: 0.1-0.3
            'min_peak_height': 0.05  # TUNABLE: 0.03-0.08 (pad is 0.1m high)
        })

        # Mission state machine
        self.mission_state = 'NAVIGATING'  # NAVIGATING, SEARCHING, CENTERING, LANDING
        self.search_waypoints = []
        self.search_waypoint_idx = 0
        self.search_start_time = None
        self.calculated_pad_center = None
        self.detected_edges = []

    # ---- Callbacks / IO ----

    def position_callback(self, timestamp, data, logconf_name):
        self.current_x = data.get('stateEstimate.x', 0.0)
        self.current_y = data.get('stateEstimate.y', 0.0)
        self.current_z = data.get('stateEstimate.z', 0.0)
        yaw_deg = data.get('stabilizer.yaw', 0.0)
        self.current_yaw = np.deg2rad(yaw_deg)
        if self.visualizer and self.visualizer.is_setup:
            self.visualizer.update_drone(self.current_x, self.current_y, self.current_yaw)

    def updateHover(self, k, v):
        with self._hover_lock:
            self.hover[k] = v

    def sendHoverCommand(self):
        # Provided for API parity; real sending is done by HoverTimer
        with self._hover_lock:
            vx = float(self.hover['x'])
            vy = float(self.hover['y'])
            yaw = float(self.hover['yaw'])
            h = float(self.hover['height'])
        # Clamp speed for safety before sending
        speed = (vx * vx + vy * vy) ** 0.5
        max_speed = getattr(self, 'max_speed_near_obstacle', None)
        if max_speed is not None and speed > 1e-6 and speed > max_speed:
            scale = max_speed / speed
            vx *= scale
            vy *= scale
        self.controller.cf.commander.send_hover_setpoint(vx, vy, yaw, h)

    def send_position_setpoint(self, x, y, z, yaw, duration_s=1.0):

        """
        Control mode where the position is sent as an absolute (world) value.

        position [x, y, z] are in m
        yaw is in degrees
        """
        dt = 1.0 / self.control_rate_hz
        steps = max(1, int(duration_s * self.control_rate_hz))
        for i in range(steps):
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

    def plan_initial_path(self):
        ok, self.waypoints_x, self.waypoints_y = plan_path_with_dstar(self.grid_map, self.start, self.goal)
        if not ok:
            raise RuntimeError('Initial planning failed')
        if self.visualizer:
            self.visualizer.update_path(self.waypoints_x, self.waypoints_y)
        if self.waypoints_x:
            self.target_filter.reset(self.waypoints_x[0], self.waypoints_y[0])
        else:
            self.target_filter.reset(self.goal[0], self.goal[1])

    def update_sensors(self):
        if not self.multiranger:
            return
        self.sensor_readings = {
            'front': self.multiranger.front,
            'back': self.multiranger.back,
            'left': self.multiranger.left,
            'right': self.multiranger.right
        }
        # Update z-range for landing pad detection
        self.current_z_range = self.multiranger.down
        
        if self.visualizer and self.visualizer.is_setup:
            self.visualizer.update_sensors(self.sensor_readings)

    def update_map_if_needed(self, step):
        if step % 3 != 0:
            return 0
        return update_map(self.grid_map, 
                          self.sensor_readings, 
                          self.current_x, 
                          self.current_y, 
                          self.current_yaw, 
                          safety_margin=self.safety_margin)

    def replan_if_blocked(self, cells_updated):
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
            ok, self.waypoints_x, self.waypoints_y = plan_path_with_dstar(self.grid_map, 
                                                                          (self.current_x, self.current_y), 
                                                                          self.goal)
            if ok:
                self.waypoint_idx = 0
                if self.visualizer:
                    self.visualizer.update_path(self.waypoints_x, self.waypoints_y)
                if self.waypoints_x:
                    self.target_filter.reset(self.waypoints_x[0], self.waypoints_y[0])

    # ---- Landing Pad Detection ----

    def generate_cross_search_pattern(self, center):
        """Generate cross pattern: center -> forward -> back -> center -> left -> right"""
        cx, cy = center
        waypoints = []
        
        # Start at center
        waypoints.append((cx, cy, 'center'))
        num_steps = int(self.search_area_sizQe / self.search_step_size)

        
        # Forward pass (positive x direction)
        for i in range(1, num_steps + 1):
            waypoints.append((cx + i * self.search_step_size, cy, 'forward'))

        # Backward pass (negative x direction)
        for i in range(1, num_steps + 1):
            waypoints.append((cx - i * self.search_step_size, cy, 'back'))    

        # Back to center
        waypoints.append((cx, cy, 'center'))

        # Left pass (positive y direction)
        for i in range(1, num_steps + 1):
            waypoints.append((cx, cy + i * self.search_step_size, 'left'))
        
        
        # Right pass (negative y direction)
        for i in range(1, num_steps + 1):
            waypoints.append((cx, cy - i * self.search_step_size, 'right'))
        
        # Back to center
        waypoints.append((cx, cy, 'center'))

        
        print(f"Generated cross search pattern with {len(waypoints)} waypoints")
        return waypoints

    def start_search_mode(self):
        """Initialize search mode"""
        print(f"Starting search mode at ({self.current_x:.3f}, {self.current_y:.3f})")
        self.mission_state = 'SEARCHING'
        self.search_start_time = time.time()
        
        # Generate search pattern around current position (landing zone)
        self.search_waypoints = self.generate_cross_search_pattern((self.current_x, self.current_y))
        self.search_waypoint_idx = 0
        
        # Start detection
        self.detector.start_detection()
        self.detected_edges = []

    def process_detection(self):
        """Process height measurement for landing pad detection"""
        if self.mission_state != 'SEARCHING':
            return
        
        if self.current_z_range is None or self.current_z_range < 0.05 or self.current_z_range > 1.0:
            return
        
        # Calculate height (flight_height - z_range gives height above surface)
        surface_height = self.flight_height - self.current_z_range
        
        # Get current search direction
        if self.search_waypoint_idx < len(self.search_waypoints):
            direction = self.search_waypoints[self.search_waypoint_idx][2]
            self.detector.set_flight_direction(direction)
        
        # Process measurement
        self.detector.process_height_measurement(surface_height, (self.current_x, self.current_y))
        
        # Update detected edges list for visualization
        self.detected_edges = [p['position'] for p in self.detector.peak_positions]

    def check_search_completion(self):
        """Check if search should end and calculate pad center"""
        if self.mission_state != 'SEARCHING':
            return False
        
        # Check timeout
        if time.time() - self.search_start_time > self.search_timeout:
            print("Search timeout reached")
            return self.finalize_search()
        
        # Check if all waypoints visited
        if self.search_waypoint_idx >= len(self.search_waypoints):
            print("Search pattern complete")
            return self.finalize_search()
        
        return False

    def finalize_search(self):
        """Finalize search and calculate pad center"""
        self.detector.stop_detection()
        stats = self.detector.get_detection_statistics()
        print(f"Detection stats: {stats['total_border_points']} edges detected")
        
        # Try to calculate center
        center = self.detector.calculate_pad_center()
        
        if center and self.detector.is_ready_for_landing():
            self.calculated_pad_center = center
            print(f"Pad center calculated: ({center[0]:.3f}, {center[1]:.3f}), "
                  f"confidence: {self.detector.center_confidence:.2f}")
            self.mission_state = 'CENTERING'
            return True
        else:
            print(f"Detection failed or low confidence. Landing at goal position.")
            self.calculated_pad_center = self.goal
            self.mission_state = 'CENTERING'
            return True

    # ---- Command Computation ----

    def compute_and_update_hover(self):
        # Determine target and speed based on mission state
        if self.mission_state == 'NAVIGATING':
            # Normal path following
            if self.waypoint_idx < len(self.waypoints_x):
                tx, ty = self.waypoints_x[self.waypoint_idx], self.waypoints_y[self.waypoint_idx]
                if np.hypot(tx - self.current_x, ty - self.current_y) < 0.12:
                    self.waypoint_idx += 1

            if self.waypoint_idx < len(self.waypoints_x):
                target = (self.waypoints_x[self.waypoint_idx], self.waypoints_y[self.waypoint_idx])
            else:
                target = self.goal
            
            current_speed = self.cruise_speed

        elif self.mission_state == 'SEARCHING':
            # Follow search pattern
            if self.search_waypoint_idx < len(self.search_waypoints):
                tx, ty, _ = self.search_waypoints[self.search_waypoint_idx]
                if np.hypot(tx - self.current_x, ty - self.current_y) < 0.08:
                    self.search_waypoint_idx += 1
                target = (tx, ty)
            else:
                target = (self.current_x, self.current_y)  # Hold position
            
            current_speed = self.searching_speed

        elif self.mission_state == 'CENTERING':
            # Move to calculated pad center
            if self.calculated_pad_center:
                target = self.calculated_pad_center
                current_speed = self.searching_speed
            else:
                target = self.goal
                current_speed = self.searching_speed
        
        else:
            # Default: hold position
            target = (self.current_x, self.current_y)
            current_speed = 0.0

        # Filtered target to avoid sudden changes
        ftx, fty = self.target_filter.update(target[0], target[1])

        # World-frame velocities
        vx_path_w, vy_path_w = compute_path_velocity((self.current_x, self.current_y), (ftx, fty), current_speed)

        vx_avoid_w, vy_avoid_w = compute_avoidance_velocity(self.sensor_readings, 
                                                            self.current_yaw, 
                                                            danger_dist=self.near_obstacle_dist, 
                                                            gain=0.5)
        vx_w = vx_path_w + vx_avoid_w
        vy_w = vy_path_w + vy_avoid_w

        # Limit velocity near obstacles
        dists = [d for d in [self.sensor_readings.get('front'), self.sensor_readings.get('back'), self.sensor_readings.get('left'), self.sensor_readings.get('right')] if d is not None]
        min_dist = min(dists) if dists else None
        if min_dist is not None and min_dist < self.near_obstacle_dist:
            speed = np.hypot(vx_w, vy_w)
            if speed > 1e-6 and speed > self.max_speed_near_obstacle:
                scale = self.max_speed_near_obstacle / speed
                vx_w *= scale
                vy_w *= scale

        # Convert to body frame and keep yaw at 0 deg/s
        vx_b, vy_b = world_to_body(vx_w, vy_w, self.current_yaw)
        self.updateHover('x', vx_b)
        self.updateHover('y', vy_b)
        self.updateHover('yaw', 0.0)
        self.updateHover('height', self.flight_height)

        # Visualizer updates (repulsion shown using avoidance vector)
        if self.visualizer:
            self.visualizer.update_repulsion(vx_avoid_w, vy_avoid_w)
            self.visualizer.set_status(stuck=False, emergency=False, avoiding=(np.hypot(vx_avoid_w, vy_avoid_w) > 0.02))
            
            # Show detected edges and calculated center
            self.visualizer.update_detected_edges(self.detected_edges, self.calculated_pad_center)
            
            self.visualizer.render()

    def hold_position(self, duration_s=1.0):
        dt = 1.0 / self.control_rate_hz
        steps = max(1, int(duration_s * self.control_rate_hz))
        for _ in range(steps):
            self.updateHover('x', 0.0)
            self.updateHover('y', 0.0)
            self.updateHover('yaw', 0.0)
            self.updateHover('height', self.flight_height)
            time.sleep(dt)


    def takeoff(self):
        self.updateHover('x', 0.0)
        self.updateHover('y', 0.0)
        self.updateHover('yaw', 0.0)
        self.updateHover('height', self.flight_height)


    def gradual_land(self, duration_s=3.0):
        dt = 1.0 / self.control_rate_hz
        steps = max(1, int(duration_s * self.control_rate_hz))
        with self._hover_lock:
            start_z = float(self.hover.get('height', self.flight_height))
        for i in range(steps):
            z = max(0.2, start_z * (1.0 - (i + 1) / steps))
            self.updateHover('x', 0.0)
            self.updateHover('y', 0.0)
            self.updateHover('yaw', 0.0)
            self.updateHover('height', z)
            time.sleep(dt)

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

            # Set initial hover commands (height set elsewhere as absolute)
            self.takeoff()

            # Hold after takeoff before moving
            # self.hold_position(duration_s=1.0)
            self.send_position_setpoint(self.start[0], self.start[1], self.flight_height, 0.0, duration_s=0.5)

            # Initial planning
            self.plan_initial_path()

            # Main control loop with state machine
            self.mission_active = True
            control_dt = 1.0 / self.control_rate_hz
            step = 0

            while self.mission_active:
                # State-based control
                dist_to_goal = np.hypot(self.goal[0] - self.current_x, self.goal[1] - self.current_y)
                
                # STATE TRANSITIONS
                if self.mission_state == 'NAVIGATING':
                    # Check if entered landing zone
                    if dist_to_goal < self.landing_zone_radius:
                        print(f"\n=== Entering landing zone (dist={dist_to_goal:.3f}m) ===")
                        self.start_search_mode()
                
                elif self.mission_state == 'SEARCHING':
                    # Process detection
                    self.process_detection()
                    
                    # Check search completion
                    if self.check_search_completion():
                        print("=== Search complete, moving to pad center ===")
                
                elif self.mission_state == 'CENTERING':
                    # Check if reached calculated center
                    if self.calculated_pad_center:
                        dist_to_center = np.hypot(self.calculated_pad_center[0] - self.current_x,
                                                   self.calculated_pad_center[1] - self.current_y)
                        if dist_to_center < self.pad_center_tolerance:
                            print(f"=== Reached pad center, initiating landing ===")
                            self.mission_state = 'LANDING'
                            # Precision positioning before landing
                            self.send_position_setpoint(self.calculated_pad_center[0], 
                                                       self.calculated_pad_center[1], 
                                                       self.flight_height, 0.0, duration_s=1.0)
                            self.gradual_land(duration_s=2.0)
                            print('Landed successfully on pad')
                            break
                
                # Sensors
                self.update_sensors()

                # Map + optional simple replan (only during NAVIGATING)
                if self.mission_state == 'NAVIGATING':
                    cells_updated = self.update_map_if_needed(step)
                    self.replan_if_blocked(cells_updated)

                # Compute and update hover velocities
                self.compute_and_update_hover()

                # Debug output every 2 seconds
                if step % (self.control_rate_hz * 2) == 0:
                    print(f"[{self.mission_state}] Pos: ({self.current_x:.2f}, {self.current_y:.2f}), "
                          f"Dist to goal: {dist_to_goal:.2f}m, "
                          f"Z-range: {self.current_z_range:.3f}m" if self.current_z_range else "Z-range: N/A")
                    if self.mission_state == 'SEARCHING':
                        print(f"  Search progress: {self.search_waypoint_idx}/{len(self.search_waypoints)}, "
                              f"Edges detected: {len(self.detected_edges)}")

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
        cruise_speed=0.3,
        searching_speed=0.1,
        flight_height=0.5,
        control_rate_hz=50.0
    )
    demo.run(show_visualizer=True)


if __name__ == "__main__":
    main()


