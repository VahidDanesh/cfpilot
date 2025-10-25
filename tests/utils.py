"""
Utility functions and classes for test programs
"""

import time
import threading
import numpy as np
import math


class HoverTimer:
    """Fixed-rate sender for hover setpoints."""

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
    """Convert world-frame velocity to body frame."""
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    vx_b =  cy * vx_w + sy * vy_w  # body x forward
    vy_b = -sy * vx_w + cy * vy_w  # body y left
    return vx_b, vy_b


def compute_path_velocity(current_xy, target_xy, speed):
    """Compute velocity vector toward target."""
    dx = target_xy[0] - current_xy[0]
    dy = target_xy[1] - current_xy[1]
    dist = np.hypot(dx, dy)
    if dist < 1e-6:
        return 0.0, 0.0
    return speed * dx / dist, speed * dy / dist


def compute_avoidance_velocity(sensors, yaw_rad, danger_dist=0.6, gain=0.3, hysteresis=0.05):
    """Compute avoidance velocity with hysteresis."""
    angles = {'front': 0.0, 'right': -np.pi/2, 'back': np.pi, 'left': np.pi/2}
    rx, ry = 0.0, 0.0
    effective_danger_dist = danger_dist + hysteresis
    
    for d in ['front', 'back', 'left', 'right']:
        dist = sensors.get(d)
        if dist is None or dist > effective_danger_dist:
            continue
        force = (effective_danger_dist - dist) / effective_danger_dist
        force = force * force
        ang = yaw_rad + angles[d]
        rx -= force * np.cos(ang)
        ry -= force * np.sin(ang)
    
    if abs(rx) < 0.05:
        rx = 0.0
    if abs(ry) < 0.05:
        ry = 0.0
    return gain * rx, gain * ry


def generate_sweep_pattern(center, width, height, spacing=0.3):
    """Generate zigzag sweep pattern for landing pad search."""
    cx, cy = center
    waypoints = []
    
    x_min = cx - width / 2
    x_max = cx + width / 2
    y_min = cy - height / 2
    y_max = cy + height / 2
    
    y = y_min
    direction = 1
    
    while y <= y_max:
        if direction == 1:
            waypoints.append((x_min, y))
            waypoints.append((x_max, y))
        else:
            waypoints.append((x_max, y))
            waypoints.append((x_min, y))
        
        y += spacing
        direction *= -1
    
    return waypoints


def interpolate_position(start_pos, end_pos, s):
    """
    Interpolate position between start and end using parameter s.
    
    Args:
        start_pos: (x, y) start position
        end_pos: (x, y) end position
        s: Path parameter from 0 (start) to 1 (end)
    
    Returns:
        (x, y) interpolated position
    """
    s = np.clip(s, 0.0, 1.0)
    x = start_pos[0] + s * (end_pos[0] - start_pos[0])
    y = start_pos[1] + s * (end_pos[1] - start_pos[1])
    return x, y


def navigate_smooth(cf_commander, start_pos, target_pos, height, duration_s, control_rate_hz):
    """
    Navigate smoothly from start to target using interpolated position setpoints.
    
    Args:
        cf_commander: Crazyflie commander object
        start_pos: (x, y) start position
        target_pos: (x, y) target position
        height: Flight height (m)
        duration_s: Duration of navigation (s)
        control_rate_hz: Control loop frequency (Hz)
    """
    dt = 1.0 / control_rate_hz
    steps = max(1, int(duration_s * control_rate_hz))
    
    for i in range(steps):
        s = (i + 1) / steps  # Path parameter from 0 to 1
        x, y = interpolate_position(start_pos, target_pos, s)
        cf_commander.send_position_setpoint(x, y, height, 0)
        time.sleep(dt)


def search_and_detect_pad(detector, current_pos_func, sweep_waypoints, 
                          cf_commander, height, control_rate_hz, 
                          confidence_threshold=0.45):
    """
    Execute sweep pattern and detect landing pad.
    
    Args:
        detector: LandingPadDetector instance
        current_pos_func: Function that returns current (x, y) position
        sweep_waypoints: List of (x, y) sweep waypoints
        cf_commander: Crazyflie commander
        height: Flight height
        control_rate_hz: Control rate
        confidence_threshold: Minimum confidence to accept detection
    
    Returns:
        (center_x, center_y) if detected, None otherwise
    """
    detector.start_detection(baseline_height=height)
    
    for i, target in enumerate(sweep_waypoints):
        current_pos = current_pos_func()
        
        # Navigate to waypoint
        navigate_smooth(cf_commander, current_pos, target, height, 
                       duration_s=2.0, control_rate_hz=control_rate_hz)
        
        # Check if pad detected
        if len(detector.peak_positions) >= 2:
            center = detector.calculate_pad_center()
            if center and detector.center_confidence > confidence_threshold:
                print(f'✅ Pad detected at ({center[0]:.3f}, {center[1]:.3f}), conf={detector.center_confidence:.2f}')
                detector.stop_detection()
                return center
    
    detector.stop_detection()
    print('⚠️  Sweep complete, no pad detected')
    return None


def land_on_pad(cf_commander, current_pos, pad_center, height, control_rate_hz):
    """Navigate to pad center and land."""
    print(f'🎯 Navigating to pad center ({pad_center[0]:.3f}, {pad_center[1]:.3f})')
    
    navigate_smooth(cf_commander, current_pos, pad_center, height,
                   duration_s=3.0, control_rate_hz=control_rate_hz)
    
    print('⏸️  Holding above center for 2 seconds...')
    for _ in range(int(2.0 * control_rate_hz)):
        cf_commander.send_position_setpoint(pad_center[0], pad_center[1], height, 0)
        time.sleep(1.0 / control_rate_hz)


def update_map(grid_map, sensors, x, y, yaw, safety_margin=0.05, drone_clearance=0.05):
    """
    Update map with sensor readings.
    
    Args:
        grid_map: GridMap instance
        sensors: Dict of sensor readings
        x, y: Current position
        yaw: Current yaw (radians)
        safety_margin: Margin around obstacles (default 0.05m)
        drone_clearance: Clearance around drone
    
    Returns:
        Number of cells updated
    """
    # Clear drone position
    for dx in np.linspace(-drone_clearance, drone_clearance, 4):
        for dy in np.linspace(-drone_clearance, drone_clearance, 4):
            if np.hypot(dx, dy) <= drone_clearance:
                grid_map.set_value_from_xy_pos(x + dx, y + dy, 0.0)
    
    # Update obstacles
    dirs = {'front': 0, 'right': -np.pi/2, 'back': np.pi, 'left': np.pi/2}
    cells_updated = 0
    
    for d, dist in sensors.items():
        if dist is None or dist > 2.0 or dist < 0.02:
            continue
        
        ang = yaw + dirs[d]
        ox = x + dist * np.cos(ang)
        oy = y + dist * np.sin(ang)
        
        for dx in np.linspace(-safety_margin, safety_margin, 3):
            for dy in np.linspace(-safety_margin, safety_margin, 3):
                ok = grid_map.set_value_from_xy_pos(ox + dx, oy + dy, 1.0)
                if ok:
                    cells_updated += 1
    
    return cells_updated

