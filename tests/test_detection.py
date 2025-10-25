"""
Landing Pad Detection Test Program

TUNING GUIDE:
1. Set use_state_estimate = True (smoother) or False (raw sensor)
2. Start with default parameters (lag=8, threshold=1.5, min_height=0.03)
3. Press D to start detection, fly slowly over pad
4. Watch console prints: Z-score must exceed threshold to detect edge
5. Adjust parameters (line ~70):
   - threshold too high → edges missed → LOWER it (1.5 → 1.2)
   - threshold too low → false positives → RAISE it (1.5 → 2.0)
   - lag: increase (8 → 12) for smoother, decrease (8 → 5) for faster response
   - min_peak_height: pad height difference (typically 0.03-0.05m)
6. Press R to reset, fly again to test new parameters
7. Need 6+ edges from different sides for good center calculation

Controls: Arrows=Move, Q/A=Height, D=Toggle, R=Reset, C=Center, L=Land, T=Takeoff, ESC=Exit
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
from cfpilot.detection import LandingPadDetector
from cfpilot.visualization import DroneNavigationVisualizer
from utils import HoverTimer, NoOpContext


class LandingPadDetectionTest:
    """Test program for landing pad detection parameter tuning"""
    
    def __init__(self, flight_height=0.5, control_rate_hz=50.0):
        self.controller = CrazyflieController()
        self.uri = self.controller.config['connection']['uri']
        
        # Starting position
        self.start = (0.0, 0.0)
        self.current_x = self.start[0]
        self.current_y = self.start[1]
        self.current_z = 0.1
        self.current_yaw = 0.0  # radians
        
        # Flight parameters
        self.flight_height = flight_height
        self.control_rate_hz = control_rate_hz
        self.keyboard_velocity = 0.15  # m/s for keyboard control (reduced for better control)
        self.keyboard_height_rate = 0.05  # m/s for height change (reduced)
        
        # Sensors - use multiranger with down sensor
        self.multiranger = None
        self.sensor_readings = {}
        
        # Landing pad detector
        self.detector = LandingPadDetector()
        
        # Detection source - choose which height measurement to use
        self.use_state_estimate = True  # True = stateEstimate.z (smoother), False = range.zrange (raw sensor)
        
        # Configure detection parameters (tunable)
        self.detector.configure_detection({
            'lag': 5,              # Window size for running statistics (try 5-15)
            'threshold': 2,        # Z-score threshold (try 1.0-2.5, lower = more sensitive)
            'influence': 0.6,      # Peak influence on running mean (0.1-0.5)
            'min_peak_height': 0.04,  # Minimum height difference in meters (try 0.02-0.05)
            'min_edge_distance': 0.03  # Minimum distance between edges to avoid clustering (try 0.08-0.15)
        })
        
        # Detection state
        self.detection_active = False
        
        # Visualizer
        self.visualizer = None
        self.position_text = None
        self.detection_text = None
        
        # Hover control
        self.hover = {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'height': self.flight_height}
        self._hover_lock = threading.Lock()
        self.hover_timer = None
        
        # Keyboard control
        self.keyboard_vx = 0.0  # body frame
        self.keyboard_vy = 0.0
        self.keyboard_height_delta = 0.0
        self._keyboard_lock = threading.Lock()
        
        # Command flags
        self.landing_requested = False
        self.takeoff_requested = False
        self.running = True
        self.step_counter = 0
    
    # ---- Callbacks ----
    
    def position_callback(self, timestamp, data, logconf_name):
        """Update position from state estimate."""
        self.current_x = data.get('stateEstimate.x', 0.0)
        self.current_y = data.get('stateEstimate.y', 0.0)
        self.current_z = data.get('stateEstimate.z', 0.0)
        yaw_deg = data.get('stabilizer.yaw', 0.0)
        self.current_yaw = np.deg2rad(yaw_deg)
        
        if self.visualizer and self.visualizer.is_setup:
            self.visualizer.update_drone(self.current_x, self.current_y, self.current_yaw)
    
    def process_multiranger_data(self):
        """Process height sensor for landing pad detection."""
        if not self.detection_active:
            return
        
        # Choose height source
        if self.use_state_estimate:
            # Use state estimate Z (smoother, filtered by Kalman filter)
            height_m = self.current_z
            source = "StateEst.z"
        else:
            # Use raw z-range sensor (noisier but direct measurement)
            if not self.multiranger or self.multiranger.down is None:
                return
            height_m = self.multiranger.down
            source = "zrange"
        
        # Calculate z-score for debugging
        z_score = 0.0
        if len(self.detector.running_mean) > 0:
            current_mean = np.mean(self.detector.running_mean)
            current_std = np.mean(self.detector.running_std) if self.detector.running_std else 0.001
            if current_std > 0:
                z_score = abs(height_m - current_mean) / current_std
        
        # Print detection info every 20 measurements
        if len(self.detector.height_data) % 20 == 0 and len(self.detector.height_data) > 0:
            baseline = self.detector.baseline_height if self.detector.baseline_height else 0.0
            height_diff = abs(height_m - baseline)
            print(f'{source}={height_m:.3f}m, Z-score={z_score:.2f}, diff={height_diff:.3f}m, thresh={self.detector.threshold}')
        
        # Process height measurement for detection
        position = (self.current_x, self.current_y)
        self.detector.process_height_measurement(height_m, position)
        
        # Auto-calculate center when enough edges detected
        num_edges = len(self.detector.peak_positions)
        if num_edges >= 2 and num_edges % 2 == 0:  # Calculate every 2 new edges
            center = self.detector.calculate_pad_center()
            if center:
                conf_status = "READY" if self.detector.center_confidence > 0.6 else "LOW CONFIDENCE"
                print(f'Center: ({center[0]:.3f}, {center[1]:.3f}), conf={self.detector.center_confidence:.2f} [{conf_status}]')
        
        # Update visualization with detected edges and center
        if self.visualizer:
            edge_positions = [p['position'] for p in self.detector.peak_positions]
            center = self.detector.calculated_center
            self.visualizer.update_detected_edges(edge_positions, center)
    
    def on_key_press(self, event):
        """Handle keyboard press events."""
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
            elif event.key == 'd':
                # Toggle detection
                self.toggle_detection()
            elif event.key == 'l':
                print('Key: L - Landing requested')
                self.landing_requested = True
            elif event.key == 't':
                print('Key: T - Takeoff requested')
                self.takeoff_requested = True
            elif event.key == 'r':
                # Reset detection (clear detected edges)
                print('Key: R - Resetting detection data')
                self.detector.peak_positions.clear()
                self.detector.calculated_center = None
                if self.visualizer:
                    self.visualizer.update_detected_edges([], None)
            elif event.key == 'c':
                # Calculate pad center
                print('Key: C - Calculating pad center')
                center = self.detector.calculate_pad_center()
                if center:
                    print(f'✅ Pad center: ({center[0]:.3f}, {center[1]:.3f}), '
                          f'confidence: {self.detector.center_confidence:.2f}')
                else:
                    print('❌ Insufficient data for center calculation')
            elif event.key == 'escape':
                print('Key: ESC - Exiting program...')
                self.running = False
    
    def on_key_release(self, event):
        """Handle keyboard release events."""
        with self._keyboard_lock:
            if event.key in ['up', 'down']:
                self.keyboard_vx = 0.0
            elif event.key in ['left', 'right']:
                self.keyboard_vy = 0.0
            elif event.key in ['q', 'a']:
                self.keyboard_height_delta = 0.0
    
    def toggle_detection(self):
        """Toggle landing pad detection on/off."""
        self.detection_active = not self.detection_active
        if self.detection_active:
            self.detector.start_detection(baseline_height=self.flight_height)
            print('🟢 Detection STARTED')
        else:
            self.detector.stop_detection()
            print('🔴 Detection STOPPED')
    
    # ---- Setup ----
    
    def connect_cf(self):
        """Connect to Crazyflie."""
        cflib.crtp.init_drivers()
        self.controller.connect(self.uri, x=self.start[0], y=self.start[1], 
                               z=self.current_z, yaw=self.current_yaw)
        if not self.controller.wait_for_connection(timeout=10.0):
            raise RuntimeError('Connection failed')
        
        # Add position callback
        self.controller.add_data_callback(self.position_callback)
    
    def setup_sensors(self):
        """Setup multiranger sensors (including down sensor for detection)."""
        self.multiranger = Multiranger(self.controller.cf, rate_ms=100)
        self.multiranger.start()
        print('✅ Multiranger started (with down sensor for detection)')
    
    def setup_visualizer(self):
        """Setup visualization with detection display."""
        self.visualizer = DroneNavigationVisualizer(
            xlim=(-1, 3), ylim=(-1, 3), figsize=(14, 10), animation_speed=0.001
        )
        self.visualizer.setup(start=self.start, goal=(1.5, 0.0))
        
        import matplotlib.pyplot as plt
        plt.ion()  # Interactive mode
        
        # Connect keyboard handlers
        self.visualizer.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.visualizer.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        
        # Add controls text
        controls_text = (
            'Controls:\n'
            '  Arrows: Move  Q/A: Height\n'
            '  D: Toggle Detection  R: Reset\n'
            '  C: Calculate Center\n'
            '  L: Land  T: Takeoff  ESC: Exit'
        )
        self.visualizer.ax.text(
            0.02, 0.98, controls_text,
            transform=self.visualizer.ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9)
        )
        
        # Add position display
        self.position_text = self.visualizer.ax.text(
            0.02, 0.75,
            '',
            transform=self.visualizer.ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9)
        )
        
        # Add detection status display
        self.detection_text = self.visualizer.ax.text(
            0.02, 0.50,
            '',
            transform=self.visualizer.ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9)
        )
    
    def start_hover_timer(self):
        """Start hover command timer."""
        self.hover_timer = HoverTimer(
            self.controller.cf,
            rate_hz=self.control_rate_hz,
            hover_state=self.hover,
            lock=self._hover_lock
        )
        self.hover_timer.start()
    
    def stop_hover_timer(self):
        """Stop hover command timer."""
        if self.hover_timer:
            self.hover_timer.stop()
    
    # ---- Control ----
    
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
    
    def compute_and_update_hover(self):
        """Compute hover commands from keyboard input."""
        # Process multiranger data for detection first
        self.process_multiranger_data()
        
        with self._keyboard_lock:
            vx_b = self.keyboard_vx
            vy_b = self.keyboard_vy
            height_cmd = self.flight_height + self.keyboard_height_delta
        
        # Obstacle avoidance - stop if too close
        min_dist = self.get_min_sensor_distance()
        if min_dist is not None and min_dist < 0.2:
            # Too close to obstacle, stop movement
            vx_b = 0.0
            vy_b = 0.0
        
        # Update hover commands
        self.update_hover(vx=vx_b, vy=vy_b, yaw=0.0, height=height_cmd)
        
        # Update visualizations - only render every 10 steps for responsiveness
        if self.visualizer and hasattr(self, 'step_counter'):
            self.update_sensor_display()
            self._update_position_display()
            self._update_detection_display()
            if self.step_counter % 10 == 0:
                try:
                    self.visualizer.render(force=True)
                    import matplotlib.pyplot as plt
                    plt.pause(0.001)
                except Exception as e:
                    pass
    
    def update_sensor_display(self):
        """Update sensor visualization."""
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
    
    def get_min_sensor_distance(self):
        """Get minimum distance from all active sensors."""
        dists = [d for d in self.sensor_readings.values() if d is not None]
        return min(dists) if dists else None
    
    def _update_position_display(self):
        """Update position text."""
        if self.visualizer and hasattr(self, 'position_text'):
            down_str = f'{self.multiranger.down:.3f}m' if self.multiranger and self.multiranger.down is not None else 'N/A'
            source = "StateEst" if self.use_state_estimate else "zrange"
            pos_str = (
                f'Position:\n'
                f'  X: {self.current_x:.3f}m\n'
                f'  Y: {self.current_y:.3f}m\n'
                f'  StateEst.z: {self.current_z:.3f}m\n'
                f'  zrange: {down_str}\n'
                f'  Using: {source}'
            )
            self.position_text.set_text(pos_str)
    
    def _update_detection_display(self):
        """Update detection status text."""
        if self.visualizer and hasattr(self, 'detection_text'):
            status = '🟢 ACTIVE' if self.detection_active else '🔴 INACTIVE'
            num_edges = len(self.detector.peak_positions)
            
            det_str = f'Detection: {status}\n'
            det_str += f'Edges: {num_edges}'
            
            # Show center and landing readiness
            if self.detector.calculated_center:
                cx, cy = self.detector.calculated_center
                conf = self.detector.center_confidence
                det_str += f'\n\n✅ CENTER FOUND!\n'
                det_str += f'  X: {cx:.3f}m\n'
                det_str += f'  Y: {cy:.3f}m\n'
                det_str += f'  Confidence: {conf:.2f}\n'
                if conf > 0.6:
                    det_str += f'\n🟢 READY TO LAND!'
                else:
                    det_str += f'\n🟡 Low confidence\n   (get more edges for better)'
            elif num_edges >= 1:
                det_str += f'\n\n🟡 Need 1 more edge for center'
            else:
                det_str += f'\n\n🔴 Fly over pad edges'
            
            # Show detection parameters
            det_str += f'\n\nParams:\n'
            det_str += f'  lag={self.detector.lag}\n'
            det_str += f'  threshold={self.detector.threshold}\n'
            det_str += f'  min_height={self.detector.min_peak_height}m\n'
            det_str += f'  min_dist={self.detector.min_edge_distance}m\n'
            
            # Show current baseline
            if self.detector.baseline_height is not None:
                det_str += f'  baseline={self.detector.baseline_height:.3f}m\n'
            
            # Show running stats if available
            if len(self.detector.running_mean) > 0:
                current_mean = np.mean(self.detector.running_mean)
                current_std = np.mean(self.detector.running_std) if self.detector.running_std else 0
                det_str += f'  mean={current_mean:.3f}m\n'
                det_str += f'  std={current_std:.4f}m\n'
            
            self.detection_text.set_text(det_str)
    
    def takeoff(self, duration_s=3.0):
        """Gradual takeoff."""
        dt = 1.0 / self.control_rate_hz
        steps = max(1, int(duration_s * self.control_rate_hz))
        min_height = 0.2
        
        for i in range(steps):
            z = max(min_height, self.current_z + (self.flight_height - self.current_z) * (i + 1) / steps)
            self.update_hover(vx=0.0, vy=0.0, yaw=0.0, height=z)
            time.sleep(dt)
    
    def navigate_to_center(self, duration_s=3.0):
        """Navigate to detected pad center using position setpoint."""
        if not self.detector.calculated_center:
            print('❌ No center calculated yet!')
            return False
        
        target_x, target_y = self.detector.calculated_center
        print(f'🎯 Navigating to center ({target_x:.3f}, {target_y:.3f})...')
        
        # Send position setpoint to go directly to center
        dt = 1.0 / self.control_rate_hz
        steps = max(1, int(duration_s * self.control_rate_hz))
        
        for i in range(steps):
            self.controller.cf.commander.send_position_setpoint(
                target_x, target_y, self.flight_height, 0  # yaw=0 degrees
            )
            
            # Update visualization periodically
            if i % 10 == 0 and self.visualizer:
                self.update_sensor_display()
                self._update_position_display()
                self._update_detection_display()
                try:
                    self.visualizer.render(force=True)
                    import matplotlib.pyplot as plt
                    plt.pause(0.001)
                except Exception:
                    pass
            
            time.sleep(dt)
        
        # Check if arrived
        dist = np.hypot(self.current_x - target_x, self.current_y - target_y)
        print(f'✅ Position command sent. Final distance: {dist:.3f}m')
        return True
    
    def land(self, duration_s=3.0):
        """Gradual landing."""
        dt = 1.0 / self.control_rate_hz
        steps = max(1, int(duration_s * self.control_rate_hz))
        min_height = 0.2
        
        for i in range(steps):
            z = max(min_height, self.current_z * (1.0 - (i + 1) / steps))
            self.update_hover(vx=0.0, vy=0.0, yaw=0.0, height=z)
            time.sleep(dt)
        
        # Stop motors
        self.stop_hover_timer()
        try:
            self.controller.cf.commander.send_stop_setpoint()
            time.sleep(0.1)
            self.controller.cf.commander.send_notify_setpoint_stop()
        except Exception as e:
            print(f'Warning: {e}')
    
    # ---- Main Loop ----
    
    def run(self, show_visualizer=True):
        """Run the test program."""
        print('\n' + '='*60)
        print('Landing Pad Detection Test Program')
        print('='*60)
        print('\nAssuming landing pad is ~1m forward from start position')
        print('\nControls:')
        print('  Arrows: Move forward/backward/left/right')
        print('  Q/A: Increase/decrease height')
        print('  D: Toggle detection ON/OFF')
        print('  R: Reset detected edges')
        print('  C: Calculate pad center')
        print('  L: Land, T: Takeoff')
        print('  ESC: Exit program')
        print('='*60)
        
        print(f'\nConnecting to Crazyflie at {self.uri}...')
        self.connect_cf()
        print('✅ Connected')
        
        try:
            self.setup_sensors()
            time.sleep(0.3)
            
            if show_visualizer:
                self.setup_visualizer()
            
            # Start hover timer
            self.start_hover_timer()
            
            # Takeoff
            print('\n🚁 Taking off...')
            self.takeoff(duration_s=2.0)
            print('✅ Airborne - Use keyboard to control')
            print('\n💡 Tip: Press D to start detection, then fly forward slowly')
            print('         Watch for red X markers showing detected edges')
            
            # Main control loop
            while self.running:
                # Handle landing command
                if self.landing_requested:
                    self.landing_requested = False
                    
                    # Stop detection during landing
                    was_detecting = self.detection_active
                    if was_detecting:
                        self.detector.stop_detection()
                        self.detection_active = False
                        print('🔴 Detection stopped for landing')
                    
                    # If center is detected, navigate to it first
                    if self.detector.calculated_center and self.detector.center_confidence > 0.45:
                        print('\n🎯 Center detected - navigating to center before landing...')
                        self.navigate_to_center(duration_s=3.0)
                        
                        print('⏸️  Holding position above center for 2 seconds...')
                        cx, cy = self.detector.calculated_center
                        for _ in range(int(2.0 * self.control_rate_hz)):
                            self.controller.cf.commander.send_position_setpoint(
                                cx, cy, self.flight_height, 0
                            )
                            time.sleep(1.0 / self.control_rate_hz)
                        
                        print('🛬 Landing on pad center...')
                        self.land(duration_s=2.0)
                        print('✅ Landed on pad!')
                    else:
                        print('\n🛬 No center detected - landing at current position...')
                        self.land(duration_s=2.0)
                        print('✅ Landed')
                    
                    time.sleep(0.5)
                    continue
                
                # Handle takeoff command
                if self.takeoff_requested:
                    self.takeoff_requested = False
                    self.start_hover_timer()
                    print('\n🚁 Taking off...')
                    self.takeoff(duration_s=2.0)
                    print('✅ Airborne')
                    continue
                
                # Update hover commands and process detection
                self.compute_and_update_hover()
                self.step_counter += 1
                
                time.sleep(0.02)
        
        except KeyboardInterrupt:
            print('\n⚠️  Interrupted by user')
        except Exception as e:
            print(f'\n❌ Error: {e}')
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
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
            print('\n✅ Cleanup complete')
            
            # Print final statistics
            if len(self.detector.peak_positions) > 0:
                print('\n' + '='*60)
                print('DETECTION SUMMARY')
                print('='*60)
                stats = self.detector.get_detection_statistics()
                print(f'Total measurements: {stats["total_measurements"]}')
                print(f'Border points detected: {stats["total_border_points"]}')
                print(f'Baseline height: {stats["baseline_height"]:.3f}m')
                if stats["calculated_center"]:
                    cx, cy = stats["calculated_center"]
                    print(f'Pad center: ({cx:.3f}, {cy:.3f})')
                    print(f'Confidence: {stats["center_confidence"]:.2f}')
                print('='*60)


def main():
    test = LandingPadDetectionTest(
        flight_height=0.5,
        control_rate_hz=50.0
    )
    test.run(show_visualizer=True)


if __name__ == "__main__":
    main()

