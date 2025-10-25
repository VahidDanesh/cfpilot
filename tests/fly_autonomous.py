"""
Autonomous Landing Pad Detection Mission

Mission:
1. Takeoff from start
2. Navigate to goal zone  
3. Search and detect landing pad at goal
4. Land on detected pad
5. Takeoff and return to start
6. Search and detect landing pad at start
7. Land on detected pad at start
"""

import time
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
from cfpilot.planning.DStarLite.d_star_lite import DStarLite
from cfpilot.mapping import GridMap
from cfpilot.visualization import DroneNavigationVisualizer
from utils import (HoverTimer, ExponentialFilter, world_to_body,
                   compute_path_velocity, compute_avoidance_velocity,
                   generate_sweep_pattern, navigate_smooth, 
                   search_and_detect_pad, land_on_pad, update_map)


class AutonomousMission:
    """Autonomous mission with landing pad detection at both start and goal"""
    
    def __init__(self, cruise_speed=0.3, flight_height=0.5, control_rate_hz=20.0):
        self.controller = CrazyflieController()
        self.uri = self.controller.config['connection']['uri']
        
        self.start = (0.6, 1.3)
        self.goal = (4.5, 2.0)
        self.current_x, self.current_y, self.current_z = self.start[0], self.start[1], 0.1
        self.current_yaw = 0.0
        
        self.cruise_speed = cruise_speed
        self.flight_height = flight_height
        self.control_rate_hz = control_rate_hz
        
        # Map and planning
        self.grid_map = GridMap(width=int(5.4/0.05), height=int(2.6/0.05),
                               resolution=0.05, center_x=2.7, center_y=1.3)
        self.grid_map.occupy_boundaries(boundary_width=2)
        self.planner = None
        self.waypoints_x, self.waypoints_y = np.array([]), np.array([])
        self.waypoint_idx = 0
        
        # Sensors and detection
        self.multiranger = None
        self.sensor_readings = {}
        self.detector = LandingPadDetector()
        self.detector.configure_detection({
            'lag': 5, 'threshold': 2, 'influence': 0.6,
            'min_peak_height': 0.04, 'min_edge_distance': 0.03
        })
        
        # Hover control
        self.hover = {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'height': self.flight_height}
        self._hover_lock = __import__('threading').Lock()
        self.hover_timer = None
        
        # Filters
        self.target_filter = ExponentialFilter(alpha=0.6)
        self.velocity_filter = ExponentialFilter(alpha=0.7, initial_x=0.0, initial_y=0.0)
        
        # Visualizer
        self.visualizer = None
        
        self.mission_active = False
        self.step_counter = 0
        self.emergency_stop = False
    
    def position_callback(self, timestamp, data, logconf_name):
        """Update position."""
        self.current_x = data.get('stateEstimate.x', 0.0)
        self.current_y = data.get('stateEstimate.y', 0.0)
        self.current_z = data.get('stateEstimate.z', 0.0)
        self.current_yaw = np.deg2rad(data.get('stabilizer.yaw', 0.0))
        
        if self.visualizer and self.visualizer.is_setup:
            self.visualizer.update_drone(self.current_x, self.current_y, self.current_yaw)
    
    def process_detection(self):
        """Process detection during flight."""
        height_m = self.current_z
        self.detector.process_height_measurement(height_m, (self.current_x, self.current_y))
    
    def connect_cf(self):
        """Connect to Crazyflie."""
        cflib.crtp.init_drivers()
        self.controller.connect(self.uri, x=self.start[0], y=self.start[1], 
                               z=self.current_z, yaw=self.current_yaw)
        if not self.controller.wait_for_connection(timeout=10.0):
            raise RuntimeError('Connection failed')
        self.controller.add_data_callback(self.position_callback)
        
        self.multiranger = Multiranger(self.controller.cf, rate_ms=100)
        self.multiranger.start()
    
    def start_hover_timer(self):
        """Start hover timer."""
        self.hover_timer = HoverTimer(self.controller.cf, rate_hz=self.control_rate_hz,
                                     hover_state=self.hover, lock=self._hover_lock,
                                     max_speed=0.2)
        self.hover_timer.start()
    
    def stop_hover_timer(self):
        """Stop hover timer."""
        if self.hover_timer:
            self.hover_timer.stop()
    
    def update_hover(self, vx=None, vy=None, yaw=None, height=None):
        """Update hover state."""
        with self._hover_lock:
            if vx is not None: self.hover['x'] = vx
            if vy is not None: self.hover['y'] = vy
            if yaw is not None: self.hover['yaw'] = yaw
            if height is not None: self.hover['height'] = height
    
    def plan_path(self, from_pos, to_pos, create_new_planner=True):
        """Plan path using D* Lite with dynamic replanning support."""
        # Convert positions to scalars
        from_pos = (float(from_pos[0]), float(from_pos[1]))
        to_pos = (float(to_pos[0]), float(to_pos[1]))
        
        if create_new_planner or self.planner is None:
            print(f"Creating new D* Lite planner from {from_pos} to {to_pos}")
            self.planner = DStarLite(self.grid_map, obstacle_threshold=0.5, show_animation=False)
            ok, self.waypoints_x, self.waypoints_y = self.planner.run(
                start=from_pos, goal=to_pos, simplify=True, smooth=False
            )
            if not ok or len(self.waypoints_x) == 0:
                raise RuntimeError('Planning failed')
            
            # Convert to numpy arrays
            self.waypoints_x = np.array(self.waypoints_x)
            self.waypoints_y = np.array(self.waypoints_y)
        else:
            # Dynamic replanning
            print("Using D* Lite dynamic replanning...")
            try:
                start_x_ind, start_y_ind = self.planner.world_to_grid(from_pos[0], from_pos[1])
                if start_x_ind is None:
                    print("Start position out of bounds")
                    return False
                
                self.planner.start.x = int(start_x_ind)
                self.planner.start.y = int(start_y_ind)
                
                path = self.planner.compute_current_path()
                if path is None or len(path) == 0:
                    print("No path found during replanning")
                    return False
                
                self.waypoints_x, self.waypoints_y = self.planner.path_to_world_coords(path)
                self.waypoints_x, self.waypoints_y = self.planner.simplify_path(
                    self.waypoints_x, self.waypoints_y
                )
                
                # Convert to numpy arrays
                self.waypoints_x = np.array(self.waypoints_x)
                self.waypoints_y = np.array(self.waypoints_y)
                
                print(f"Replanned path with {len(self.waypoints_x)} waypoints")
            except Exception as e:
                print(f"Error during replanning: {e}")
                import traceback
                traceback.print_exc()
                return self.plan_path(from_pos, to_pos, create_new_planner=True)
        
        self.waypoint_idx = 0
        if len(self.waypoints_x) > 0:
            self.target_filter.reset(float(self.waypoints_x[0]), float(self.waypoints_y[0]))
        else:
            self.target_filter.reset(to_pos[0], to_pos[1])
        return True
    
    def update_sensors(self):
        """Update sensor readings."""
        if not self.multiranger:
            return
        self.sensor_readings = {
            'front': self.multiranger.front, 'back': self.multiranger.back,
            'left': self.multiranger.left, 'right': self.multiranger.right
        }
        if self.visualizer and self.visualizer.is_setup:
            self.visualizer.update_sensors(self.sensor_readings)
    
    def replan_if_blocked(self, cells_updated, current_target):
        """Replan path when obstacles detected using D* Lite."""
        if cells_updated == 0 or self.planner is None:
            return
        if len(self.waypoints_x) == 0:
            return
        
        print(f"Obstacles detected ({cells_updated} cells), triggering D* Lite replanning...")
        
        replanned = self.planner.update_map()
        
        if replanned:
            print("Extracting new path from D* Lite...")
            success = self.plan_path((self.current_x, self.current_y), current_target,
                                    create_new_planner=False)
            if not success:
                print("Dynamic replanning failed, creating new planner")
                self.plan_path((self.current_x, self.current_y), current_target,
                              create_new_planner=True)
            
            if self.visualizer:
                self.visualizer.update_path(self.waypoints_x, self.waypoints_y)
        else:
            print("D* Lite reports no path changes needed")
    
    def setup_visualizer(self):
        """Setup visualization with keyboard handler."""
        self.visualizer = DroneNavigationVisualizer(
            xlim=(0, 6), ylim=(0, 3), figsize=(14, 10), animation_speed=0.001
        )
        self.visualizer.setup(start=self.start, goal=self.goal, grid_map=self.grid_map)
        
        import matplotlib.pyplot as plt
        plt.ion()
        
        # Add keyboard handler for emergency stop
        def on_key_press(event):
            if event.key == 'escape':
                print('\n⚠️  EMERGENCY STOP - ESC pressed!')
                self.emergency_stop = True
                self.mission_active = False
        
        self.visualizer.fig.canvas.mpl_connect('key_press_event', on_key_press)
        print('ℹ️  Press ESC to emergency stop')
    
    def navigate_with_avoidance(self, target):
        """Navigate to target with obstacle avoidance."""
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
        
        # Compute velocities
        vx_path_w, vy_path_w = compute_path_velocity(
            (self.current_x, self.current_y), (ftx, fty), self.cruise_speed
        )
        vx_avoid_w, vy_avoid_w = compute_avoidance_velocity(
            self.sensor_readings, self.current_yaw, danger_dist=0.4, gain=0.5
        )
        
        vx_w, vy_w = vx_path_w + vx_avoid_w, vy_path_w + vy_avoid_w
        
        # Speed limiting near obstacles
        dists = [d for d in self.sensor_readings.values() if d is not None]
        if dists and min(dists) < 0.4:
            speed = np.hypot(vx_w, vy_w)
            if speed > 0.2:
                vx_w, vy_w = 0.2 * vx_w / speed, 0.2 * vy_w / speed
        
        vx_w_f, vy_w_f = self.velocity_filter.update(vx_w, vy_w)
        vx_b, vy_b = world_to_body(vx_w_f, vy_w_f, self.current_yaw)
        
        self.update_hover(vx=vx_b, vy=vy_b, yaw=0.0, height=self.flight_height)
        
        # Update visualizer more frequently
        if self.visualizer and self.step_counter % 3 == 0:
            self.visualizer.update_repulsion(vx_avoid_w, vy_avoid_w)
            avoiding = np.hypot(vx_avoid_w, vy_avoid_w) > 0.02
            self.visualizer.set_status(stuck=False, emergency=self.emergency_stop, avoiding=avoiding)
            try:
                self.visualizer.render(force=True)
                import matplotlib.pyplot as plt
                plt.pause(0.001)
            except Exception:
                pass
    
    def takeoff(self, duration_s=3.0):
        """Takeoff."""
        dt = 1.0 / self.control_rate_hz
        steps = max(1, int(duration_s * self.control_rate_hz))
        for i in range(steps):
            z = max(0.2, self.current_z + (self.flight_height - self.current_z) * (i+1) / steps)
            self.update_hover(vx=0.0, vy=0.0, yaw=0.0, height=z)
            time.sleep(dt)
    
    def land(self, duration_s=3.0):
        """Land and stop motors."""
        dt = 1.0 / self.control_rate_hz
        steps = max(1, int(duration_s * self.control_rate_hz))
        for i in range(steps):
            z = max(0.2, self.current_z * (1.0 - (i+1) / steps))
            self.update_hover(vx=0.0, vy=0.0, yaw=0.0, height=z)
            time.sleep(dt)
        
        self.stop_hover_timer()
        try:
            self.controller.cf.commander.send_stop_setpoint()
            time.sleep(0.1)
            self.controller.cf.commander.send_notify_setpoint_stop()
        except Exception:
            pass
    
    def run(self):
        """Run autonomous mission."""
        print('\n' + '='*60)
        print('AUTONOMOUS LANDING PAD DETECTION MISSION')
        print('='*60)
        
        self.connect_cf()
        print('✅ Connected')
        
        try:
            time.sleep(0.3)
            self.setup_visualizer()
            self.start_hover_timer()
            
            # PHASE 1: Takeoff
            print('\n🚁 Taking off...')
            self.takeoff(duration_s=2.0)
            print('✅ Airborne')
            
            # PHASE 2: Navigate to goal
            print(f'\n🗺️  Planning path to goal {self.goal}...')
            self.plan_path(self.start, self.goal)
            print('✅ Path planned')
            
            if self.visualizer:
                self.visualizer.update_path(self.waypoints_x, self.waypoints_y)
            
            print('🚀 Navigating to goal...')
            self.mission_active = True
            while self.mission_active and not self.emergency_stop:
                self.update_sensors()
                
                # Update map every few steps
                if self.step_counter % 2 == 0:
                    cells_updated = update_map(self.grid_map, self.sensor_readings,
                                              self.current_x, self.current_y, self.current_yaw,
                                              safety_margin=0.05)
                    
                    # Replan if obstacles detected
                    self.replan_if_blocked(cells_updated, self.goal)
                
                dist = np.hypot(self.current_x - self.goal[0], self.current_y - self.goal[1])
                if dist < 0.2:
                    break
                self.navigate_with_avoidance(self.goal)
                self.step_counter += 1
                time.sleep(0.01)
            
            if self.emergency_stop:
                raise KeyboardInterrupt('Emergency stop')
            print('✅ Arrived at goal')
            
            # PHASE 3: Search and land on pad at goal
            print('\n🔍 Searching for landing pad at goal...')
            sweep = generate_sweep_pattern(self.goal, width=1.0, height=1.0, spacing=0.3)
            
            # Show sweep pattern on visualizer
            if self.visualizer:
                sweep_x = [wp[0] for wp in sweep]
                sweep_y = [wp[1] for wp in sweep]
                self.visualizer.ax.plot(sweep_x, sweep_y, 'c--', linewidth=1, alpha=0.5, label='Sweep Pattern')
                self.visualizer.ax.scatter(sweep_x, sweep_y, c='cyan', s=30, alpha=0.6, marker='x')
            
            # Execute sweep with live visualization
            self.detector.start_detection(baseline_height=self.flight_height)
            pad_center = None
            
            for i, sweep_target in enumerate(sweep):
                print(f'Sweep waypoint {i+1}/{len(sweep)}: ({sweep_target[0]:.2f}, {sweep_target[1]:.2f})')
                
                # Plan path to sweep waypoint
                self.plan_path((self.current_x, self.current_y), sweep_target, create_new_planner=False)
                if self.visualizer:
                    self.visualizer.update_path(self.waypoints_x, self.waypoints_y)
                
                # Navigate to sweep waypoint with visualization
                self.mission_active = True
                while self.mission_active and not self.emergency_stop:
                    self.update_sensors()
                    self.process_detection()
                    
                    dist = np.hypot(self.current_x - sweep_target[0], self.current_y - sweep_target[1])
                    if dist < 0.1:
                        break
                    
                    self.navigate_with_avoidance(sweep_target)
                    self.step_counter += 1
                    time.sleep(0.01)
                
                # Check if pad detected
                if len(self.detector.peak_positions) >= 2:
                    center = self.detector.calculate_pad_center()
                    if center and self.detector.center_confidence > 0.45:
                        print(f'✅ Pad detected at ({center[0]:.3f}, {center[1]:.3f}), conf={self.detector.center_confidence:.2f}')
                        pad_center = center
                        break
            
            self.detector.stop_detection()
            
            # Show detected edges and center
            if self.visualizer and len(self.detector.peak_positions) > 0:
                edges_x = [edge['position'][0] for edge in self.detector.peak_positions]
                edges_y = [edge['position'][1] for edge in self.detector.peak_positions]
                self.visualizer.ax.scatter(edges_x, edges_y, c='yellow', s=100, marker='o', 
                                          edgecolors='orange', linewidths=2, label='Detected Edges', zorder=10)
                
                if pad_center:
                    self.visualizer.ax.scatter([pad_center[0]], [pad_center[1]], c='lime', s=200, 
                                              marker='*', edgecolors='green', linewidths=2, 
                                              label='Landing Center', zorder=11)
                    self.visualizer.ax.text(pad_center[0], pad_center[1] + 0.1, 
                                           f'Center\n({pad_center[0]:.2f}, {pad_center[1]:.2f})',
                                           ha='center', fontsize=8, color='green', weight='bold')
                
                self.visualizer.ax.legend(loc='upper right', fontsize=8)
                self.visualizer.render(force=True)
                import matplotlib.pyplot as plt
                plt.pause(0.5)
            
            if pad_center:
                land_on_pad(self.controller.cf.commander, (self.current_x, self.current_y),
                           pad_center, self.flight_height, self.control_rate_hz)
            else:
                print('⚠️  No pad found, landing at current position')
            
            print('🛬 Landing...')
            self.land(duration_s=2.0)
            print('✅ Landed at goal')
            
            # PHASE 4: Wait and takeoff
            print('\n⏸️  Waiting 5 seconds...')
            time.sleep(5.0)
            
            print('🚁 Taking off to return...')
            self.start_hover_timer()
            self.takeoff(duration_s=2.0)
            
            # PHASE 5: Return to start
            print(f'\n🗺️  Planning return to start {self.start}...')
            self.plan_path((self.current_x, self.current_y), self.start)
            
            if self.visualizer:
                self.visualizer.goal = self.start
                self.visualizer.update_path(self.waypoints_x, self.waypoints_y)
            
            print('🚀 Returning to start...')
            self.mission_active = True
            while self.mission_active and not self.emergency_stop:
                self.update_sensors()
                
                # Update map every few steps
                if self.step_counter % 2 == 0:
                    cells_updated = update_map(self.grid_map, self.sensor_readings,
                                              self.current_x, self.current_y, self.current_yaw,
                                              safety_margin=0.05)
                    
                    # Replan if obstacles detected
                    self.replan_if_blocked(cells_updated, self.start)
                
                dist = np.hypot(self.current_x - self.start[0], self.current_y - self.start[1])
                if dist < 0.2:
                    break
                self.navigate_with_avoidance(self.start)
                self.step_counter += 1
                time.sleep(0.01)
            
            if self.emergency_stop:
                raise KeyboardInterrupt('Emergency stop')
            print('✅ Arrived at start')
            
            # PHASE 6: Search and land on pad at start
            print('\n🔍 Searching for landing pad at start...')
            self.detector.peak_positions.clear()  # Reset detector
            self.detector.calculated_center = None
            
            sweep = generate_sweep_pattern(self.start, width=1.0, height=1.0, spacing=0.3)
            
            # Show sweep pattern on visualizer
            if self.visualizer:
                sweep_x = [wp[0] for wp in sweep]
                sweep_y = [wp[1] for wp in sweep]
                self.visualizer.ax.plot(sweep_x, sweep_y, 'm--', linewidth=1, alpha=0.5, label='Sweep Pattern (Start)')
                self.visualizer.ax.scatter(sweep_x, sweep_y, c='magenta', s=30, alpha=0.6, marker='x')
            
            # Execute sweep with live visualization
            self.detector.start_detection(baseline_height=self.flight_height)
            pad_center = None
            
            for i, sweep_target in enumerate(sweep):
                print(f'Sweep waypoint {i+1}/{len(sweep)}: ({sweep_target[0]:.2f}, {sweep_target[1]:.2f})')
                
                # Plan path to sweep waypoint
                self.plan_path((self.current_x, self.current_y), sweep_target, create_new_planner=False)
                if self.visualizer:
                    self.visualizer.update_path(self.waypoints_x, self.waypoints_y)
                
                # Navigate to sweep waypoint with visualization
                self.mission_active = True
                while self.mission_active and not self.emergency_stop:
                    self.update_sensors()
                    self.process_detection()
                    
                    dist = np.hypot(self.current_x - sweep_target[0], self.current_y - sweep_target[1])
                    if dist < 0.1:
                        break
                    
                    self.navigate_with_avoidance(sweep_target)
                    self.step_counter += 1
                    time.sleep(0.01)
                
                # Check if pad detected
                if len(self.detector.peak_positions) >= 2:
                    center = self.detector.calculate_pad_center()
                    if center and self.detector.center_confidence > 0.45:
                        print(f'✅ Pad detected at ({center[0]:.3f}, {center[1]:.3f}), conf={self.detector.center_confidence:.2f}')
                        pad_center = center
                        break
            
            self.detector.stop_detection()
            
            # Show detected edges and center
            if self.visualizer and len(self.detector.peak_positions) > 0:
                edges_x = [edge['position'][0] for edge in self.detector.peak_positions]
                edges_y = [edge['position'][1] for edge in self.detector.peak_positions]
                self.visualizer.ax.scatter(edges_x, edges_y, c='yellow', s=100, marker='o', 
                                          edgecolors='orange', linewidths=2, label='Detected Edges (Start)', zorder=10)
                
                if pad_center:
                    self.visualizer.ax.scatter([pad_center[0]], [pad_center[1]], c='lime', s=200, 
                                              marker='*', edgecolors='green', linewidths=2, 
                                              label='Landing Center (Start)', zorder=11)
                    self.visualizer.ax.text(pad_center[0], pad_center[1] + 0.1, 
                                           f'Center\n({pad_center[0]:.2f}, {pad_center[1]:.2f})',
                                           ha='center', fontsize=8, color='green', weight='bold')
                
                self.visualizer.ax.legend(loc='upper right', fontsize=8)
                self.visualizer.render(force=True)
                import matplotlib.pyplot as plt
                plt.pause(0.5)
            
            if pad_center:
                land_on_pad(self.controller.cf.commander, (self.current_x, self.current_y),
                           pad_center, self.flight_height, self.control_rate_hz)
            else:
                print('⚠️  No pad found, landing at current position')
            
            print('🛬 Landing...')
            self.land(duration_s=2.0)
            print('✅ Landed at start')
            print('\n🎉 Mission complete!')
            
        except KeyboardInterrupt:
            print('\n⚠️  Mission interrupted')
        except Exception as e:
            print(f'\n❌ Error: {e}')
            import traceback
            traceback.print_exc()
        finally:
            self.mission_active = False
            
            # Emergency landing if stopped mid-flight
            if self.emergency_stop and self.current_z > 0.15:
                print('\n🛬 Emergency landing...')
                try:
                    self.land(duration_s=2.0)
                except Exception as e:
                    print(f'Warning: Emergency landing error: {e}')
            
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


def main():
    mission = AutonomousMission(cruise_speed=0.3, 
                                flight_height=0.5, 
                                control_rate_hz=50.0)
    mission.run()


if __name__ == "__main__":
    main()
