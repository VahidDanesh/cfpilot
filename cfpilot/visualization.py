"""
Real-time Point Cloud Visualization for Multiranger Data

Standalone visualization module for plotting multiranger sensor data in 3D.
"""

import math
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json


# try:
#     from vispy import scene
#     from vispy.scene import visuals
#     from vispy.scene.cameras import TurntableCamera
#     from PyQt5 import QtCore, QtWidgets
#     VISUALIZATION_AVAILABLE = True
#     print("✅ Visualization dependencies available")
# except ImportError:
#     VISUALIZATION_AVAILABLE = False
#     print("❌ Visualization dependencies not available")


# class PointCloudPlotter:
#     """Real-time 3D point cloud plotter for multiranger sensor data"""
    
#     def __init__(self, sensor_threshold: int = 2000):
#         if not VISUALIZATION_AVAILABLE:
#             raise ImportError("Visualization dependencies not available. Install with: pip install vispy PyQt5")
        
#         self.logger = logging.getLogger(__name__)
#         self.sensor_threshold = sensor_threshold
#         self.running = False
#         self.app = None
#         self.window = None
#         self.canvas = None
#         self.owns_app = False
#         self.thread = None
        
#         # Data storage
#         self.last_position = [0, 0, 0]
#         self.position_data = np.array([0, 0, 0], ndmin=2)
#         self.measurement_data = np.array([0, 0, 0], ndmin=2)
    
#     def start(self) -> None:
#         """Start the visualization with simple window creation"""
#         if self.running:
#             return
        
#         try:
#             # Import Qt in the main thread
#             from PyQt5 import QtWidgets
            
#             # Check if QApplication already exists
#             app = QtWidgets.QApplication.instance()
#             if app is None:
#                 # Create QApplication but don't run exec_()
#                 self.app = QtWidgets.QApplication([])
#                 self.owns_app = True
#             else:
#                 self.app = app
#                 self.owns_app = False
            
#             # Create and show window
#             self.window = VisualizationWindow(self.sensor_threshold)
#             self.canvas = self.window.canvas
#             self.window.show()
            
#             # Force window to appear by processing events once
#             self.app.processEvents()
            
#             self.running = True
#             self.logger.info("Point cloud plotter window created")
            
#         except Exception as e:
#             self.logger.error(f"Visualization error: {e}")
#             self.running = False
    
#     def stop(self) -> None:
#         """Stop the visualization"""
#         if not self.running:
#             return
        
#         self.running = False
#         if self.window:
#             self.window.close()
#         if self.app and self.owns_app:
#             self.app.quit()
#         self.logger.info("Point cloud plotter stopped")
    
#     def update_position(self, x: float, y: float, z: float) -> None:
#         """Update drone position"""
#         if not self.running or not self.canvas:
#             return
        
#         self.last_position = [x, y, z]
#         if hasattr(self.canvas, 'set_position'):
#             self.canvas.set_position([x, y, z])
        
#         # Process Qt events to update the display
#         if self.app:
#             self.app.processEvents()
    
#     def update_sensors(self, measurements: Dict[str, float]) -> None:
#         """Update sensor measurements"""
#         if not self.running or not self.canvas:
#             return
        
#         if hasattr(self.canvas, 'set_measurement'):
#             self.canvas.set_measurement(measurements)
        
#         # Process Qt events to update the display
#         if self.app:
#             self.app.processEvents()
    
#     def process_events(self) -> None:
#         """Process Qt events to keep window responsive"""
#         if self.app and self.running:
#             self.app.processEvents()
    


# class VisualizationWindow(QtWidgets.QMainWindow):
#     """Qt window for 3D visualization"""
    
#     def __init__(self, sensor_threshold: int):
#         super().__init__()
#         self.resize(700, 500)
#         self.setWindowTitle('CFPilot - Multiranger Point Cloud')
        
#         self.canvas = VisualizationCanvas(sensor_threshold)
#         self.canvas.create_native()
#         self.canvas.native.setParent(self)
#         self.setCentralWidget(self.canvas.native)


# class VisualizationCanvas(scene.SceneCanvas):
#     """3D visualization canvas"""
    
#     def __init__(self, sensor_threshold: int):
#         super().__init__(keys=None)
#         self.unfreeze()
#         self.sensor_threshold = sensor_threshold
#         self.size = 800, 600
#         self.view = self.central_widget.add_view()
#         self.view.bgcolor = '#ffffff'
#         self.view.camera = TurntableCamera(fov=10.0, distance=30.0, up='+z', center=(0.0, 0.0, 0.0))
        
#         # Data arrays
#         self.last_pos = [0, 0, 0]
#         self.pos_data = np.array([0, 0, 0], ndmin=2)
#         self.meas_data = np.array([0, 0, 0], ndmin=2)
        
#         # Visual elements
#         self.pos_markers = visuals.Markers()
#         self.meas_markers = visuals.Markers()
#         self.lines = [visuals.Line() for _ in range(6)]
        
#         # Add to scene
#         self.view.add(self.pos_markers)
#         self.view.add(self.meas_markers)
#         for line in self.lines:
#             self.view.add(line)
        
#         scene.visuals.XYZAxis(parent=self.view.scene)
#         self.freeze()
    
#     def set_position(self, pos: List[float]) -> None:
#         """Set drone position"""
#         self.last_pos = pos
#         self.pos_data = np.append(self.pos_data, [pos], axis=0)
#         self.pos_markers.set_data(self.pos_data, face_color='red', size=5)
    
#     def set_measurement(self, measurements: Dict[str, float]) -> None:
#         """Set sensor measurements"""
#         data = self._create_sensor_points(measurements)
        
#         # Update lines
#         for i, line in enumerate(self.lines):
#             if i < len(data):
#                 line.set_data(np.array([self.last_pos, data[i]]))
#             else:
#                 line.set_data(np.array([self.last_pos, self.last_pos]))
        
#         # Update measurement points
#         if data:
#             self.meas_data = np.append(self.meas_data, data, axis=0)
#         self.meas_markers.set_data(self.meas_data, face_color='blue', size=5)
    
#     def _create_sensor_points(self, m: Dict[str, float]) -> List[List[float]]:
#         """Create 3D points from sensor measurements"""
#         data = []
#         o = self.last_pos
#         roll, pitch, yaw = m.get('roll', 0), -m.get('pitch', 0), m.get('yaw', 0)
        
#         # Check each sensor direction
#         sensors = [
#             ('up', [o[0], o[1], o[2] + m.get('up', 8000) / 1000.0]),
#             ('down', [o[0], o[1], o[2] - m.get('down', 8000) / 1000.0]),
#             ('left', [o[0], o[1] + m.get('left', 8000) / 1000.0, o[2]]),
#             ('right', [o[0], o[1] - m.get('right', 8000) / 1000.0, o[2]]),
#             ('front', [o[0] + m.get('front', 8000) / 1000.0, o[1], o[2]]),
#             ('back', [o[0] - m.get('back', 8000) / 1000.0, o[1], o[2]])
#         ]
        
#         for direction, point in sensors:
#             distance = m.get(direction, 8000)
#             if distance < self.sensor_threshold:
#                 rotated_point = self._rotate_point(roll, pitch, yaw, o, point)
#                 data.append(rotated_point)
        
#         return data
    
#     def _rotate_point(self, roll: float, pitch: float, yaw: float, 
#                      origin: List[float], point: List[float]) -> List[float]:
#         """Rotate point around origin"""
#         # Convert to radians
#         r, p, y = map(math.radians, [roll, pitch, yaw])
        
#         # Rotation matrices
#         cos_r, sin_r = math.cos(r), math.sin(r)
#         cos_p, sin_p = math.cos(p), math.sin(p)
#         cos_y, sin_y = math.cos(y), math.sin(y)
        
#         rot_y = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])
#         rot_p = np.array([[cos_p, 0, sin_p], [0, 1, 0], [-sin_p, 0, cos_p]])
#         rot_r = np.array([[1, 0, 0], [0, cos_r, -sin_r], [0, sin_r, cos_r]])
        
#         # Combined rotation
#         rot_matrix = np.dot(np.dot(rot_r, rot_p), rot_y)
        
#         # Apply rotation
#         point_relative = np.subtract(point, origin)
#         rotated_relative = np.dot(rot_matrix, point_relative)
#         return np.add(rotated_relative, origin).tolist()


#!/usr/bin/env python3
"""
Flight Data Visualization Tool
Plots flight data from Crazyflie landing pad detection missions
"""


class DroneNavigationVisualizer:
    """
    Real-time 2D navigation visualization for Crazyflie
    
    Works with both simulation and real hardware:
    - Simulation: Pass sensor readings from MockMultiranger
    - Real Hardware: Pass sensor readings from cflib.utils.multiranger.Multiranger
    
    Usage with simulation:
        visualizer = DroneNavigationVisualizer(xlim=(0, 5), ylim=(0, 3))
        visualizer.setup(start, goal, true_obstacles=obstacles, grid_map=grid_map)
        
        # In control loop:
        visualizer.update_drone(x, y, yaw)
        visualizer.update_sensors({'front': sensor.front, 'back': sensor.back, ...})
        visualizer.update_path(waypoints_x, waypoints_y)
        visualizer.render()
    
    Usage with real hardware:
        visualizer = DroneNavigationVisualizer(xlim=(-2, 2), ylim=(-2, 2))
        visualizer.setup(start=(0, 0), goal=(1.5, 1.5))
        
        # In controller callback:
        def data_callback(timestamp, data, logconf_name):
            x = data.get('stateEstimate.x', 0)
            y = data.get('stateEstimate.y', 0)
            yaw = data.get('stabilizer.yaw', 0)
            visualizer.update_drone(x, y, yaw)
            
            # With Multiranger sensor:
            sensors = {
                'front': multiranger.front,
                'back': multiranger.back,
                'left': multiranger.left,
                'right': multiranger.right
            }
            visualizer.update_sensors(sensors)
            visualizer.render()
    """
    
    def __init__(self, xlim=(0, 5), ylim=(0, 3), figsize=(14, 9), animation_speed=0.01):
        """
        Initialize visualizer
        
        Args:
            xlim: X-axis limits (min, max) in meters
            ylim: Y-axis limits (min, max) in meters
            figsize: Figure size (width, height) in inches
            animation_speed: Pause time between frames (seconds)
        """
        self.xlim = xlim
        self.ylim = ylim
        self.figsize = figsize
        self.animation_speed = animation_speed
        
        # Matplotlib elements
        self.fig = None
        self.ax = None
        self.is_setup = False
        
        # Plot elements
        self.drone_pos = None
        self.trajectory_line = None
        self.path_line = None
        self.sensor_lines = []
        self.avoidance_arrows = []
        self.discovered_map = None
        self.replan_markers = None
        self.detected_edges_scatter = None
        self.pad_center_marker = None
        
        # State tracking
        self.trajectory_x = []
        self.trajectory_y = []
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.current_sensors = {}
        self.repulsion_force = (0.0, 0.0)
        self.replans = 0
        self.replan_positions = []
        self.step_counter = 0
        self.status_flags = {'stuck': False, 'emergency': False, 'avoiding': False}
        
        # Landing pad detection
        self.detected_edges = []
        self.pad_center = None
        
        # Map and environment
        self.grid_map = None
        self.true_obstacles = []
        self.start_pos = None
        self.goal_pos = None
    
    def setup(self, start, goal, true_obstacles=None, grid_map=None):
        """
        Setup visualization with initial environment
        
        Args:
            start: (x, y) tuple for start position
            goal: (x, y) tuple for goal position
            true_obstacles: List of (x, y, radius) tuples (optional, for simulation)
            grid_map: GridMap instance (optional, for discovered obstacles)
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        
        if self.is_setup:
            print("⚠️  Visualizer already setup. Call reset() first to reinitialize.")
            return
        
        # Store environment
        self.start_pos = start
        self.goal_pos = goal
        self.true_obstacles = true_obstacles if true_obstacles else []
        self.grid_map = grid_map
        
        # Reset trajectory
        self.trajectory_x = [start[0]]
        self.trajectory_y = [start[1]]
        self.current_x = start[0]
        self.current_y = start[1]
        
        # Create figure
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        
        # Plot true obstacles (if provided, for simulation)
        if self.true_obstacles:
            for obs_x, obs_y, radius in self.true_obstacles:
                circle = Circle((obs_x, obs_y), radius, color='red', alpha=0.15,
                               linewidth=1, edgecolor='red', linestyle='--',
                               label='True obstacles (unknown)' if obs_x == self.true_obstacles[0][0] else '')
                self.ax.add_patch(circle)
        
        # Plot start and goal
        self.ax.plot(start[0], start[1], 'g^', markersize=15, label='Start', zorder=10)
        self.ax.plot(goal[0], goal[1], 'm*', markersize=20, label='Goal', zorder=10)
        
        # Initialize plot elements
        self.trajectory_line, = self.ax.plot([], [], 'g-', linewidth=2, label='Trajectory', alpha=0.7)
        self.drone_pos, = self.ax.plot([], [], 'bo', markersize=12, label='Drone', zorder=11)
        self.path_line, = self.ax.plot([], [], 'b--', alpha=0.4, linewidth=1, label='Planned Path')
        
        # Configure axes
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_title('Real-time Navigation - Initializing...')
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right', fontsize=9)
        self.ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.pause(0.1)
        
        self.is_setup = True
        print("✅ Visualizer setup complete")
    
    def update_drone(self, x, y, yaw=0.0):
        """
        Update drone position and orientation
        
        Args:
            x: X position in meters
            y: Y position in meters
            yaw: Yaw angle in radians (optional)
        """
        self.current_x = x
        self.current_y = y
        self.current_yaw = yaw
        
        # Add to trajectory
        self.trajectory_x.append(x)
        self.trajectory_y.append(y)
        self.step_counter += 1
    
    def update_sensors(self, sensors):
        """
        Update sensor readings (compatible with Multiranger API)
        
        Args:
            sensors: Dict with keys 'front', 'back', 'left', 'right'
                     Values are distances in meters or None
        """
        self.current_sensors = sensors.copy()
    
    def update_path(self, waypoints_x, waypoints_y):
        """
        Update planned path waypoints
        
        Args:
            waypoints_x: List of X coordinates
            waypoints_y: List of Y coordinates
        """
        if not self.is_setup:
            return
        
        self.path_line.set_data(waypoints_x, waypoints_y)
    
    def update_repulsion(self, rep_x, rep_y):
        """
        Update repulsion force vector for visualization
        
        Args:
            rep_x: Repulsion force in X direction
            rep_y: Repulsion force in Y direction
        """
        self.repulsion_force = (rep_x, rep_y)
    
    def update_detected_edges(self, edge_positions, pad_center=None):
        """
        Update detected landing pad edges and calculated center
        
        Args:
            edge_positions: List of (x, y) tuples representing detected edges
            pad_center: (x, y) tuple for calculated pad center (optional)
        """
        self.detected_edges = edge_positions if edge_positions else []
        self.pad_center = pad_center
    
    def mark_replan(self, x=None, y=None):
        """
        Mark a replanning event at current or specified position
        
        Args:
            x: X position (uses current position if None)
            y: Y position (uses current position if None)
        """
        x = x if x is not None else self.current_x
        y = y if y is not None else self.current_y
        
        self.replans += 1
        self.replan_positions.append((x, y))
    
    def set_status(self, stuck=False, emergency=False, avoiding=False):
        """
        Set status flags for display
        
        Args:
            stuck: True if drone is stuck
            emergency: True if emergency stop is active
            avoiding: True if actively avoiding obstacles
        """
        self.status_flags = {
            'stuck': stuck,
            'emergency': emergency,
            'avoiding': avoiding
        }
    
    def render(self, force=False):
        """
        Render current visualization state
        
        Args:
            force: If True, render every call. If False, render every 5 steps (for performance)
        """
        if not self.is_setup:
            return
        
        # Render every 5 steps for performance (unless forced)
        if not force and self.step_counter % 5 != 0:
            return
        
        # Update drone position
        self.drone_pos.set_data([self.current_x], [self.current_y])
        
        # Update trajectory
        self.trajectory_line.set_data(self.trajectory_x, self.trajectory_y)
        
        # Draw sensor ranges
        for line in self.sensor_lines:
            line.remove()
        self.sensor_lines = []
        
        for direction, distance in self.current_sensors.items():
            if distance is not None and distance < 2.0:
                angle_offset = {'front': 0, 'right': -np.pi/2,
                               'back': np.pi, 'left': np.pi/2}.get(direction, 0)
                angle = self.current_yaw + angle_offset
                end_x = self.current_x + distance * np.cos(angle)
                end_y = self.current_y + distance * np.sin(angle)
                
                # Color code: red if close, yellow if moderate distance
                color = 'red' if distance < 0.4 else 'orange' if distance < 0.6 else 'yellow'
                alpha = 0.7 if distance < 0.4 else 0.5 if distance < 0.6 else 0.3
                
                line = self.ax.plot([self.current_x, end_x], [self.current_y, end_y],
                                   color=color, alpha=alpha, linewidth=2, zorder=5)[0]
                self.sensor_lines.append(line)
        
        # Clear old avoidance arrows
        for arrow in self.avoidance_arrows:
            arrow.remove()
        self.avoidance_arrows = []
        
        # Draw repulsion force if active
        rep_x, rep_y = self.repulsion_force
        rep_mag = np.hypot(rep_x, rep_y)
        if rep_mag > 0.05:
            arrow = self.ax.arrow(self.current_x, self.current_y, rep_x * 0.4, rep_y * 0.4,
                                 head_width=0.08, head_length=0.08,
                                 fc='orange', ec='darkorange',
                                 linewidth=2, alpha=0.8, zorder=8)
            self.avoidance_arrows.append(arrow)
        
        # Update discovered map (if available)
        if self.grid_map is not None:
            if self.discovered_map is not None:
                self.discovered_map.remove()
            
            # Create heatmap of discovered obstacles
            extent = [self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]]
            self.discovered_map = self.ax.imshow(self.grid_map.data.T, cmap='Blues',
                                                alpha=0.6, vmin=0, vmax=1,
                                                origin='lower', extent=extent, zorder=1)
        
        # Draw detected landing pad edges
        if self.detected_edges_scatter is not None:
            self.detected_edges_scatter.remove()
            self.detected_edges_scatter = None
        
        if self.detected_edges and len(self.detected_edges) > 0:
            edge_x = [pos[0] for pos in self.detected_edges]
            edge_y = [pos[1] for pos in self.detected_edges]
            self.detected_edges_scatter = self.ax.scatter(edge_x, edge_y, 
                                                         c='red', marker='x', s=100, 
                                                         linewidths=2, zorder=12,
                                                         label=f'Pad Edges ({len(self.detected_edges)})')
        
        # Draw calculated pad center
        if self.pad_center_marker is not None:
            self.pad_center_marker.remove()
            self.pad_center_marker = None
        
        if self.pad_center is not None:
            self.pad_center_marker = self.ax.scatter([self.pad_center[0]], [self.pad_center[1]], 
                                                     c='lime', marker='*', s=400, 
                                                     edgecolors='darkgreen', linewidths=2,
                                                     zorder=13, label='Pad Center')
        
        # Update title with status
        status = f'Step {self.step_counter} | Replans: {self.replans}'
        if self.status_flags['stuck']:
            status += ' | ⚠️ STUCK'
        elif self.status_flags['emergency']:
            status += ' | 🛑 EMERGENCY'
        elif self.status_flags['avoiding']:
            status += ' | 🔄 AVOIDING'
        self.ax.set_title(status)
        
        # Update legend if detection elements are present
        if self.detected_edges or self.pad_center:
            self.ax.legend(loc='upper right', fontsize=9)
        
        plt.pause(self.animation_speed)
    
    def finalize(self, save_path='navigation_result.png'):
        """
        Finalize visualization and save result
        
        Args:
            save_path: Path to save final image
        """
        if not self.is_setup:
            return
        
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        
        plt.ioff()  # Disable interactive mode
        
        # Final update
        self.drone_pos.set_data([self.current_x], [self.current_y])
        self.trajectory_line.set_data(self.trajectory_x, self.trajectory_y)
        
        # Add replan markers
        if self.replan_positions:
            rx, ry = zip(*self.replan_positions)
            self.ax.plot(rx, ry, 'ro', markersize=12, label=f'Replans ({self.replans})', zorder=9)
        
        # Redraw true obstacles more visible (if simulation)
        if self.true_obstacles:
            for obs_x, obs_y, radius in self.true_obstacles:
                circle = Circle((obs_x, obs_y), radius, color='red', alpha=0.3,
                               linewidth=2, edgecolor='red', linestyle='-')
                self.ax.add_patch(circle)
        
        self.ax.set_title(f'Final Result (Steps: {self.step_counter}, Replans: {self.replans})')
        self.ax.legend(loc='upper right')
        plt.tight_layout()
        
        print(f"\n📈 Saving final visualization to {save_path}...")
        plt.savefig(save_path, dpi=150)
        print(f"💾 Saved to {save_path}")
        plt.show()
    
    def reset(self):
        """Reset visualizer state"""
        import matplotlib.pyplot as plt
        
        if self.fig is not None:
            plt.close(self.fig)
        
        self.fig = None
        self.ax = None
        self.is_setup = False
        self.trajectory_x = []
        self.trajectory_y = []
        self.step_counter = 0
        self.replans = 0
        self.replan_positions = []
        self.sensor_lines = []
        self.avoidance_arrows = []
        self.discovered_map = None
        self.detected_edges = []
        self.pad_center = None
        self.detected_edges_scatter = None
        self.pad_center_marker = None
        
        print("✅ Visualizer reset")


class FlightDataPlotter:
    """Visualize flight data and landing pad detection results"""
    
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.data = None
        self.detection_events = []
        
    def load_data(self):
        """Load flight data from CSV log file"""
        try:
            self.data = pd.read_csv(self.log_file)
            print(f"✅ Loaded {len(self.data)} data points from {self.log_file}")
            print(f"📊 Columns: {list(self.data.columns)}")
            
            # Convert timestamp if exists
            if 'timestamp' in self.data.columns:
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'] - self.data['timestamp'].iloc[0], unit='ms')
                self.data['time_elapsed'] = (self.data['timestamp'] - self.data['timestamp'].iloc[0]).dt.total_seconds()
            else:
                # Create time index based on logging frequency
                self.data['time_elapsed'] = np.arange(len(self.data)) * 0.1  # 100ms logging
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
        return True
    
    def plot_flight_trajectory(self, save_path: str = None):
        """Plot 3D flight trajectory"""
        if self.data is None:
            print("❌ No data loaded")
            return
            
        fig = plt.figure(figsize=(15, 10))
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(221, projection='3d')
        
        x = self.data.get('stateEstimate.x', np.zeros(len(self.data)))
        y = self.data.get('stateEstimate.y', np.zeros(len(self.data)))
        z = self.data.get('stateEstimate.z', np.zeros(len(self.data)))
        
        # Color trajectory by time
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.data)))
        ax1.scatter(x, y, z, c=colors, s=1, alpha=0.6)
        ax1.plot(x, y, z, 'b-', alpha=0.3, linewidth=0.5)
        
        # Mark start and end
        ax1.scatter(x.iloc[0], y.iloc[0], z.iloc[0], c='green', s=100, marker='o', label='Start')
        ax1.scatter(x.iloc[-1], y.iloc[-1], z.iloc[-1], c='red', s=100, marker='x', label='End')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Flight Trajectory')
        ax1.legend()
        
        # 2D top view
        ax2 = fig.add_subplot(222)
        ax2.plot(x, y, 'b-', alpha=0.7, linewidth=1)
        ax2.scatter(x.iloc[0], y.iloc[0], c='green', s=100, marker='o', label='Start')
        ax2.scatter(x.iloc[-1], y.iloc[-1], c='red', s=100, marker='x', label='End')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Flight Path (Top View)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_aspect(aspect='equal', adjustable='datalim')
        
        # Height over time
        ax3 = fig.add_subplot(223)
        # remove high values (noise)
        z = np.where(z > 1, np.nan, z)
        ax3.plot(self.data['time_elapsed'], z, 'g-', linewidth=1, label='Height')
        
        # Add z-range sensor if available
        if 'range.zrange' in self.data.columns:
            zrange = self.data['range.zrange'] / 1000.0  # Convert mm to m
            ax3.plot(self.data['time_elapsed'], zrange, 'r-', linewidth=1, alpha=0.7, label='Z-Range Sensor')
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Height (m)')
        ax3.set_title('Height vs Time')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Battery voltage
        ax4 = fig.add_subplot(224)
        if 'pm.vbat' in self.data.columns:
            ax4.plot(self.data['time_elapsed'], self.data['pm.vbat'], 'orange', linewidth=1)
            ax4.axhline(y=3.2, color='red', linestyle='--', alpha=0.7, label='Low Battery')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Battery (V)')
            ax4.set_title('Battery Voltage')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Flight trajectory saved to {save_path}")
        
        plt.show()
    
    def plot_landing_pad_detection(self, save_path: str = None):
        """Plot height data and detected landing pad events"""
        if self.data is None:
            print("❌ No data loaded")
            return
            
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        time = self.data['time_elapsed']
        
        # Raw height vs z-range
        ax1.plot(time, self.data.get('stateEstimate.z', np.zeros(len(self.data))), 
                'b-', linewidth=1, label='State Estimate Z', alpha=0.8)
        
        if 'range.zrange' in self.data.columns:
            zrange = self.data['range.zrange'] / 1000.0  # Convert mm to m
            ax1.plot(time, zrange, 'r-', linewidth=1, label='Z-Range Sensor', alpha=0.8)
            
            # Calculate relative height (assuming baseline at start)
            baseline = np.median(zrange[:50])  # First 50 readings as baseline
            relative_height = zrange - baseline
            
            # Mark potential landing pad detections (height > 5cm above baseline)
            pad_threshold = 0.05
            pad_detections = relative_height > pad_threshold
            
            if np.any(pad_detections):
                detection_times = time[pad_detections]
                detection_heights = zrange[pad_detections]
                ax1.scatter(detection_times, detection_heights, 
                           c='yellow', s=30, alpha=0.8, label='Potential Pad Detections', zorder=5)
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Height (m)')
        ax1.set_title('Height Measurements During Flight')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Relative height analysis
        if 'range.zrange' in self.data.columns:
            ax2.plot(time, relative_height, 'purple', linewidth=1, label='Height Above Baseline')
            ax2.axhline(y=pad_threshold, color='red', linestyle='--', alpha=0.7, 
                       label=f'Detection Threshold ({pad_threshold*100:.0f}cm)')
            ax2.fill_between(time, 0, relative_height, where=(relative_height > pad_threshold), 
                           color='yellow', alpha=0.3, label='Above Threshold')
            
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Relative Height (m)')
            ax2.set_title('Landing Pad Detection Analysis')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # Position during detection
        x = self.data.get('stateEstimate.x', np.zeros(len(self.data)))
        y = self.data.get('stateEstimate.y', np.zeros(len(self.data)))
        
        ax3.plot(x, y, 'b-', alpha=0.6, linewidth=1, label='Flight Path')
        ax3.scatter(x.iloc[0], y.iloc[0], c='green', s=100, marker='o', label='Start')
        ax3.scatter(x.iloc[-1], y.iloc[-1], c='red', s=100, marker='x', label='End')
        
        # Mark detection positions
        if 'range.zrange' in self.data.columns and np.any(pad_detections):
            detection_x = x[pad_detections]
            detection_y = y[pad_detections]
            ax3.scatter(detection_x, detection_y, c='yellow', s=50, alpha=0.8, 
                       label='Detection Positions', zorder=5)
        
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_title('Detection Positions (Top View)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.axis('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Landing pad analysis saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, save_path: str = None):
        """Generate flight statistics report"""
        if self.data is None:
            print("❌ No data loaded")
            return
            
        report = {
            'flight_summary': {
                'duration': float(self.data['time_elapsed'].iloc[-1]),
                'data_points': len(self.data),
                'logging_frequency': len(self.data) / self.data['time_elapsed'].iloc[-1]
            },
            'position_stats': {
                'max_x': float(self.data.get('stateEstimate.x', pd.Series([0])).max()),
                'min_x': float(self.data.get('stateEstimate.x', pd.Series([0])).min()),
                'max_y': float(self.data.get('stateEstimate.y', pd.Series([0])).max()),
                'min_y': float(self.data.get('stateEstimate.y', pd.Series([0])).min()),
                'max_height': float(self.data.get('stateEstimate.z', pd.Series([0])).max()),
                'min_height': float(self.data.get('stateEstimate.z', pd.Series([0])).min())
            }
        }
        
        # Battery analysis
        if 'pm.vbat' in self.data.columns:
            report['battery'] = {
                'start_voltage': float(self.data['pm.vbat'].iloc[0]),
                'end_voltage': float(self.data['pm.vbat'].iloc[-1]),
                'min_voltage': float(self.data['pm.vbat'].min()),
                'voltage_drop': float(self.data['pm.vbat'].iloc[0] - self.data['pm.vbat'].iloc[-1])
            }
        
        # Landing pad detection analysis
        if 'range.zrange' in self.data.columns:
            zrange = self.data['range.zrange'] / 1000.0
            baseline = np.median(zrange[:50])
            relative_height = zrange - baseline
            pad_detections = np.sum(relative_height > 0.05)
            
            report['landing_pad_detection'] = {
                'baseline_height': float(baseline),
                'max_relative_height': float(relative_height.max()),
                'detection_events': int(pad_detections),
                'detection_percentage': float(pad_detections / len(self.data) * 100)
            }
        
        print("\n📊 FLIGHT REPORT")
        print("=" * 50)
        print(f"Flight Duration: {report['flight_summary']['duration']:.1f}s")
        print(f"Data Points: {report['flight_summary']['data_points']}")
        print(f"Logging Rate: {report['flight_summary']['logging_frequency']:.1f} Hz")
        
        if 'battery' in report:
            print(f"\nBattery: {report['battery']['start_voltage']:.2f}V → {report['battery']['end_voltage']:.2f}V")
            print(f"Voltage Drop: {report['battery']['voltage_drop']:.2f}V")
        
        if 'landing_pad_detection' in report:
            print(f"\nLanding Pad Detection:")
            print(f"  Baseline Height: {report['landing_pad_detection']['baseline_height']:.3f}m")
            print(f"  Max Height Above Baseline: {report['landing_pad_detection']['max_relative_height']:.3f}m")
            print(f"  Detection Events: {report['landing_pad_detection']['detection_events']}")
            print(f"  Detection Rate: {report['landing_pad_detection']['detection_percentage']:.1f}%")
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📄 Report saved to {save_path}")
        
        return report

