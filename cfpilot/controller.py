"""
Core Crazyflie Controller - Asynchronous API

Main controller class for autonomous Crazyflie missions with async architecture
and comprehensive safety features.
"""

import time
import yaml
import signal
import sys
import logging
import csv
import threading
from pathlib import Path
from datetime import datetime
from queue import Queue, Empty
from typing import Optional, Tuple, Dict, Any, Callable

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.positioning.position_hl_commander import PositionHlCommander
from cflib.positioning.motion_commander import MotionCommander
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper
from cflib.utils.multiranger import Multiranger

from .detection import LandingPadDetector, SearchPattern


class CrazyflieController:
    """Main controller for autonomous Crazyflie missions - Async API"""
    
    def __init__(self, config_path: Optional[str] = None, enable_plotting: bool = False):
        """Initialize the Crazyflie controller"""
        self._setup_logging()
        self.config = self._load_config(config_path)

        self.cf = Crazyflie(ro_cache=None, rw_cache='cache')
        
        # Flight state
        self.is_connected = False
        self.emergency_triggered = False
        self.flight_active = False
        
        # Data storage
        self.flight_data = []
        self.latest_data = {}
        
        # Logging
        self.lg_pos = None
        self.lg_meas = None
        
        # Landing pad detection
        self.landing_detector = LandingPadDetector()
        self.search_pattern = SearchPattern()
        
        # Plotting
        self.enable_plotting = enable_plotting
        self.plotter = None
        
        # Safety
        self.safety_timer = None
        
        # Callbacks
        self.data_callbacks = []
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Setup Crazyflie callbacks
        self._setup_callbacks()
        
        self.logger.info("Crazyflie Controller initialized (Async)")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if config_path is None:
            script_dir = Path(__file__).parent
            config_path = script_dir / "config" / "flight_config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"Config file not found: {config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing config: {e}")
            sys.exit(1)
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals"""
        self.logger.warning(f"Signal {signum} received - initiating emergency shutdown")
        self.emergency_shutdown()
    
    def _setup_callbacks(self) -> None:
        """Setup Crazyflie connection callbacks"""
        self.cf.connected.add_callback(self._connected)
        self.cf.disconnected.add_callback(self._disconnected)
        self.cf.connection_failed.add_callback(self._connection_failed)
        self.cf.connection_lost.add_callback(self._connection_lost)
    
    def _connected(self, link_uri):
        """Called when Crazyflie is connected"""
        self.is_connected = True
        self.logger.info(f"🔌 Connected to {link_uri} through callback")

    
    def _disconnected(self, link_uri):
        """Called when Crazyflie is disconnected"""
        self.is_connected = False
        self.logger.info(f"❌ Disconnected from {link_uri}")
    
    def _connection_failed(self, link_uri, msg):
        """Called when connection fails"""
        self.logger.error(f"❌ Connection to {link_uri} failed: {msg}")
        self.is_connected = False
    
    def _connection_lost(self, link_uri, msg):
        """Called when connection is lost"""
        self.logger.warning(f"❌ Connection to {link_uri} lost: {msg}")
        self.is_connected = False
    
    def _setup_crazyflie_params(self) -> None:
        """Setup Crazyflie parameters"""
        # Reset estimator
        self.cf.param.set_value('kalman.initialX', '0')
        self.cf.param.set_value('kalman.initialY', '0') 
        self.cf.param.set_value('kalman.initialZ', '0')
        self.cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        self.cf.param.set_value('kalman.resetEstimation', '0')
        
        # Arm
        self.cf.platform.send_arming_request(True)
        self.logger.info("✅ Crazyflie setup completed")
    
    def _setup_logging_async(self) -> None:
        """Setup async logging with separate configs"""
        period_ms = self.config['logging']['period_ms']
        
        # Position logger
        self.lg_pos = LogConfig(name='Position', period_in_ms=period_ms)
        self.lg_pos.add_variable('stateEstimate.x', 'float')
        self.lg_pos.add_variable('stateEstimate.y', 'float')
        self.lg_pos.add_variable('stateEstimate.z', 'float')
        self.lg_pos.add_variable('pm.vbat', 'float')
        
        # Measurement logger
        self.lg_meas = LogConfig(name='Meas', period_in_ms=period_ms)
        self.lg_meas.add_variable('stabilizer.roll', 'float')
        self.lg_meas.add_variable('stabilizer.pitch', 'float')
        self.lg_meas.add_variable('stabilizer.yaw', 'float')
        self.lg_meas.add_variable('range.zrange', 'uint16_t')
        self.lg_meas.add_variable('range.up', 'uint16_t')
        self.lg_meas.add_variable('range.front', 'uint16_t')
        self.lg_meas.add_variable('range.back', 'uint16_t') 
        self.lg_meas.add_variable('range.left', 'uint16_t')
        self.lg_meas.add_variable('range.right', 'uint16_t')

        try:
            self.cf.log.add_config(self.lg_pos)
            self.lg_pos.data_received_cb.add_callback(self._pos_data_callback)
            self.lg_pos.error_cb.add_callback(self._log_error_callback)

            self.lg_pos.start()
        except KeyError as e:
            self.logger.error(f"Could not start log configuration: {e}")
        except AttributeError:
            self.logger.error("Could not add Position log config, bad configuration.")



        try:
            self.cf.log.add_config(self.lg_meas)
            self.lg_meas.data_received_cb.add_callback(self._meas_data_callback)
            self.lg_meas.error_cb.add_callback(self._log_error_callback)

            self.lg_meas.start()
        except KeyError as e:
            self.logger.error(f"Could not start log configuration: {e}")
        except AttributeError:
            self.logger.error("Could not add Measurement log config, bad configuration.")


        
        # Add to Crazyflie and start
        
        
        self.logger.info("✅ Async logging started.")
    
    def _pos_data_callback(self, timestamp, data, logconf_name):
        """Callback for position data"""
        
        self.latest_data['timestamp'] = timestamp
        self.latest_data.update(data)
        
        # Safety checks
        # self._check_battery_safety(data)
        
        # Landing detection if we have position
        if 'range.zrange' in self.latest_data:
            height = self.latest_data.get('range.zrange', 8000)
            x = data.get('stateEstimate.x', 0)
            y = data.get('stateEstimate.y', 0)
            z = data.get('stateEstimate.z', 0)
            self.landing_detector.process_height_measurement(height, (x, y, z))
        
        # Update plotter position
        if self.plotter:
            self._update_plotter_position(data)
        
        # Call user callbacks
        for callback in self.data_callbacks:
            try:
                callback(timestamp, data, logconf_name)
            except Exception as e:
                self.logger.error(f"Error in pos callback: {e}")
    
    def _meas_data_callback(self, timestamp, data, logconf_name):
        """Callback for measurement data"""
        self.latest_data.update(data)
        self.flight_data.append(self.latest_data.copy())
        
        
        # Update plotter sensors
        if self.plotter:
            self._update_plotter_sensors(data)
        
        # Call user callbacks
        for callback in self.data_callbacks:
            try:
                callback(timestamp, data, logconf_name)
            except Exception as e:
                self.logger.error(f"Error in meas callback: {e}")
    
    def _log_error_callback(self, logconf, msg):
        """Callback for log errors"""
        self.logger.error(f"Log error: {msg}")
    
    def _update_plotter_position(self, data: Dict[str, Any]) -> None:
        """Update plotter position"""
        if not self.plotter:
            return
        
        x = data.get('stateEstimate.x', 0)
        y = data.get('stateEstimate.y', 0) 
        z = data.get('stateEstimate.z', 0)
        self.plotter.update_position(x, y, z)
        self.plotter.process_events()
    
    def _update_plotter_sensors(self, data: Dict[str, Any]) -> None:
        """Update plotter sensors"""
        if not self.plotter:
            return
        
        if 'range.front' in data:
            sensor_data = {
                'roll': data.get('stabilizer.roll', 0),
                'pitch': data.get('stabilizer.pitch', 0),
                'yaw': data.get('stabilizer.yaw', 0),
                'front': data.get('range.front', 8000),
                'back': data.get('range.back', 8000),
                'left': data.get('range.left', 8000),
                'right': data.get('range.right', 8000),
                'up': data.get('range.up', 8000),
                'down': data.get('range.zrange', 8000)
            }
            self.plotter.update_sensors(sensor_data)
            self.plotter.process_events()
    
    def _check_battery_safety(self, data: Dict[str, Any]) -> None:
        """Check battery safety from log data"""
        battery_voltage = data.get('pm.vbat', 0)
        if battery_voltage < self.config['safety']['battery_threshold'] and battery_voltage > 0:
            self.logger.warning(f"🔋 Low battery: {battery_voltage:.2f}V")
            self.emergency_shutdown()
    
    def add_data_callback(self, callback: Callable[[int, Dict[str, Any], str], None]) -> None:
        """Add a data callback function"""
        self.data_callbacks.append(callback)
    
    def remove_data_callback(self, callback: Callable[[int, Dict[str, Any], str], None]) -> None:
        """Remove a data callback function"""
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)
    
    def connect(self, uri: str) -> None:
        """Connect to Crazyflie"""
        self.logger.info(f"Connecting to {uri}")
        self.cf.open_link(uri)
        self._setup_crazyflie_params()
        self._setup_logging_async()

    
    def disconnect(self) -> None:
        """Disconnect from Crazyflie"""
        if self.lg_pos:
            self.lg_pos.stop()
        if self.lg_meas:
            self.lg_meas.stop()
        self.cf.close_link()
    
    def wait_for_connection(self, timeout: float = 10.0) -> bool:
        """Wait for connection to establish"""
        start_time = time.time()
        while not self.is_connected and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        return self.is_connected
    
    def get_latest_data(self) -> Dict[str, Any]:
        """Get latest flight data"""
        return self.latest_data.copy()
    
    def setup_safety_timer(self) -> None:
        """Setup safety timer for emergency shutdown"""
        self.safety_timer = threading.Timer(
            self.config['safety']['max_flight_time'], 
            self.emergency_shutdown
        )
        self.safety_timer.start()
    
    def setup_plotter(self) -> None:
        """Setup 3D point cloud plotter if enabled"""
        if not self.enable_plotting:
            return
        
        try:
            from .visualization import PointCloudPlotter
            self.plotter = PointCloudPlotter()
            self.plotter.start()
            self.logger.info("3D visualization started")
        except ImportError:
            self.logger.warning("Visualization not available - install vispy and PyQt5")
            self.enable_plotting = False
    
    def emergency_shutdown(self) -> None:
        """Emergency shutdown procedure"""
        self.emergency_triggered = True
        self.flight_active = False
        if self.safety_timer:
            self.safety_timer.cancel()
        self.logger.warning("❌ Emergency shutdown initiated")
    
    def save_flight_data(self) -> None:
        """Save flight data to CSV"""
        if not self.flight_data:
            return

        script_dir = Path(__file__).parent
        log_dir = script_dir.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"flight_log_{timestamp}.csv"
        
        try:
            with open(log_file, 'w', newline='') as csvfile:
                if self.flight_data:
                    fieldnames = self.flight_data[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.flight_data)
            
            self.logger.info(f"🛜 Flight data saved to: {log_file}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save flight data: {e}")
    
    def cleanup(self) -> None:
        """Cleanup resources and save data"""
        if self.safety_timer:
            self.safety_timer.cancel()
        
        if self.plotter:
            self.plotter.stop()
        
        self.save_flight_data()
        self.flight_active = False
        self.logger.info("Controller cleanup completed")