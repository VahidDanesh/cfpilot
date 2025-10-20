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

    def _setup_crazyflie_params(self, x: float=0.0, y: float=0.0, z: float=0.0, yaw: float=0.0) -> None:
        """Setup Crazyflie parameters"""
        # Reset estimator
        self.cf.param.set_value('kalman.initialX', x)
        self.cf.param.set_value('kalman.initialY', y) 
        self.cf.param.set_value('kalman.initialZ', z)
        self.cf.param.set_value('kalman.initialYaw', yaw)
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
        self.lg = LogConfig(name='State', period_in_ms=period_ms)
        self.lg.add_variable('stateEstimate.x', 'float')
        self.lg.add_variable('stateEstimate.y', 'float')
        self.lg.add_variable('stateEstimate.z', 'float')
        self.lg.add_variable('stabilizer.yaw', 'float')
        # self.lg.add_variable('stabilizer.roll', 'float')  # packet limit, need to add new logger if more info is needed
        # self.lg.add_variable('stabilizer.pitch', 'float')
        self.lg.add_variable('pm.vbat', 'float')

        try:
            self.cf.log.add_config(self.lg)
            self.lg.data_received_cb.add_callback(self._pos_data_callback)
            self.lg.error_cb.add_callback(self._log_error_callback)
            self.lg.start()
        except KeyError as e:
            self.logger.error(f"Could not start log configuration: {e}")
        except AttributeError:
            self.logger.error("Could not add Position log config, bad configuration.")
        
        self.logger.info("✅ Async logging started.")
    
    def _pos_data_callback(self, timestamp, data, logconf_name):
        """Callback for position data"""
        
        self.latest_data['timestamp'] = timestamp
        self.latest_data.update(data)
        
        # Safety checks
        # self._check_battery_safety(data)
        

        x = data.get('stateEstimate.x', 0)
        y = data.get('stateEstimate.y', 0)
        z = data.get('stateEstimate.z', 0)
        yaw = data.get('stabilizer.yaw', 0)

        
        # Call user callbacks
        for callback in self.data_callbacks:
            try:
                callback(timestamp, data, logconf_name)
            except Exception as e:
                self.logger.error(f"Error in pos callback: {e}")
    
    # def _meas_data_callback(self, timestamp, data, logconf_name):
    #     """Callback for measurement data"""
    #     self.latest_data.update(data)
    #     self.flight_data.append(self.latest_data.copy())
        
        
    #     # Update plotter sensors
    #     if self.plotter:
    #         self._update_plotter_sensors(data)
        
    #     # Call user callbacks
    #     for callback in self.data_callbacks:
    #         try:
    #             callback(timestamp, data, logconf_name)
    #         except Exception as e:
    #             self.logger.error(f"Error in meas callback: {e}")
    
    def _log_error_callback(self, logconf, msg):
        """Callback for log errors"""
        self.logger.error(f"Log error: {msg}")

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

    def connect(self, uri: str, x: float=0.0, y: float=0.0, z: float=0.0, yaw: float=0.0) -> None:
        """Connect to Crazyflie"""
        self.logger.info(f"Connecting to {uri}")
        self.cf.open_link(uri)
        time.sleep(0.5)
        self._setup_crazyflie_params(x, y, z, yaw)
        self._setup_logging_async()

    
    def disconnect(self) -> None:
        """Disconnect from Crazyflie"""
        if self.lg_pos:
            self.lg_pos.stop()
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
    
    def emergency_shutdown(self) -> None:
        """Emergency shutdown procedure"""
        self.emergency_triggered = True
        self.flight_active = False
        if self.safety_timer:
            self.safety_timer.cancel()
        self.cleanup()
        self.disconnect()
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
        self.logger.info(f"💾 Saving flight data to {log_file}")
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