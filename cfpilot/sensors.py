"""
Sensor filtering and processing for Crazyflie multiranger

Provides robust filtering for ToF sensor data with outlier rejection and smoothing.
"""
import numpy as np
from collections import deque
from typing import Dict, Optional
from cflib.utils.multiranger import Multiranger


class Sensor:
    """
    Queue-based sensor filter for multiranger data
    
    Maintains sliding window of N samples per sensor with:
    - Outlier rejection (spike detection)
    - Median filtering (noise reduction)
    - Moving average (signal smoothing)
    """
    
    def __init__(self, window_size: int = 5, max_valid_range: float = 4.0):
        """
        Initialize sensor filter
        
        Args:
            window_size: Number of samples to keep (5-10 for 10Hz)
            max_valid_range: Maximum valid sensor range in meters
        """
        self.window_size = window_size
        self.max_valid_range = max_valid_range
        
        self.queues = {
            'front': deque(maxlen=window_size),
            'back': deque(maxlen=window_size),
            'left': deque(maxlen=window_size),
            'right': deque(maxlen=window_size),
            'up': deque(maxlen=window_size),
            'down': deque(maxlen=window_size)
        }
    
    def _is_valid(self, reading: Optional[float]) -> bool:
        """Check if reading is valid"""
        if reading is None:
            return False
        if reading <= 0.02 or reading > self.max_valid_range:
            return False
        return True
    
    def _detect_spike(self, reading: float, queue: deque, threshold: float = 0.08) -> bool:
        """Detect if reading is a spike compared to recent history"""
        if len(queue) < 2:
            return False
        recent_median = np.median(list(queue))
        return abs(reading - recent_median) > threshold
    
    def update(self, multiranger: Multiranger) -> Dict[str, Optional[float]]:
        """
        Read and filter all sensors
        
        Args:
            multiranger: Multiranger object
            
        Returns:
            Dictionary with filtered distances in meters
        """
        raw_readings = {
            'front': multiranger.front,
            'back': multiranger.back,
            'left': multiranger.left,
            'right': multiranger.right,
            'up': multiranger.up,
            'down': multiranger.down
        }
        
        filtered = {}
        
        for direction, raw_value in raw_readings.items():
            queue = self.queues[direction]
            
            if not self._is_valid(raw_value):
                if len(queue) > 0:
                    filtered[direction] = np.median(list(queue))
                else:
                    filtered[direction] = None
                continue
            
            if self._detect_spike(raw_value, queue):
                if len(queue) > 0:
                    filtered[direction] = np.median(list(queue))
                else:
                    filtered[direction] = raw_value
                    queue.append(raw_value)
                continue
            
            queue.append(raw_value)
            
            if len(queue) >= 3:
                filtered[direction] = np.median(list(queue))
            elif len(queue) >= 1:
                filtered[direction] = np.mean(list(queue))
            else:
                filtered[direction] = None
        
        return filtered
    
    def get_min_distance(self, filtered_readings: Dict[str, Optional[float]]) -> float:
        """Get minimum valid distance from filtered readings"""
        valid = [d for d in filtered_readings.values() if d is not None]
        return min(valid) if valid else float('inf')
    
    def reset(self):
        """Clear all queues"""
        for queue in self.queues.values():
            queue.clear()


def read_multiranger_raw(multiranger: Multiranger) -> Dict[str, Optional[float]]:
    """Read multiranger sensors without filtering"""
    return {
        'front': multiranger.front,
        'back': multiranger.back,
        'left': multiranger.left,
        'right': multiranger.right,
        'up': multiranger.up,
        'down': multiranger.down
    }


def get_min_distance(sensors: Dict[str, Optional[float]]) -> float:
    """Get minimum valid sensor distance"""
    valid = [d for d in sensors.values() if d is not None]
    return min(valid) if valid else float('inf')
