"""
Landing Pad Detection Module

Advanced landing pad detection using z-range sensor and peak detection algorithms.
"""

import math
import numpy as np
import logging
from collections import deque
from typing import List, Tuple, Optional, Dict, Any


class LandingPadDetector:
    """Detects landing pads using z-range sensor height measurements"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Detection parameters (configured via config)
        self.lag = 8                    
        self.threshold = 1.5            
        self.influence = 0.2            
        self.min_peak_height = 0.03     
        
        # Data storage
        self.height_data = deque(maxlen=1000)
        self.position_data = deque(maxlen=1000)
        self.peak_positions = []
        self.flight_direction = None
        
        # Detection state
        self.detection_active = False
        self.baseline_height = None
        self.running_mean = deque(maxlen=self.lag)
        self.running_std = deque(maxlen=self.lag)
        
        # Results
        self.calculated_center = None
        self.center_confidence = 0.0
        
        self.logger.info("Landing pad detector initialized")
    
    def configure_detection(self, params: Dict[str, Any]) -> None:
        """Configure detection parameters from config"""
        self.lag = params.get('lag', 8)
        self.threshold = params.get('threshold', 1.5)
        self.influence = params.get('influence', 0.2)
        self.min_peak_height = params.get('min_peak_height', 0.03)
        
        # Reset running statistics
        self.running_mean = deque(maxlen=self.lag)
        self.running_std = deque(maxlen=self.lag)
        
        self.logger.info(f"Detection configured: lag={self.lag}, threshold={self.threshold}, "
                        f"influence={self.influence}, min_height={self.min_peak_height}")
    
    def start_detection(self, baseline_height: float=0.5) -> None:
        """Start the landing pad detection process"""
        self.detection_active = True
        self.height_data.clear()
        self.position_data.clear()
        self.peak_positions.clear()
        self.baseline_height = baseline_height
        self.calculated_center = None
        self.center_confidence = 0.0
        
        self.logger.info("Landing pad detection started")
    
    def stop_detection(self) -> None:
        """Stop the detection process"""
        self.detection_active = False
        self.logger.info(f"Detection stopped. Found {len(self.peak_positions)} border points")
    
    def set_flight_direction(self, direction: str) -> None:
        """Set current flight direction for border classification"""
        self.flight_direction = direction
    
    def process_height_measurement(self, height: float, position: Tuple[float, float]) -> None:
        """
        Process a height measurement for landing pad detection.
        
        Args:
            height: Height measurement from z-range sensor (meters)
            position: Current (x, y) position of the drone
        """
        if not self.detection_active:
            return
        
        # Store data
        self.height_data.append(height)
        self.position_data.append(position)
        
        # Initialize baseline if not set
        if self.baseline_height is None and len(self.height_data) >= 5:
            self.baseline_height = np.median(list(self.height_data)[-5:])
            self.logger.info(f"Baseline height established: {self.baseline_height:.3f}m")
        
        # Need sufficient data for peak detection
        if len(self.height_data) < self.lag + 1:
            return
        
        # Apply z-score peak detection algorithm
        self._detect_peaks(height, position)
    
    def _detect_peaks(self, current_height: float, position: Tuple[float, float]) -> None:
        """Apply z-score based peak detection algorithm"""
        
        # Get recent data for running statistics
        recent_heights = list(self.height_data)[-self.lag:]
        
        # Update running mean and std
        if len(self.running_mean) == 0:
            # Initialize with recent data
            self.running_mean = deque(recent_heights, maxlen=self.lag)
            self.running_std = deque([np.std(recent_heights)] * len(recent_heights), maxlen=self.lag)
            return
        
        # Calculate current mean and std
        current_mean = np.mean(self.running_mean)
        current_std = np.mean(self.running_std) if self.running_std else 0.001
        
        # Z-score for peak detection
        if current_std > 0:
            z_score = abs(current_height - current_mean) / current_std
        else:
            z_score = 0
        
        # Check for significant peak (platform edge)
        if z_score > self.threshold:
            self.logger.info(f"Z-score: {z_score:.2f}, current_height: {current_height:.3f}, current_mean: {current_mean:.3f}, current_std: {current_std:.3f}, baseline_height: {self.baseline_height:.3f}")
            height_diff = abs(current_height - self.baseline_height) if self.baseline_height else 0
            
            # Verify it's a significant height change
            if height_diff > self.min_peak_height:
                # This looks like a platform edge
                self.peak_positions.append({
                    'position': position,
                    'height': current_height,
                    'height_diff': height_diff,
                    'direction': self.flight_direction,
                    'z_score': z_score
                })
                
                self.logger.info(f"Platform edge detected at ({position[0]:.3f}, {position[1]:.3f}) "
                               f"height_diff={height_diff:.3f}m, z_score={z_score:.2f}")
                
                # Update running statistics with reduced influence
                influenced_value = self.influence * current_height + (1 - self.influence) * current_mean
                self.running_mean.append(influenced_value)
            else:
                # Normal update
                self.running_mean.append(current_height)
        else:
            # Normal update
            self.running_mean.append(current_height)
        
        # Update running standard deviation
        new_std = np.std(self.running_mean)
        self.running_std.append(new_std)
    
    def calculate_pad_center(self) -> Optional[Tuple[float, float]]:
        """
        Calculate landing pad center from detected border points.
        
        Returns:
            (x, y) center coordinates or None if insufficient data
        """
        if len(self.peak_positions) < 4:
            self.logger.warning(f"Insufficient border points for center calculation: {len(self.peak_positions)}")
            return None
        
        # Extract positions
        positions = [p['position'] for p in self.peak_positions]
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        
        # Simple center calculation (centroid)
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        
        # Calculate confidence based on point distribution
        distances = [math.sqrt((x - center_x)**2 + (y - center_y)**2) for x, y in positions]
        std_distance = np.std(distances)
        mean_distance = np.mean(distances)
        
        # Confidence: lower std relative to mean = higher confidence
        if mean_distance > 0:
            self.center_confidence = max(0, 1 - (std_distance / mean_distance))
        else:
            self.center_confidence = 0.0
        
        self.calculated_center = (center_x, center_y)
        
        self.logger.info(f"Calculated pad center: ({center_x:.3f}, {center_y:.3f}) "
                        f"confidence: {self.center_confidence:.2f}")
        
        return self.calculated_center
    
    def is_ready_for_landing(self) -> bool:
        """Check if we have sufficient data for confident landing"""
        if len(self.peak_positions) < 6:
            return False
        
        if self.calculated_center is None:
            return False
        
        # Check confidence threshold
        return self.center_confidence > 0.6
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get detection statistics for debugging"""
        return {
            'total_measurements': len(self.height_data),
            'total_border_points': len(self.peak_positions),
            'baseline_height': self.baseline_height,
            'calculated_center': self.calculated_center,
            'center_confidence': self.center_confidence,
            'detection_active': self.detection_active
        }


class SearchPattern:
    """Generates search patterns for systematic area coverage"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_grid_pattern(self, center: Tuple[float, float], 
                            grid_size: float = 0.5, 
                            grid_spacing: float = 0.3) -> List[Tuple[float, float, str]]:
        """
        Generate a grid search pattern around a center point.
        
        Args:
            center: (x, y) center of search area
            grid_size: Size of search area (meters)
            grid_spacing: Distance between grid points (meters)
            
        Returns:
            List of (x, y, direction) waypoints
        """
        cx, cy = center
        half_size = grid_size / 2
        
        waypoints = []
        
        # Generate grid points
        x_points = np.arange(cx - half_size, cx + half_size + grid_spacing/2, grid_spacing)
        y_points = np.arange(cy - half_size, cy + half_size + grid_spacing/2, grid_spacing)
        
        # Create zigzag pattern for efficient coverage
        for i, y in enumerate(y_points):
            if i % 2 == 0:  # Even rows: left to right
                for x in x_points:
                    direction = 'forward' if i == 0 else 'right'
                    waypoints.append((x, y, direction))
            else:  # Odd rows: right to left
                for x in reversed(x_points):
                    direction = 'left'
                    waypoints.append((x, y, direction))
        
        self.logger.info(f"Generated {len(waypoints)} waypoints for grid search")
        return waypoints
    
    def generate_spiral_pattern(self, center: Tuple[float, float], 
                              max_radius: float = 0.8, 
                              spacing: float = 0.2) -> List[Tuple[float, float, str]]:
        """
        Generate a spiral search pattern.
        
        Args:
            center: (x, y) center of search area
            max_radius: Maximum spiral radius (meters)
            spacing: Distance between spiral turns (meters)
            
        Returns:
            List of (x, y, direction) waypoints
        """
        cx, cy = center
        waypoints = []
        
        # Start from center
        waypoints.append((cx, cy, 'forward'))
        
        # Generate spiral
        radius = spacing
        while radius <= max_radius:
            # Calculate number of points for this radius
            circumference = 2 * math.pi * radius
            num_points = max(8, int(circumference / (spacing * 0.5)))
            
            for i in range(num_points):
                angle = 2 * math.pi * i / num_points
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                
                direction = 'forward'  # Simplified direction
                waypoints.append((x, y, direction))
            
            radius += spacing
        
        self.logger.info(f"Generated {len(waypoints)} waypoints for spiral search")
        return waypoints
