"""
CFPilot - Crazyflie Autonomous Flight Package

An advanced autonomous flight system for the Crazyflie platform with comprehensive 
safety features, obstacle avoidance, and landing pad detection capabilities.
"""

__version__ = "1.0.0"
__author__ = "Crazyflie Autonomous Team"

from .controller import CrazyflieController
from .detection import LandingPadDetector, SearchPattern
from .missions import BasicFlightMission, SensorExplorationMission, LandingPadDetectionMission

try:
    from .visualization import PointCloudPlotter
    __all__ = [
        "CrazyflieController", "LandingPadDetector", "SearchPattern",
        "BasicFlightMission", "SensorExplorationMission", "LandingPadDetectionMission"
    ]
except ImportError:
    __all__ = [
        "CrazyflieController", "LandingPadDetector", "SearchPattern", 
        "BasicFlightMission", "SensorExplorationMission", "LandingPadDetectionMission"
    ]
