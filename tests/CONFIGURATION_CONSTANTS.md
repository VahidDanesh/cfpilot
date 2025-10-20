# Configuration Constants Reference

## Quick Access to All Tunable Parameters

All constants are defined at the top of the `RealHardwareHoverNav` class for easy modification.

**File**: `tests/real_hardware_hover_nav.py`  
**Lines**: 173-248

---

## How to Modify Parameters

1. Open `tests/real_hardware_hover_nav.py`
2. Find the `RealHardwareHoverNav` class definition
3. Locate the `CONFIGURATION PARAMETERS` section (starts at line 166)
4. Modify the desired constant values
5. Save and run

**No need to search through code** - all constants are in ONE place!

---

## Complete Constants List

### 🎯 Mission Parameters
```python
START_POSITION = (0.7, 1.5)              # Starting position (x, y) in meters
GOAL_POSITION = (2.8, 1.5)               # Goal/landing zone position (x, y) in meters
INITIAL_Z = 0.1                          # Initial Z estimate in meters
INITIAL_YAW = 0.0                        # Initial yaw in radians
```

### ✈️ Flight Parameters
```python
DEFAULT_CRUISE_SPEED = 0.4               # Normal navigation speed (m/s)
DEFAULT_SEARCHING_SPEED = 0.2            # Speed during landing pad search (m/s)
DEFAULT_FLIGHT_HEIGHT = 0.5              # Flight altitude (m)
DEFAULT_CONTROL_RATE_HZ = 50.0           # Control loop frequency (Hz)
```

### 🧭 Navigation Parameters
```python
NAV_WAYPOINT_TOLERANCE = 0.12            # Distance to consider waypoint reached during navigation (m)
SEARCH_WAYPOINT_TOLERANCE = 0.08         # Distance to consider waypoint reached during search (m)
TARGET_FILTER_ALPHA = 0.4                # Exponential smoothing factor for target position (0-1)
```

### 🚧 Obstacle Avoidance Parameters
```python
OBSTACLE_SAFETY_MARGIN = 0.05            # Extra margin around detected obstacles (m)
NEAR_OBSTACLE_DISTANCE = 0.5             # Distance to start obstacle repulsion (m)
MAX_SPEED_NEAR_OBSTACLE = 0.15           # Speed limit when near obstacles (m/s)
AVOIDANCE_GAIN = 0.5                     # Gain for obstacle repulsion force
AVOIDANCE_MIN_THRESHOLD = 0.01           # Minimum repulsion force to apply (m/s)
AVOIDANCE_VECTOR_THRESHOLD = 0.02        # Threshold to show "avoiding" status in viz (m/s)
```

### 🎯 Landing Pad Detection Parameters
```python
# Search behavior
LANDING_ZONE_RADIUS = 0.1                # Distance to goal to trigger search mode (m)
SEARCH_AREA_SIZE = 0.6                   # Cross pattern search extent from center (m)
SEARCH_STEP_SIZE = 0.3                   # Distance between search waypoints (m)
PAD_CENTER_TOLERANCE = 0.05              # Distance to consider arrived at pad center (m)
SEARCH_TIMEOUT = 60.0                    # Maximum search time before fallback (s)

# Detection algorithm
DETECTION_LAG = 12                       # Smoothing window size for peak detection
DETECTION_THRESHOLD = 3.0                # Z-score threshold for edge detection (1.0-2.5)
DETECTION_INFLUENCE = 0.1                # Peak influence on running mean (0.1-0.3)
MIN_PEAK_HEIGHT = 0.05                   # Minimum height difference for edge detection (m)

# Z-range sensor
ZRANGE_MIN_VALID = 0.05                  # Minimum valid z-range reading (m)
ZRANGE_MAX_VALID = 1.0                   # Maximum valid z-range reading (m)
```

### 🗺️ Map Parameters
```python
MAP_WIDTH = 5.0                          # Map width (m)
MAP_HEIGHT = 3.0                         # Map height (m)
MAP_RESOLUTION = 0.05                    # Grid cell size (m)
MAP_CENTER_X = 2.5                       # Map center X coordinate (m)
MAP_CENTER_Y = 1.5                       # Map center Y coordinate (m)
MAP_BOUNDARY_WIDTH = 2                   # Boundary cells to mark as occupied
MAP_OBSTACLE_THRESHOLD = 0.5             # Threshold for considering cell as obstacle
MAP_UPDATE_INTERVAL = 3                  # Update map every N steps
MAP_MAX_SENSOR_RANGE = 2.0               # Maximum sensor range to use for mapping (m)
MAP_MIN_SENSOR_RANGE = 0.05              # Minimum sensor range to use for mapping (m)
MAP_OCCUPANCY_VALUE = 1.0                # Value to set for occupied cells
```

### 🎮 Control Parameters
```python
YAW_RATE_HOLD = 0.0                      # Yaw rate to maintain current heading (deg/s)
HOVER_SETPOINT_DEFAULTS = {              # Default hover setpoint values
    'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'height': DEFAULT_FLIGHT_HEIGHT
}
```

### ⏱️ Timing Parameters
```python
CONNECTION_TIMEOUT = 10.0                # Timeout for drone connection (s)
SENSOR_INIT_DELAY = 0.3                  # Delay after sensor initialization (s)
TAKEOFF_HOLD_DURATION = 0.5              # Hold time at start position after takeoff (s)
LANDING_HOLD_DURATION = 1.0              # Hold time at pad center before landing (s)
LANDING_GRADUAL_DURATION = 2.0           # Duration of gradual landing (s)
LANDING_MIN_HEIGHT = 0.2                 # Minimum height during gradual landing (m)
DEFAULT_HOLD_DURATION = 1.0              # Default hold position duration (s)
DEBUG_PRINT_INTERVAL = 2.0               # Print debug info every N seconds
```

### 📊 Visualization Parameters
```python
VIZ_XLIM = (0, 5)                        # Visualization X-axis limits (m)
VIZ_YLIM = (0, 3)                        # Visualization Y-axis limits (m)
VIZ_FIGSIZE = (12, 9)                    # Figure size (width, height) in inches
VIZ_ANIMATION_SPEED = 0.05               # Animation frame delay (s)
```

---

## Usage Examples

### Example 1: Change Flight Speed
```python
# In RealHardwareHoverNav class constants section:
DEFAULT_CRUISE_SPEED = 0.3               # Changed from 0.4 to 0.3 m/s
DEFAULT_SEARCHING_SPEED = 0.15           # Changed from 0.2 to 0.15 m/s
```

### Example 2: Adjust Landing Detection Sensitivity
```python
# In RealHardwareHoverNav class constants section:
DETECTION_THRESHOLD = 2.0                # Changed from 3.0 - more sensitive
MIN_PEAK_HEIGHT = 0.03                   # Changed from 0.05 - detect smaller edges
```

### Example 3: Modify Search Pattern
```python
# In RealHardwareHoverNav class constants section:
SEARCH_AREA_SIZE = 0.8                   # Changed from 0.6 - larger search area
SEARCH_STEP_SIZE = 0.25                  # Changed from 0.3 - finer steps
```

### Example 4: Use Runtime Overrides
```python
# In main() function:
demo = RealHardwareHoverNav(
    cruise_speed=0.3,                     # Override default
    searching_speed=0.15,                 # Override default
    flight_height=0.6,                    # Override default
    control_rate_hz=50.0                  # Override default (or omit to use class constant)
)
```

---

## Constants Usage in Code

All constants are used throughout the code instead of hardcoded values:

✅ **Before Refactoring:**
```python
if dist < 0.12:  # Magic number - what does 0.12 mean?
    waypoint_reached = True
```

✅ **After Refactoring:**
```python
if dist < self.NAV_WAYPOINT_TOLERANCE:  # Clear meaning!
    waypoint_reached = True
```

---

## Benefits

1. **Single Source of Truth** - All parameters in one place
2. **Self-Documenting** - Each constant has clear description
3. **Easy Tuning** - Change value once, affects entire codebase
4. **No Magic Numbers** - Every value has meaningful name
5. **Better Readability** - Code intent is clear
6. **Safer Changes** - Less chance of missing a hardcoded value

---

## Quick Tuning Guide

**For faster flight:**
- ↑ `DEFAULT_CRUISE_SPEED`
- ↑ `DEFAULT_SEARCHING_SPEED`

**For more careful flight:**
- ↓ `MAX_SPEED_NEAR_OBSTACLE`
- ↑ `NEAR_OBSTACLE_DISTANCE`

**For better detection:**
- ↓ `DETECTION_THRESHOLD`
- ↓ `MIN_PEAK_HEIGHT`
- ↓ `SEARCH_STEP_SIZE`

**For faster search:**
- ↑ `SEARCH_STEP_SIZE`
- ↓ `SEARCH_AREA_SIZE`

---

## See Also

- `LANDING_PAD_DETECTION_PARAMS.md` - Detailed landing pad parameter tuning guide
- `real_hardware_hover_nav.py` - Main implementation file

---

**Last Updated**: October 2025  
**Version**: 1.0

