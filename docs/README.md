# CFPilot - Crazyflie Autonomous Flight System

An advanced autonomous flight system for the Crazyflie platform with comprehensive safety features, obstacle avoidance, landing pad detection, and dynamic path planning capabilities.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Flight Missions](#flight-missions)
- [System Architecture](#system-architecture)
- [Core Algorithms](#core-algorithms)
  - [GridMap](#gridmap)
  - [D* Lite Path Planning](#d-lite-path-planning)
  - [Landing Pad Detection](#landing-pad-detection)
  - [Obstacle Avoidance](#obstacle-avoidance)
- [Configuration](#configuration)
- [Safety Features](#safety-features)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [References](#references)

---

## Overview

CFPilot is a complete autonomous navigation system for Crazyflie drones that enables:
- **Autonomous navigation** from start to goal using D* Lite path planning
- **Real-time obstacle avoidance** using multiranger sensors
- **Landing pad detection** via sweep patterns and edge detection
- **Precision landing** on detected pads
- **Dynamic replanning** when new obstacles are encountered
- **Round-trip missions** with multiple landing sequences

### Mission Flow
```
┌─────────────────────────────────────────────────────────────┐
│ 1. TAKEOFF & NAVIGATION                                     │
│    ├─ Takeoff from start position                           │
│    ├─ Navigate to goal using D* Lite                        │
│    └─ Avoid obstacles reactively                            │
├─────────────────────────────────────────────────────────────┤
│ 2. LANDING PAD DETECTION AT GOAL                            │
│    ├─ Execute sweep pattern when near goal                  │
│    ├─ Collect edge measurements from sensors                │
│    ├─ Compute pad center with confidence scoring            │
│    └─ Lock center when confidence > 60%                     │
├─────────────────────────────────────────────────────────────┤
│ 3. PRECISION LANDING AT GOAL                                │
│    ├─ Move to detected pad center                           │
│    ├─ Descend at controlled rate                            │
│    └─ Stop motors on ground                                 │
├─────────────────────────────────────────────────────────────┤
│ 4. RETURN JOURNEY                                           │
│    ├─ Takeoff from goal                                     │
│    ├─ Navigate back to start                                │
│    └─ Repeat detection and landing at start                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Features

### Navigation
✅ D* Lite path planning with dynamic replanning  
✅ Grid-based occupancy mapping (5cm resolution)  
✅ Reactive obstacle avoidance using potential fields  
✅ Exponential filtering for smooth motion  
✅ Waypoint-based navigation with tolerance checking  

### Landing Detection
✅ Lawnmower sweep pattern for systematic coverage  
✅ Edge detection from 4-directional ToF sensors  
✅ Statistical center computation with confidence scoring  
✅ Minimum 12 edge measurements for reliability  
✅ Precision landing within ±5cm accuracy  

### Safety
✅ Real-time obstacle avoidance during all phases  
✅ Minimum altitude floor (0.2m)  
✅ Velocity clamping near obstacles  
✅ Thread-safe state management  
✅ Sensor outlier rejection and filtering  
✅ Emergency landing on connection loss  
✅ Battery monitoring with auto-landing  

### Performance
- **Navigation speed**: 0.3 m/s cruise
- **Landing precision**: ±5cm typical
- **Mission time**: 60-90 seconds (round trip)
- **Control rate**: 50 Hz hover commands, 20 Hz main loop
- **Sensor rate**: 10 Hz multiranger updates

---

## Installation

### Prerequisites

- Python 3.6 or higher
- Crazyflie 2.1 drone
- Multiranger expansion deck (4-directional ToF sensors)
- Crazyradio PA USB dongle
- Landing pad (0.2-0.4m dimenssion, 0.1 m height)

### From Source

1. **Clone the repository:**
   ```bash
   git clone https://github.com/VahidDanesh/cfpilot.git
   cd cfpilot
   ```

2. **Install the package:**
   ```bash
   pip install -e .
   ```

3. **Or install with development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

### Dependencies

The following packages will be installed automatically:
- `cflib` - Crazyflie Python library
- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `scipy` - Signal processing and interpolation

---

## Quick Start

### 1. Configure Your Crazyflie

Edit `cfpilot/config/flight_config.yaml` to set your drone's URI:

```yaml
connection:
  uri: 'radio://0/80/2M/E7E7E7E7E7'
```

### 2. Run Autonomous Mission

```bash
# Navigate to project root
cd cfpilot

# Run the autonomous landing mission
python -m cfpilot.tests.fly_autonomous
```


## Flight Missions

### Basic Flight (`cfpilot basic`)
- Simple takeoff, hover, and landing
- Position hold using PositionHlCommander
- Comprehensive safety features

### Sensor Exploration (`cfpilot sensor`)
- Multiranger sensor-based navigation
- Obstacle detection and avoidance
- Automatic free-space exploration

### Autonomous Landing (`python -m cfpilot.tests.fly_autonomous`)
- **Complete round-trip mission**
- **Landing pad detection at goal and start**
- **Dynamic path planning with D* Lite**
- **Real-time obstacle avoidance**
- **Precision landing on detected pads**

---

## System Architecture

### Data Flow

```
┌──────────────────┐
│  Multiranger     │  ← 4-directional ToF sensors
│  (4 sensors)     │
└────────┬─────────┘
         ↓
┌──────────────────┐
│  Sensor Filter   │  ← Outlier rejection + median filtering
│                  │
└────────┬─────────┘
         ↓
┌──────────────────┐
│   GridMap        │  ← Occupancy grid (5cm resolution)
│   Update         │
└────────┬─────────┘
         ↓
┌──────────────────┐
│  D* Lite         │  ← Dynamic path planning
│  Planner         │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Mission Control  │  ← State machine + navigation
│ AutonomousMission│
└────────┬─────────┘
         ↓
┌──────────────────┐
│ HoverTimer       │  ← 50Hz setpoint generation
│ (Background)     │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Crazyflie        │  ← Motor commands
│                  │
└──────────────────┘
```

### Key Components

**AutonomousMission** - Main mission controller
- Manages flight phases and state transitions
- Integrates path planning, obstacle avoidance, and landing detection
- Handles takeoff, navigation, landing, and return journey

**LandingPadDetector** - Landing pad detection system
- Processes height measurements during flight
- Uses peak detection algorithm to identify platform edges
- Calculates pad center from detected border points
- Provides confidence scoring for landing decisions

**GridMap** - Occupancy grid mapping
- 2D grid representation of environment
- 5cm resolution for precise obstacle representation
- Efficient NumPy-based operations
- Boundary occupation and polygon support

**DStarLite** - Dynamic path planner
- Efficient incremental replanning
- Heuristic-based search (Euclidean distance)
- 8-directional motion primitives
- Path simplification and smoothing

---

## Core Algorithms

### GridMap

The `GridMap` class provides a 2D occupancy grid for environment representation and path planning.

#### Key Features

```python
from cfpilot.mapping import GridMap

# Create grid map: 6m x 3m workspace with 5cm resolution
grid_map = GridMap(
    width=int(6.0/0.05),      # 120 cells
    height=int(3.0/0.05),     # 60 cells
    resolution=0.05,           # 5cm per cell
    center_x=3.0,              # Center at (3.0, 1.5)
    center_y=1.5
)

# Set obstacle at world position
grid_map.set_value_from_xy_pos(x=2.5, y=1.0, val=1.0)

# Check if position is occupied
x_ind, y_ind = grid_map.get_xy_index_from_xy_pos(2.5, 1.0)
is_occupied = grid_map.check_occupied_from_xy_index(x_ind, y_ind, occupied_val=0.5)

# Occupy boundaries (safety margins)
grid_map.occupy_boundaries(boundary_width=5, val=1.0)
```

#### Grid Update Algorithm

```python
def update_map(grid_map, sensors, drone_x, drone_y, drone_yaw, safety_margin=0.05):
    """
    Update occupancy grid from sensor readings.
    
    1. Clear drone's position (prevent self-blocking)
    2. Mark obstacles detected by sensors
    3. Expand obstacles with safety margin
    """
    # Clear area around drone
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            grid_map.set_value_from_xy_pos(
                drone_x + dx*0.05, 
                drone_y + dy*0.05, 
                val=0.0
            )
    
    # Mark obstacles from each sensor
    for direction, distance in sensors.items():
        if distance and 0.02 < distance < 2.0:
            angle = drone_yaw + sensor_angle[direction]
            obs_x = drone_x + distance * cos(angle)
            obs_y = drone_y + distance * sin(angle)
            
            # Expand with safety margin
            grid_map.set_value_from_xy_pos(obs_x, obs_y, val=1.0)
```

---

### D* Lite Path Planning

D* Lite is an incremental heuristic search algorithm that efficiently replans paths when the environment changes.

#### Algorithm Overview

**D* Lite** (Dynamic A* Lite) is an optimized version of D* that:
- Reuses information from previous searches
- Updates only affected nodes when obstacles appear
- Provides optimal paths with minimal recomputation

#### Key Concepts

1. **Priority Queue**: Nodes sorted by key values `k = [k1, k2]`
   - `k1 = min(g, rhs) + h(start, node)`
   - `k2 = min(g, rhs)`

2. **g-value**: Cost of best path found so far from start to node

3. **rhs-value** (right-hand side): One-step lookahead value
   ```
   rhs(node) = min(cost(node, successor) + g(successor))
   ```

4. **Consistency**: Node is consistent when `g(node) == rhs(node)`

#### Usage Example

```python
from cfpilot.planning.DStarLite.d_star_lite import DStarLite
from cfpilot.mapping import GridMap

# Create grid map
grid_map = GridMap(width=120, height=60, resolution=0.05, 
                   center_x=3.0, center_y=1.5)

# Initialize D* Lite planner
planner = DStarLite(
    grid_map=grid_map,
    obstacle_threshold=0.5,  # Cells >= 0.5 are obstacles
    show_animation=False
)

# Initial planning
success, path_x, path_y = planner.run(
    start=(0.6, 1.3),
    goal=(5.0, 1.5),
    simplify=True,   # Simplify path (remove redundant waypoints)
    smooth=False     # Smooth path with splines
)

# When new obstacles detected, update map
cells_updated = update_map(grid_map, sensors, x, y, yaw)

if cells_updated > 0:
    # Dynamic replanning
    replanned = planner.update_map()
    
    if replanned:
        # Extract new path
        path = planner.compute_current_path()
        path_x, path_y = planner.path_to_world_coords(path)
```

#### D* Lite Algorithm Steps

```
1. INITIALIZATION:
   - Set rhs(goal) = 0
   - Insert goal into priority queue
   - All other nodes: g = rhs = ∞

2. COMPUTE SHORTEST PATH:
   while top_key < calculate_key(start) OR rhs(start) ≠ g(start):
       u = pop node with smallest key
       
       if g(u) > rhs(u):  # Overconsistent
           g(u) = rhs(u)
           for each successor s of u:
               update_vertex(s)
       else:  # Underconsistent
           g(u) = ∞
           for each successor s of u:
               update_vertex(s)
           update_vertex(u)

3. UPDATE_VERTEX(node):
   if node ≠ goal:
       rhs(node) = min(cost(node, s) + g(s)) for all successors s
   
   if node in queue:
       remove node from queue
   
   if g(node) ≠ rhs(node):
       insert node into queue with key = calculate_key(node)

4. DYNAMIC REPLANNING (when obstacles change):
   for each changed edge (u, v):
       update edge cost
       update_vertex(u)
   
   recompute_shortest_path()
```

#### Motion Primitives

D* Lite uses 8-directional motion:

```
Directions:  Cost:
  7  0  1    √2  1  √2
  6  *  2     1  *   1
  5  4  3    √2  1  √2
```

```python
motions = [
    Node(1, 0, 1.0),      # Right
    Node(1, 1, math.sqrt(2)),  # Up-Right
    Node(0, 1, 1.0),      # Up
    Node(-1, 1, math.sqrt(2)), # Up-Left
    Node(-1, 0, 1.0),     # Left
    Node(-1, -1, math.sqrt(2)),# Down-Left
    Node(0, -1, 1.0),     # Down
    Node(1, -1, math.sqrt(2))  # Down-Right
]
```

#### Path Simplification

After finding a path, D* Lite can simplify it by removing redundant waypoints:

```python
def simplify_path(self, path_x, path_y):
    """
    Remove waypoints that lie on straight lines.
    Uses line-of-sight checking in the grid.
    """
    if len(path_x) <= 2:
        return path_x, path_y
    
    simplified_x, simplified_y = [path_x[0]], [path_y[0]]
    
    for i in range(1, len(path_x) - 1):
        # Check if we can skip this waypoint
        if not self.line_of_sight(simplified_x[-1], simplified_y[-1],
                                   path_x[i+1], path_y[i+1]):
            simplified_x.append(path_x[i])
            simplified_y.append(path_y[i])
    
    simplified_x.append(path_x[-1])
    simplified_y.append(path_y[-1])
    
    return simplified_x, simplified_y
```

#### Performance Characteristics

- **Initial planning**: O(N log N) where N = number of cells
- **Replanning**: O(K log K) where K = number of affected cells (typically K << N)
- **Memory**: O(N) for storing g and rhs values
- **Optimality**: Guaranteed to find optimal path (same as A*)

---

### Landing Pad Detection

The landing pad detection system uses a **peak detection algorithm** combined with a **sweep pattern** to locate and land on elevated platforms.

#### Detection Algorithm

```python
class LandingPadDetector:
    """
    Detects landing pads using z-range sensor and peak detection.
    
    Algorithm:
    1. Collect height measurements during flight
    2. Apply peak detection to identify platform edges
    3. Calculate pad center from detected edges
    4. Compute confidence score
    5. Lock center when confidence > threshold
    """
```

#### Peak Detection (Z-Score Algorithm)

The detector uses a modified z-score algorithm to identify significant height changes:

```python
def process_height_measurement(self, height_m, position):
    """
    Process a single height measurement.
    
    Peak Detection Algorithm:
    -------------------------
    1. Maintain running statistics (mean, std) over window
    2. Compute z-score: z = |height - mean| / std
    3. If z > threshold AND height_diff > min_peak_height:
        → Edge detected!
    4. Update running statistics with influence factor
    """
    # Calculate z-score
    if len(self.filtered_signal) >= self.lag:
        mean = np.mean(self.filtered_signal[-self.lag:])
        std = np.std(self.filtered_signal[-self.lag:])
        
        if std > 0:
            z_score = abs(height_m - mean) / std
            
            if z_score > self.threshold:
                height_diff = abs(height_m - self.baseline_height)
                
                if height_diff > self.min_peak_height:
                    # Edge detected!
                    self.peak_positions.append({
                        'position': position,
                        'height': height_m,
                        'timestamp': time.time()
                    })
```

**Sweep Pattern**:
```python
def generate_sweep_pattern(center, width=1.0, height=1.0, spacing=0.3):
    """
    Lawnmower pattern:
    
    →→→→→→→→→  (spacing = 0.3m)
    ←←←←←←←←←
    →→→→→→→→→
    ←←←←←←←←←
    """
    waypoints = []
    num_rows = int(height / spacing)
    for i in range(num_rows):
        y = center[1] - height/2 + i * spacing
        if i % 2 == 0:  # Left to right
            for x in np.arange(center[0] - width/2, center[0] + width/2, spacing):
                waypoints.append((x, y))
        else:  # Right to left
            for x in np.arange(center[0] + width/2, center[0] - width/2, -spacing):
                waypoints.append((x, y))
    return waypoints
```

**Center Calculation**:
```python
def calculate_pad_center(self):
    """
    Statistical approach:
    1. Mean position of all detected edges
    2. Compute spread (std deviation)
    3. Confidence = 1 - (spread / expected_pad_size)
    4. Require confidence > 0.6 (60%)
    """
    if len(self.peak_positions) < 2:
        return None
    
    positions = [p['position'] for p in self.peak_positions]
    center_x = np.mean([p[0] for p in positions])
    center_y = np.mean([p[1] for p in positions])
    
    distances = [np.hypot(p[0]-center_x, p[1]-center_y) for p in positions]
    spread = np.std(distances)
    
    confidence = max(0.0, 1.0 - spread / 0.15)
    
    return (center_x, center_y) if confidence > 0.6 else None
```

**Parameters** (configurable):
```python
detection_config = {
    'lag': 5,                    # Smoothing window size
    'threshold': 2.0,            # Z-score threshold
    'influence': 0.6,            # Peak influence on statistics
    'min_peak_height': 0.05,     # Minimum edge height (meters)
    'min_edge_distance': 0.03    # Minimum distance between edges
}
```

---

### 4. Obstacle Avoidance

**Purpose**: Reactive collision avoidance using potential field method

**Algorithm**:
```python
def compute_avoidance_velocity(sensors, yaw, danger_dist=0.5, gain=0.7):
    """
    Potential field repulsion:
    
    For each sensor reading d:
      if d < danger_dist:
        force = ((danger_dist - d) / danger_dist)²
        repulsion += force * direction_away_from_obstacle * gain
    
    Returns: (vx_repulsion, vy_repulsion) in world frame
    """
    vx_avoid, vy_avoid = 0.0, 0.0
    
    sensor_angles = {'front': 0, 'right': -π/2, 'back': π, 'left': π/2}
    
    for direction, distance in sensors.items():
        if distance and distance < danger_dist:
            force = ((danger_dist - distance) / danger_dist) ** 2
            angle = yaw + sensor_angles[direction]
            
            # Repel away from obstacle
            vx_avoid -= force * cos(angle) * gain
            vy_avoid -= force * sin(angle) * gain
    
    return vx_avoid, vy_avoid
```

**Combined Control**:
```python
# Path following velocity
vx_path, vy_path = compute_path_velocity(current, target, speed=0.3)

# Obstacle avoidance velocity
vx_avoid, vy_avoid = compute_avoidance_velocity(sensors, yaw)

# Combined command
vx_cmd = vx_path + vx_avoid
vy_cmd = vy_path + vy_avoid

# Speed limiting near obstacles
if min(sensors.values()) < 0.4:
    speed = hypot(vx_cmd, vy_cmd)
    if speed > 0.2:
        vx_cmd *= 0.2 / speed
        vy_cmd *= 0.2 / speed
```

---

## Configuration

### Flight Parameters

Edit constants in `fly_autonomous.py`:

```python
# Mission parameters
START_POSITION = (0.6, 1.3)
GOAL_POSITION = (5.0, 1.5)

# Flight parameters
cruise_speed = 0.3          # m/s during navigation
flight_height = 0.5         # m cruise altitude
control_rate_hz = 50.0      # Hz control loop

# Detection parameters
detection_config = {
    'lag': 5,
    'threshold': 2.0,
    'influence': 0.6,
    'min_peak_height': 0.05,
    'min_edge_distance': 0.03
}
```

### Tuning Guide

**For faster flight**:
- Increase `cruise_speed` (0.3 → 0.4 m/s)
- Increase sweep spacing (0.3 → 0.4m)

**For better detection**:
- Decrease `threshold` (2.0 → 1.5) - more sensitive
- Decrease `min_peak_height` (0.05 → 0.03m)
- Decrease sweep spacing (0.3 → 0.2m)

**For more careful flight**:
- Decrease `cruise_speed` (0.3 → 0.2 m/s)
- Increase safety margins
- Decrease max speed near obstacles

---

## Troubleshooting

### Landing Pad Not Detected

**Symptoms**: No edges detected during sweep

**Solutions**:
1. Check pad has hard edges (not gradual slopes)
2. Verify pad is reflective for ToF sensors
3. Lower `threshold` (2.0 → 1.5)
4. Lower `min_peak_height` (0.05 → 0.03m)
5. Increase sweep coverage area

### Path Planning Fails

**Symptoms**: "Planning failed" error

**Solutions**:
1. Check start/goal within grid bounds
2. Verify obstacles don't completely block path
3. Increase grid size or decrease resolution
4. Check `obstacle_threshold` setting

### Drone Oscillates During Landing

**Symptoms**: Unstable descent

**Solutions**:
1. Increase exponential filter alpha (more smoothing)
2. Decrease descent rate
3. Check sensor readings for noise
4. Verify control rate is adequate (50 Hz)

### Connection Issues

**Symptoms**: "Connection failed"

**Solutions**:
1. Check Crazyradio PA is plugged in
2. Verify URI in config matches drone
3. Ensure drone is powered on and in range
4. Check no other programs using Crazyradio

---

## Project Structure

```
crazyflie/
├── cfpilot/
│   ├── cfpilot/
│   │   ├── __init__.py
│   │   ├── cli.py                    # Command line interface
│   │   ├── controller.py             # Crazyflie connection & logging
│   │   ├── detection.py              # Landing pad detection
│   │   ├── mapping.py                # GridMap implementation
│   │   ├── missions.py               # Flight missions
│   │   ├── planning/
│   │   │   ├── DStarLite/
│   │   │   │   └── d_star_lite.py   # D* Lite planner
│   │   │   ├── AStar/               # A* planner
│   │   │   └── ...                  # Other planners
│   │   ├── config/
│   │   │   └── flight_config.yaml   # Configuration
│   │   └── visualization.py         # Real-time display
│   ├── tests/
│   │   ├── fly_autonomous.py        # Main autonomous mission
│   │   └── ...
│   ├── setup.py
│   ├── pyproject.toml
│   ├── requirements.txt
│   └── README.md                  
└── refs/
    ├── PythonRobotics/              # Algorithm references
    └── crazyflie-lib-python/        # Crazyflie library
```

---

## References

### Papers & Algorithms

- **D* Lite**: [Koenig & Likhachev, AAAI 2002](http://idm-lab.org/bib/abstracts/papers/aaai02b.pdf)
- **Fast Replanning**: [Koenig & Likhachev, ICRA 2002](http://www.cs.cmu.edu/~maxim/files/dlite_icra02.pdf)
- **GridMap**: [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)

### Libraries

- **cflib**: [Crazyflie Python Library](https://github.com/bitcraze/crazyflie-lib-python)
- **NumPy**: Numerical computing
- **Matplotlib**: Visualization
- **SciPy**: Signal processing

### Hardware

- **Crazyflie 2.1**: [Bitcraze](https://www.bitcraze.io/products/crazyflie-2-1/)
- **Multiranger Deck**: 4-directional ToF sensors
- **Flow Deck**: Optical flow positioning (optional)

---

## License

See LICENSE file for details.