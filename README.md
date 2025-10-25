# CFPilot - Crazyflie Autonomous Flight System

An advanced autonomous flight system for Crazyflie drones with obstacle avoidance, dynamic path planning, and precision landing capabilities.

## Features

✅ **D* Lite path planning** with dynamic replanning  
✅ **Real-time obstacle avoidance** using multiranger sensors  
✅ **Landing pad detection** via sweep patterns and peak detection  
✅ **Precision landing** within ±5cm accuracy  
✅ **Grid-based mapping** with 5cm resolution  
✅ **Round-trip autonomous missions**  

---

## Installation

### Prerequisites
- Python 3.6+
- Crazyflie 2.1 + Multiranger deck + Crazyradio PA
- Landing pad (0.2-0.4m diameter, reflective surface)

### Install

```bash
# Clone repository
git clone https://github.com/yourusername/crazyflie.git
cd crazyflie/cfpilot

# Install package
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

**Dependencies**: `cflib`, `numpy`, `matplotlib`, `scipy` (auto-installed)

---

## Quick Start

### 1. Configure Drone URI

Edit `cfpilot/config/flight_config.yaml`:
```yaml
connection:
  uri: 'radio://0/80/2M/E7E7E7E7E7'
```

### 2. Run Autonomous Mission

```bash
python -m cfpilot.tests.fly_autonomous
```

### 3. Mission Flow

```
1. Takeoff from start (0.6, 1.3)
2. Navigate to goal (5.0, 1.5) using D* Lite
3. Search for landing pad with sweep pattern
4. Land precisely on detected pad
5. Return to start and repeat landing
```

**Mission time**: 60-90 seconds

---

## Core Algorithms

### 1. GridMap - Occupancy Grid

2D environment representation for path planning (120×60 cells, 5cm resolution):

```python
from cfpilot.mapping import GridMap

grid_map = GridMap(width=120, height=60, resolution=0.05, 
                   center_x=3.0, center_y=1.5)
grid_map.set_value_from_xy_pos(x=2.5, y=1.0, val=1.0)  # Mark obstacle
grid_map.occupy_boundaries(boundary_width=5)  # Safety margins
```

**Reference**: [PythonRobotics GridMap](https://github.com/AtsushiSakai/PythonRobotics/blob/master/Mapping/grid_map_lib/grid_map_lib.py)

### 2. D* Lite - Dynamic Path Planning

Efficient incremental replanning when environment changes:

```python
from cfpilot.planning.DStarLite.d_star_lite import DStarLite

planner = DStarLite(grid_map, obstacle_threshold=0.5)
success, path_x, path_y = planner.run(start=(0.6, 1.3), goal=(5.0, 1.5))

# When obstacles detected, replan efficiently
if cells_updated > 0:
    replanned = planner.update_map()
    path = planner.compute_current_path()
```

**Key concepts**:
- **g-value**: Cost of best path found
- **rhs-value**: One-step lookahead value
- **Consistency**: Node is consistent when g == rhs
- **Performance**: O(K log K) replanning where K = affected cells (K << N)

**References**: [D* Lite Paper](http://idm-lab.org/bib/abstracts/papers/aaai02b.pdf), [Fast Replanning](http://www.cs.cmu.edu/~maxim/files/dlite_icra02.pdf)

### 3. Landing Pad Detection

Peak detection with z-score algorithm:

```python
# Z-score peak detection
z = |height - mean| / std
if z > threshold (2.0) AND height_diff > min_peak_height (0.05m):
    → Edge detected!

# Center calculation from edges
center = mean(edge_positions)
confidence = 1 - (spread / expected_pad_size)
if confidence > 0.6:
    → Lock center and land
```

**Sweep pattern**: Lawnmower pattern with 0.3m spacing for full coverage

### 4. Obstacle Avoidance

Potential field repulsion:

```python
for sensor_reading in sensors:
    if distance < danger_dist:
        force = ((danger_dist - distance) / danger_dist)²
        repulsion += force * direction_away * gain

velocity_command = path_velocity + avoidance_velocity
```

---

## Configuration

Key parameters in `fly_autonomous.py`:

```python
# Mission
START_POSITION = (0.6, 1.3)
GOAL_POSITION = (5.0, 1.5)

# Flight
cruise_speed = 0.3          # m/s
flight_height = 0.5         # m
control_rate_hz = 50.0      # Hz

# Detection
detection_config = {
    'lag': 5,               # Smoothing window
    'threshold': 2.0,       # Z-score threshold
    'influence': 0.6,       # Peak influence
    'min_peak_height': 0.05 # Minimum edge height (m)
}
```

**Tuning tips**:
- Faster flight: ↑ `cruise_speed` (0.3 → 0.4)
- Better detection: ↓ `threshold` (2.0 → 1.5)
- More careful: ↓ `cruise_speed`, ↑ safety margins

---

## System Architecture

```
Multiranger (10Hz) → Sensor Filter → GridMap → D* Lite Planner
                                         ↓
                              AutonomousMission (20Hz)
                                         ↓
                              HoverTimer (50Hz) → Crazyflie
```

**Control flow**:
1. Sensors detect obstacles → Update grid map
2. D* Lite replans path if needed
3. Mission controller computes velocities (path + avoidance)
4. HoverTimer sends commands at 50Hz

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Pad not detected | Lower `threshold` (2.0→1.5), check pad has hard edges |
| Path planning fails | Check start/goal in bounds, verify path not blocked |
| Drone oscillates | Increase filter smoothing, decrease descent rate |
| Connection issues | Check URI, verify Crazyradio plugged in |

---

## Project Structure

```
cfpilot/
├── cfpilot/
│   ├── controller.py           # Crazyflie connection & logging
│   ├── detection.py            # Landing pad detection
│   ├── mapping.py              # GridMap implementation
│   ├── planning/
│   │   └── DStarLite/
│   │       └── d_star_lite.py  # D* Lite planner
│   ├── config/
│   │   └── flight_config.yaml  # Configuration
│   └── visualization.py        # Real-time display
├── tests/
│   └── fly_autonomous.py       # Main autonomous mission
├── setup.py
└── README.md                   # This file
```


---

## References

- **D* Lite**: Koenig & Likhachev, AAAI 2002
- **GridMap**: [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)
- **cflib**: [Crazyflie Python Library](https://github.com/bitcraze/crazyflie-lib-python)


---

## License

See LICENSE file for details.

