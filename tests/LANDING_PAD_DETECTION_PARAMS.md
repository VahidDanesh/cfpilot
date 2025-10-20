# Landing Pad Detection - Tunable Parameters Guide

## Overview
This guide describes all tunable parameters for the landing pad detection system implemented in `real_hardware_hover_nav.py`.

## Landing Pad Specifications
- **Dimensions**: 0.3m x 0.3m
- **Height above ground**: 0.1m
- **Detection method**: Z-range sensor + cross pattern search

---

## 1. Mission Control Parameters

### `landing_zone_radius` (default: 0.4m)
- **Location**: `RealHardwareHoverNav.__init__`
- **Purpose**: Distance to goal position where search mode activates
- **Tuning range**: 0.3 - 0.6m
- **Effect**: 
  - Smaller = triggers search closer to goal (more precise entry)
  - Larger = triggers search earlier (more exploration area)

### `search_area_size` (default: 0.5m)
- **Location**: `RealHardwareHoverNav.__init__`
- **Purpose**: How far to search in each direction from center
- **Tuning range**: 0.4 - 0.8m
- **Effect**: 
  - Must be larger than pad dimensions (>0.3m)
  - Too small = may miss pad edges
  - Too large = longer search time

### `search_step_size` (default: 0.15m)
- **Location**: `RealHardwareHoverNav.__init__`
- **Purpose**: Distance between waypoints in cross pattern
- **Tuning range**: 0.1 - 0.2m
- **Effect**: 
  - Smaller = more waypoints, better coverage, slower
  - Larger = fewer waypoints, faster, may miss edges

### `pad_center_tolerance` (default: 0.05m)
- **Location**: `RealHardwareHoverNav.__init__`
- **Purpose**: Distance threshold to consider "arrived at pad center"
- **Tuning range**: 0.03 - 0.08m
- **Effect**: 
  - Smaller = more precise positioning, may take longer
  - Larger = faster arrival, less precise

### `search_timeout` (default: 60s)
- **Location**: `RealHardwareHoverNav.__init__`
- **Purpose**: Maximum time allowed for search phase before fallback
- **Tuning range**: 30 - 120s
- **Effect**: Safety timeout to prevent infinite search

---

## 2. Detection Algorithm Parameters

### `lag` (default: 8)
- **Location**: `detector.configure_detection()`
- **Purpose**: Window size for running statistics (smoothing)
- **Tuning range**: 5 - 15
- **Effect**: 
  - Smaller = more responsive, more noise
  - Larger = smoother, slower to detect edges

### `threshold` (default: 1.5) ⚠️ CRITICAL
- **Location**: `detector.configure_detection()`
- **Purpose**: Z-score threshold for peak detection (edge sensitivity)
- **Tuning range**: 1.0 - 2.5
- **Effect**: 
  - Lower (1.0-1.3) = more sensitive, may detect false edges
  - Higher (1.8-2.5) = less sensitive, may miss edges
  - **Start with 1.5, decrease if missing edges, increase if too many false positives**

### `influence` (default: 0.2)
- **Location**: `detector.configure_detection()`
- **Purpose**: How much detected peaks affect running statistics
- **Tuning range**: 0.1 - 0.3
- **Effect**: 
  - Lower = peaks have less influence on mean (more robust)
  - Higher = peaks affect future detection more

### `min_peak_height` (default: 0.05m) ⚠️ CRITICAL
- **Location**: `detector.configure_detection()`
- **Purpose**: Minimum height difference to consider as pad edge
- **Tuning range**: 0.03 - 0.08m
- **Effect**: 
  - Pad is 0.1m high, so threshold should be 0.03-0.08m
  - Lower = more sensitive to small height changes
  - Higher = only detects significant height changes
  - **Start with 0.05m (50% of pad height)**

---

## 3. Speed Parameters

### `cruise_speed` (default: 0.2 m/s)
- **Location**: `RealHardwareHoverNav.__init__`
- **Purpose**: Speed during normal navigation
- **Already tuned, no change needed for detection**

### `searching_speed` (default: 0.1 m/s)
- **Location**: `RealHardwareHoverNav.__init__`
- **Purpose**: Speed during search pattern
- **Tuning range**: 0.05 - 0.15 m/s
- **Effect**: 
  - Slower = more measurements per area, better detection
  - Faster = quicker search, fewer measurements
  - **0.1 m/s is good balance**

### `max_speed_near_obstacle` (default: 0.15 m/s)
- **Location**: `RealHardwareHoverNav.__init__`
- **Purpose**: Speed limit when near obstacles
- **Already tuned for safety**

---

## 4. Confidence Parameters

These are in `detection.py` and generally don't need tuning:

### `min_border_points` (in `is_ready_for_landing`)
- **Default**: 6 points
- **Purpose**: Minimum edge detections before calculating center
- **Location**: `detection.py:201`

### `confidence_threshold` (in `is_ready_for_landing`)
- **Default**: 0.6
- **Purpose**: Minimum confidence to accept calculated center
- **Location**: `detection.py:208`

---

## Recommended Tuning Procedure

### Step 1: Basic Detection Test
1. Start with default parameters
2. Run mission and observe console output
3. Check: "Detection stats: X edges detected"
4. **Goal**: Detect 4-8 edges (one per pad side, possibly multiple per side)

### Step 2: If Too Few Edges Detected (< 4)
- **Decrease** `threshold` from 1.5 → 1.2
- **Decrease** `min_peak_height` from 0.05 → 0.03
- **Decrease** `searching_speed` from 0.1 → 0.08
- **Decrease** `search_step_size` from 0.15 → 0.12

### Step 3: If Too Many False Edges (> 15)
- **Increase** `threshold` from 1.5 → 1.8
- **Increase** `min_peak_height` from 0.05 → 0.07
- **Increase** `lag` from 8 → 10

### Step 4: If Search Takes Too Long
- **Increase** `search_step_size` from 0.15 → 0.18
- **Decrease** `search_area_size` from 0.5 → 0.45
- **Increase** `searching_speed` from 0.1 → 0.12

### Step 5: If Calculated Center is Inaccurate
- Collect more edges (tune per Step 2)
- Check ground surface is flat
- Verify z-range sensor is working correctly

---

## Debugging Output

During operation, the system prints:
- `Starting search mode at (x, y)` - Search initiated
- `Platform edge detected at (x, y)` - Edge found
- `Search progress: N/M waypoints, X edges detected` - Every 2 seconds
- `Detection stats: X border points` - After search
- `Pad center calculated: (x, y), confidence: 0.XX` - Final result

### What to Look For:
- **No edges detected**: Increase sensitivity (decrease threshold/min_peak_height)
- **Many edges at same location**: Increase lag or influence
- **Center way off**: Need more edges or better coverage
- **Low confidence (<0.5)**: Edges are scattered, improve detection

---

## Quick Reference Table

| Parameter | Default | Increase to... | Decrease to... |
|-----------|---------|----------------|----------------|
| `threshold` | 1.5 | Reduce false positives | Detect more edges |
| `min_peak_height` | 0.05m | Filter noise | Be more sensitive |
| `searching_speed` | 0.1 m/s | Search faster | Get more data |
| `search_step_size` | 0.15m | Search faster | Better coverage |
| `lag` | 8 | Smoother detection | More responsive |

---

## Testing Checklist

- [ ] Drone navigates to landing zone successfully
- [ ] Search mode activates at correct distance
- [ ] Cross pattern executes (forward-back-left-right)
- [ ] Z-range sensor readings are valid (0.05-1.0m)
- [ ] Edges detected (aim for 4-8 detections)
- [ ] Pad center calculated with confidence >0.6
- [ ] Drone moves to calculated center
- [ ] Landing occurs at correct location
- [ ] Obstacle avoidance still works during search

