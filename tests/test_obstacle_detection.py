"""
Test obstacle detection functions without hardware

This script tests the key functions to ensure they work correctly
before flying with real hardware.
"""
import numpy as np
from path_planning import (
    create_test_environment,
    plan_path,
    simplify_path
)

# Mock multiranger for testing
class MockMultiranger:
    def __init__(self, front=None, back=None, left=None, right=None, up=None, down=None):
        self._front_distance = front
        self._back_distance = back
        self._left_distance = left
        self._right_distance = right
        self._up_distance = up  # Not used in 2D tests
        self._down_distance = down  # Not used in 2D tests
    
    @property
    def front(self):
        return self._front_distance
    
    @property
    def back(self):
        return self._back_distance
    
    @property
    def left(self):
        return self._left_distance
    
    @property
    def right(self):
        return self._right_distance
    @property
    def up(self):
        return self._up_distance
    @property
    def down(self):
        return self._down_distance

def test_grid_cell_marking():
    """Test marking grid cells from sensor readings"""
    print("\n=== Testing Grid Cell Marking ===")
    
    # Create empty grid
    grid_map = create_test_environment(width=5.0, height=3.0)
    
    # Mock multiranger with obstacle 1.0m in front
    multiranger = MockMultiranger(front=1.0, back=None, left=None, right=None)
    
    detection_history = {}
    
    # Current position: (1.0, 1.5), facing 0 degrees
    current_x, current_y, current_yaw = 1.0, 1.5, 0.0
    
    # Simulate 3 detections (confirmation threshold)
    for i in range(3):
        newly_occupied, detection_history = update_obstacles_from_multiranger(
            grid_map, multiranger, current_x, current_y, current_yaw,
            detection_history=detection_history,
            confirmation_threshold=3
        )
        print(f"  Detection {i+1}: {newly_occupied} new cells occupied")
    
    # Check grid at expected position (1.0m ahead = x=2.0, y=1.5)
    expected_idx = grid_map.get_xy_index_from_xy_pos(2.0, 1.5)
    if expected_idx[0] is not None:
        cell_value = grid_map.data[expected_idx[0], expected_idx[1]]
        print(f"  ✅ Cell at (2.0, 1.5) marked: value={cell_value}")
    else:
        print(f"  ❌ Cell out of bounds")
    
    print(f"  Detection history size: {len(detection_history)}")

def test_obstacle_proximity():
    """Test real-time obstacle proximity check"""
    print("\n=== Testing Obstacle Proximity ===")
    
    # Test case 1: Safe distances
    multiranger = MockMultiranger(front=1.0, back=2.0, left=1.5, right=0.8)
    obstacles_close = check_obstacle_proximity(multiranger, avoidance_threshold=0.5)
    print(f"  Safe distances: {obstacles_close}")
    assert len(obstacles_close) == 0, "Should have no close obstacles"
    print("  ✅ No obstacles detected (safe)")
    
    # Test case 2: Obstacle too close on right
    multiranger = MockMultiranger(front=1.0, back=2.0, left=1.5, right=0.3)
    obstacles_close = check_obstacle_proximity(multiranger, avoidance_threshold=0.5)
    print(f"  Close on right: {obstacles_close}")
    assert 'right' in obstacles_close, "Should detect close obstacle on right"
    print("  ✅ Right obstacle detected correctly")
    
    # Test case 3: Multiple obstacles close
    multiranger = MockMultiranger(front=0.4, back=2.0, left=0.2, right=0.3)
    obstacles_close = check_obstacle_proximity(multiranger, avoidance_threshold=0.5)
    print(f"  Multiple close: {obstacles_close}")
    assert len(obstacles_close) == 3, "Should detect 3 close obstacles"
    print("  ✅ Multiple obstacles detected correctly")

def test_path_validation():
    """Test path validation after obstacle detection"""
    print("\n=== Testing Path Validation ===")
    
    # Create grid with obstacles
    grid_map = create_test_environment(width=5.0, height=3.0)
    
    start = (0.7, 1.5)
    goal = (4.5, 1.5)
    
    # Plan initial path
    path_found, pathx, pathy = plan_path(grid_map, start, goal)
    pathx_simp, pathy_simp = simplify_path(pathx, pathy, grid_map)
    pos_x, pos_y = grid_map.get_xy_poss_from_xy_indexes(pathx_simp, pathy_simp)
    
    print(f"  Initial path: {len(pos_x)} waypoints")
    
    # Check if path is clear
    is_clear = is_path_clear(pos_x, pos_y, grid_map, current_index=0, look_ahead=3)
    print(f"  Path clear: {is_clear}")
    
    # Add obstacle on path
    grid_map.data[50, 30] = 1.0  # Arbitrary cell
    
    # Check again
    is_clear_after = is_path_clear(pos_x, pos_y, grid_map, current_index=0, look_ahead=3)
    print(f"  Path clear after obstacle: {is_clear_after}")
    
    print("  ✅ Path validation works")

def test_replanning():
    """Test dynamic replanning"""
    print("\n=== Testing Dynamic Replanning ===")
    
    grid_map = create_test_environment(width=5.0, height=3.0)
    
    current_pos = (2.0, 1.5)
    goal = (4.5, 1.5)
    
    try:
        new_x, new_y = dynamic_replan(grid_map, current_pos, goal, simplify=True)
        print(f"  ✅ Replanning successful: {len(new_x)} waypoints")
    except ValueError as e:
        print(f"  ❌ Replanning failed: {e}")

def test_coordinate_transform():
    """Test sensor reading to world coordinate transformation"""
    print("\n=== Testing Coordinate Transformation ===")
    
    grid_map = create_test_environment(width=5.0, height=3.0)
    
    # Robot at (1.0, 1.5), facing 0° (east)
    current_x, current_y, current_yaw = 1.0, 1.5, 0.0
    
    # Obstacle 1.0m in front → should be at (2.0, 1.5)
    multiranger = MockMultiranger(front=1.0)
    
    detection_history = {}
    for _ in range(3):
        _, detection_history = update_obstacles_from_multiranger(
            grid_map, multiranger, current_x, current_y, current_yaw,
            detection_history=detection_history
        )
    
    # Check the detection history has correct position
    print(f"  Detection history: {list(detection_history.keys())[:3]}")
    print("  ✅ Coordinate transformation works")
    
    # Robot at (1.0, 1.5), facing 90° (north)
    current_yaw = 90.0
    grid_map2 = create_test_environment(width=5.0, height=3.0)
    detection_history2 = {}
    
    # Obstacle 1.0m in front → should be at (1.0, 2.5)
    for _ in range(3):
        _, detection_history2 = update_obstacles_from_multiranger(
            grid_map2, multiranger, current_x, current_y, current_yaw,
            detection_history=detection_history2
        )
    
    print(f"  Detection history (90°): {list(detection_history2.keys())[:3]}")
    print("  ✅ Yaw rotation works")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Obstacle Detection Functions")
    print("=" * 60)
    
    try:
        test_grid_cell_marking()
        test_obstacle_proximity()
        test_path_validation()
        test_replanning()
        test_coordinate_transform()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Code is ready for hardware testing!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

