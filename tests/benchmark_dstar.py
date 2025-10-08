#!/usr/bin/env python3
"""
Benchmark D* Lite replanning performance
"""
import time
import numpy as np
from cfpilot.mapping import GridMap
from cfpilot.planning.DStarLite.d_star_lite import DStarLite

def benchmark_replanning():
    """Benchmark replanning speed with multiple obstacle additions"""
    print("=" * 70)
    print("D* Lite Replanning Performance Benchmark")
    print("=" * 70)
    
    # Create a larger grid for more realistic scenario
    map_width = 100
    map_height = 100
    resolution = 0.5  # 0.5m per cell
    
    grid_map = GridMap(map_width, map_height, resolution, 
                      25.0, 25.0, init_val=0.0)
    
    # Add boundaries and some obstacles
    grid_map.occupy_boundaries(boundary_width=1, val=1.0)
    
    # Add some initial obstacles
    for i in range(20, 60):
        grid_map.set_value_from_xy_index(30, i, 1.0)
    
    for i in range(40, 80):
        grid_map.set_value_from_xy_index(60, i, 1.0)
    
    print(f"\nMap: {map_width}x{map_height} cells, resolution={resolution}m")
    print(f"Physical size: {map_width*resolution}m x {map_height*resolution}m")
    
    # Create planner
    planner = DStarLite(grid_map, obstacle_threshold=0.5, show_animation=False)
    
    # Initial planning
    start = (5.0, 5.0)
    goal = (45.0, 45.0)
    
    print(f"\nInitial planning from {start} to {goal}...")
    t0 = time.time()
    planner.initialize(start[0], start[1], goal[0], goal[1])
    planner.compute_shortest_path()
    t1 = time.time()
    
    initial_time = t1 - t0
    print(f"  Initial planning time: {initial_time*1000:.2f} ms")
    
    initial_path = planner.compute_current_path()
    print(f"  Initial path length: {len(initial_path)} waypoints")
    
    # Test replanning with different numbers of obstacles
    test_scenarios = [
        ("Small change", 5),
        ("Medium change", 20),
        ("Large change", 50),
        ("Very large change", 100),
    ]
    
    print("\n" + "-" * 70)
    print("Replanning Performance Tests:")
    print("-" * 70)
    
    for scenario_name, num_obstacles in test_scenarios:
        # Generate random obstacles
        obstacles = []
        for _ in range(num_obstacles):
            x = np.random.randint(10, 90)
            y = np.random.randint(10, 90)
            obstacles.append((x, y))
        
        # Add obstacles to map
        for x, y in obstacles:
            grid_map.set_value_from_xy_index(x, y, 1.0)
        
        # Time the replanning
        t0 = time.time()
        planner.update_map(obstacles)
        t1 = time.time()
        
        replan_time = t1 - t0
        
        # Try to compute new path
        try:
            new_path = planner.compute_current_path()
            path_length = len(new_path)
            success = True
        except:
            path_length = 0
            success = False
        
        print(f"\n{scenario_name}: {num_obstacles} new obstacles")
        print(f"  Replanning time: {replan_time*1000:.2f} ms")
        print(f"  Speed ratio: {initial_time/replan_time:.1f}x faster than initial" if replan_time > 0 else "  N/A")
        print(f"  New path: {path_length} waypoints" if success else "  No path found")
   
   
if __name__ == "__main__":
    benchmark_replanning()
