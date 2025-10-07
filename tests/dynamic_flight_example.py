"""
Dynamic Obstacle Detection and Avoidance Flight Example

This script demonstrates:
1. Marking grid cells as occupied from multiranger readings
2. Real-time obstacle avoidance (stop if too close)
3. Dynamic path replanning when obstacles detected
"""
import time
import logging
import numpy as np
import cflib.crtp
from cflib.positioning.position_hl_commander import PositionHlCommander
from cflib.utils.multiranger import Multiranger

from cfpilot.controller import CrazyflieController
from path_planning import (
    create_test_environment,
    plan_path,
    simplify_path,
    update_obstacles_from_multiranger,
    check_obstacle_proximity,
    is_path_clear,
    dynamic_replan,
    visualize_path
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create grid map with initial obstacles
grid_map = create_test_environment(width=5.0, height=3.0)
start = (0.7, 1.5)
goal = (4.5, 2.5)

# Plan initial path
path_found, pathx, pathy = plan_path(grid_map, start, goal)
pathx_simp, pathy_simp = simplify_path(pathx, pathy, grid_map)
pos_x, pos_y = grid_map.get_xy_poss_from_xy_indexes(pathx_simp, pathy_simp)
map_fig = visualize_path(grid_map, start_pos=start, goal_pos=goal, show_smooth=False);
logger.info(f"✅ Initial path: {len(pos_x)} waypoints")

# Initialize controller
controller = CrazyflieController(enable_plotting=False)
config = controller.config

# Initialize drivers
cflib.crtp.init_drivers()
logger.info("✅ Drivers initialized")

# Connect to Crazyflie
uri = config['connection']['uri']
time.sleep(0.5)
controller.connect(uri, x=start[0], y=start[1], z=0.0, yaw=0.0)

if not controller.wait_for_connection(timeout=10.0):
    logger.error("❌ Connection failed!")
    raise Exception("Could not connect to Crazyflie")

logger.info("✅ Connected successfully!")

try:
    # Setup position commander
    pc = PositionHlCommander(
        crazyflie=controller.cf,
        x=start[0],
        y=start[1],
        z=0.1,
        default_velocity=0.2,
        default_height=0.5,
        controller=PositionHlCommander.CONTROLLER_PID
    )
    
    # Start multiranger
    multiranger = Multiranger(controller.cf)
    multiranger.start()
    logger.info("✅ Multiranger started")
    
    # Takeoff
    logger.info("🚁 Taking off...")
    height = 0.5
    pc.take_off(height=height, velocity=0.5)
    time.sleep(2)
    
    # Navigation with dynamic replanning and obstacle avoidance
    waypoint_idx = 0
    replan_count = 0
    detection_history = {}
    avoidance_threshold = 0.3  # Stop if obstacle < 0.5m
    
    while waypoint_idx < 3:
        # Get current position
        latest = controller.get_latest_data()
        current_x = latest.get('stateEstimate.x', 0)
        current_y = latest.get('stateEstimate.y', 0)
        current_yaw = latest.get('stabilizer.yaw', 0)
        
        logger.info(f"Current position: ({current_x:.2f}, {current_y:.2f}), yaw: {current_yaw:.2f}")
        # REAL-TIME AVOIDANCE: Check immediate obstacle proximity
        # obstacles_close = check_obstacle_proximity(multiranger, avoidance_threshold)
        
        # if obstacles_close:
        #     logger.warning(f"⚠️ OBSTACLE TOO CLOSE: {obstacles_close}")
        #     logger.warning("🛑 STOPPING - Holding position...")
            
        #     # Hold current position (stop moving)
        #     pc.go_to(current_x, current_y, height, velocity=0.1)
        #     time.sleep(0.5)
            
        #     # Emergency replan from current position
        #     try:
        #         pos_x, pos_y = dynamic_replan(grid_map, (current_x, current_y), goal)
        #         waypoint_idx = 0
        #         replan_count += 1
        #         logger.info(f"✅ Emergency replan #{replan_count}: {len(pos_x)} waypoints")
        #     except ValueError as e:
        #         logger.error(f"❌ Cannot replan: {e}")
        #         break
        #     continue
        
        # # Update grid map: mark cells as occupied from sensor readings
        # newly_occupied, detection_history = update_obstacles_from_multiranger(
        #     grid_map,
        #     multiranger,
        #     current_x,
        #     current_y,
        #     current_yaw,
        #     max_range=3.0,
        #     detection_history=detection_history,
        #     confirmation_threshold=3  # Need 3 detections to confirm
        # )
        
        # if newly_occupied > 0:
        #     logger.info(f"🔍 Marked {newly_occupied} new occupied cells")
            
        #     # Check if path still valid after map update
        #     if not is_path_clear(pos_x, pos_y, grid_map, waypoint_idx, look_ahead=3):
        #         logger.warning("⚠️ Path blocked by new obstacles! Replanning...")
        #         try:
        #             pos_x, pos_y = dynamic_replan(grid_map, (current_x, current_y), goal)
        #             waypoint_idx = 0
        #             replan_count += 1
        #             logger.info(f"✅ Replan #{replan_count}: {len(pos_x)} waypoints")
        #         except ValueError as e:
        #             logger.error(f"❌ Cannot replan: {e}")
        #             break
        
        # Navigate to next waypoint
        target_x, target_y = pos_x[waypoint_idx], pos_y[waypoint_idx]
        logger.info(f"🎯 Waypoint {waypoint_idx+1}/{len(pos_x)}: ({target_x:.2f}, {target_y:.2f})")
        

        
        # Check if reached waypoint
        distance = np.sqrt((current_x - target_x)**2 + (current_y - target_y)**2)
        if distance < 0.1:
            waypoint_idx += 1
            logger.info(f"✅ Reached waypoint {waypoint_idx}/{len(pos_x)}")
            logger.info(f"Distance to waypoint: {distance:.2f}")
        else:
            logger.info(f"Moving to waypoint: ({target_x:.2f}, {target_y:.2f})")
            pc.go_to(target_x, target_y, height, velocity=0.1)
            time.sleep(0.1)
    
    logger.info(f"🎉 Navigation complete! Total replans: {replan_count}")
    
    # Land
    logger.info("🛬 Landing...")
    pc.land()
    time.sleep(3)
    
    # Stop multiranger
    multiranger.stop()
    controller.save_flight_data()
    logger.info("✅ Flight complete!")
    
except Exception as e:
    logger.error(f"❌ Error during flight: {e}")
    import traceback
    traceback.print_exc()
    try:
        pc.land()
    except:
        pass
    controller.emergency_shutdown()
    raise
finally:
    controller.cleanup()
    controller.disconnect()

