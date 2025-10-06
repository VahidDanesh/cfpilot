"""
Waypoint Following Flight Test

Complete path planning and autonomous navigation test:
1. Create grid map with obstacles
2. Plan path using D* Lite
3. Simplify path to remove unnecessary waypoints
4. Follow waypoints with Crazyflie
5. Land at goal position

Press Ctrl+C at any time to abort and land safely.
"""
import time
import signal
import sys
import logging

import cflib.crtp
from cflib.positioning.position_hl_commander import PositionHlCommander
from cflib.utils.multiranger import Multiranger

from cfpilot.controller import CrazyflieController
from path_planning import (
    create_test_environment, 
    plan_path, 
    simplify_path, 
    visualize_path
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for emergency shutdown
controller = None
pc = None
multiranger = None
abort_requested = False


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global abort_requested
    logger.warning("\n⚠️  ABORT requested! Landing safely...")
    abort_requested = True


def emergency_land():
    """Emergency landing procedure"""
    global pc, multiranger, controller
    
    try:
        if pc is not None:
            logger.info("🛬 Emergency landing...")
            pc.land()
            time.sleep(2)
    except Exception as e:
        logger.error(f"Error during emergency landing: {e}")
    
    try:
        if multiranger is not None:
            multiranger.stop()
    except Exception as e:
        logger.error(f"Error stopping multiranger: {e}")
    
    try:
        if controller is not None:
            controller.save_flight_data()
            controller.cleanup()
            controller.disconnect()
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


def main():
    """Main flight execution"""
    global controller, pc, multiranger, abort_requested
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # ===== CONFIGURATION =====
    start_pos = (0.7, 1.3)
    goal_pos = (3.9, 1.0)
    flight_height = 0.5
    waypoint_velocity = 0.3
    mission_timeout = 30.0  # seconds
    
    logger.info("=" * 60)
    logger.info("🚁 WAYPOINT FOLLOWING FLIGHT TEST")
    logger.info("=" * 60)
    logger.info(f"Start position: {start_pos}")
    logger.info(f"Goal position: {goal_pos}")
    logger.info(f"Flight height: {flight_height}m")
    logger.info(f"Press Ctrl+C at any time to abort safely")
    logger.info("=" * 60)
    
    # ===== PATH PLANNING =====
    logger.info("\n📍 Step 1: Path Planning")
    logger.info("-" * 40)
    
    grid_map = create_test_environment()
    logger.info("✅ Grid map created")
    
    path_found, pathx, pathy = plan_path(grid_map, start_pos, goal_pos)
    if not path_found:
        logger.error("❌ No path found!")
        return False
    logger.info(f"✅ Path found with {len(pathx)} points")
    
    # Simplify path
    pathx_simp, pathy_simp = simplify_path(pathx, pathy, grid_map)
    logger.info(f"✅ Path simplified to {len(pathx_simp)} waypoints")
    
    # Convert to world coordinates
    pos_x, pos_y = grid_map.get_xy_poss_from_xy_indexes(pathx_simp, pathy_simp)
    
    # Visualize (optional - comment out if running headless)
    try:
        visualize_path(grid_map, start_pos, goal_pos, show_smooth=False)
    except Exception as e:
        logger.warning(f"Could not visualize path: {e}")
    
    # Print waypoints
    logger.info("\n📋 Waypoints to follow:")
    for i, (x, y) in enumerate(zip(pos_x, pos_y)):
        logger.info(f"  {i+1}. ({x:.3f}, {y:.3f})")
    
    # ===== CRAZYFLIE CONNECTION =====
    logger.info("\n🔌 Step 2: Connecting to Crazyflie")
    logger.info("-" * 40)
    
    controller = CrazyflieController(enable_plotting=False)
    config = controller.config
    
    cflib.crtp.init_drivers()
    uri = config['connection']['uri']
    logger.info(f"Connecting to {uri}...")
    
    controller.connect(uri)
    
    if not controller.wait_for_connection(timeout=10.0):
        logger.error("❌ Connection failed!")
        return False
    
    logger.info("✅ Connected successfully!")
    
    # Wait for logging data
    logger.info("⏳ Waiting for sensor data...")
    timeout_start = time.time()
    while len(controller.get_latest_data()) == 0:
        if time.time() - timeout_start > 5.0:
            logger.error("❌ No sensor data received!")
            return False
        time.sleep(0.1)
    logger.info("✅ Sensor data streaming")
    
    # ===== FLIGHT EXECUTION =====
    logger.info("\n🚁 Step 3: Flight Execution")
    logger.info("-" * 40)
    
    try:
        start_time = time.time()
        
        # Setup position commander
        pc = PositionHlCommander(
            crazyflie=controller.cf,
            x=start_pos[0],
            y=start_pos[1],
            z=0.0,
            default_velocity=0.2,
            default_height=flight_height,
            controller=2
        )
        logger.info("✅ Position commander initialized")
        
        # Start multiranger
        multiranger = Multiranger(controller.cf)
        multiranger.start()
        logger.info("✅ Multiranger started")
        time.sleep(0.5)
        
        # Takeoff
        if abort_requested:
            logger.warning("Abort requested before takeoff")
            return False
        
        logger.info(f"🛫 Taking off to {flight_height}m...")
        pc.take_off(height=flight_height, velocity=0.5)
        time.sleep(3)  # Wait for stable takeoff
        
        latest = controller.get_latest_data()
        current_z = latest.get('stateEstimate.z', 0)
        logger.info(f"✅ Airborne at z={current_z:.2f}m")
        
        # Follow waypoints
        logger.info(f"\n🗺️  Following {len(pos_x)} waypoints...")
        logger.info("-" * 40)
        
        for i, (target_x, target_y) in enumerate(zip(pos_x, pos_y)):
            # Check for abort or timeout
            if abort_requested:
                logger.warning("⚠️  Abort requested during flight")
                break
            
            if time.time() - start_time > mission_timeout:
                logger.warning("⏰ Mission timeout reached")
                break
            
            logger.info(f"\n🎯 Waypoint {i+1}/{len(pos_x)}: ({target_x:.3f}, {target_y:.3f})")
            
            # Move to waypoint
            pc.go_to(target_x, target_y, flight_height, velocity=waypoint_velocity)
            time.sleep(2)  # Allow time to reach waypoint
            
            # Verify position
            latest = controller.get_latest_data()
            current_x = latest.get('stateEstimate.x', 0)
            current_y = latest.get('stateEstimate.y', 0)
            current_z = latest.get('stateEstimate.z', 0)
            
            distance = ((current_x - target_x)**2 + (current_y - target_y)**2)**0.5
            
            logger.info(f"   📍 Position: ({current_x:.3f}, {current_y:.3f}, {current_z:.3f})")
            logger.info(f"   📏 Distance to target: {distance:.3f}m")
            
            # Optional: Fine adjustment for last waypoint (goal)
            if i == len(pos_x) - 1 and distance > 0.15:
                logger.info(f"   🔧 Fine adjustment for landing position...")
                pc.go_to(target_x, target_y, flight_height, velocity=0.1)
                time.sleep(2)
                
                latest = controller.get_latest_data()
                current_x = latest.get('stateEstimate.x', 0)
                current_y = latest.get('stateEstimate.y', 0)
                distance = ((current_x - target_x)**2 + (current_y - target_y)**2)**0.5
                logger.info(f"   📏 Final distance: {distance:.3f}m")
            
            logger.info(f"   ✅ Waypoint {i+1} reached")
        
        # Final position check
        logger.info("\n📍 Final Position Check")
        logger.info("-" * 40)
        latest = controller.get_latest_data()
        final_x = latest.get('stateEstimate.x', 0)
        final_y = latest.get('stateEstimate.y', 0)
        final_z = latest.get('stateEstimate.z', 0)
        
        distance_to_goal = ((final_x - goal_pos[0])**2 + (final_y - goal_pos[1])**2)**0.5
        
        logger.info(f"Current: ({final_x:.3f}, {final_y:.3f}, {final_z:.3f})")
        logger.info(f"Goal: ({goal_pos[0]:.3f}, {goal_pos[1]:.3f})")
        logger.info(f"Distance to goal: {distance_to_goal:.3f}m")
        
        # Landing
        logger.info("\n🛬 Landing...")
        logger.info("-" * 40)
        pc.land(velocity=0.3)
        
        # Monitor landing
        for _ in range(6):
            time.sleep(0.5)
            latest = controller.get_latest_data()
            z = latest.get('stateEstimate.z', 0)
            logger.info(f"   Altitude: {z:.3f}m")
            if z < 0.05:
                break
        
        logger.info("✅ Landed")
        time.sleep(1)
        
        # Stop multiranger
        multiranger.stop()
        logger.info("✅ Multiranger stopped")
        
        # Save data
        controller.save_flight_data()
        logger.info("✅ Flight data saved")
        
        # Success summary
        logger.info("\n" + "=" * 60)
        logger.info("✅ MISSION COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Duration: {time.time() - start_time:.1f}s")
        logger.info(f"Waypoints followed: {len(pos_x)}")
        logger.info(f"Final distance to goal: {distance_to_goal:.3f}m")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Error during flight: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        logger.info("\n🧹 Cleanup...")
        try:
            if pc is not None:
                pc.land()
                time.sleep(1)
        except:
            pass
        
        try:
            if multiranger is not None:
                multiranger.stop()
        except:
            pass
        
        try:
            if controller is not None:
                controller.cleanup()
                controller.disconnect()
        except:
            pass
        
        logger.info("✅ Cleanup complete")


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user")
        emergency_land()
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        emergency_land()
        sys.exit(1)
