import logging
import sys
import time
from threading import Event

import numpy as np
import matplotlib.pyplot as plt

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils.multiranger import Multiranger
from cflib.crazyflie.log import LogConfig

"-----------------------------------------------------------------------------------------------------------------------"

# --- CONNECTION SETUP ---
URI = "radio://0/88/2M/E7E7E7E7F0"
deck_attached_event = Event()
logging.basicConfig(level=logging.ERROR)

# --- PARAMETERS ---
HOVER_PAUSE = 2.0           # seconds
HOVER_HEIGHT = 0.5          # meters
BOUNDARY_THRESHOLD = 0.5   # meters (for detecting objects/walls)
EDGE_TOL = 0.5             # meters
VELOCITY = 0.2              # meters/second
GRID_SIZE = 0.4            # meters
EDGE_POINTS = 4            # number of distance readings per edge
EDGE_SCAN_DISTANCE = 1.5   # meters to travel while scanning
SWEEP_WIDTH = 1.0          # meters
SWEEP_DEPTH = 1.5          # meters

"-----------------------------------------------------------------------------------------------------------------------"

def param_deck_flow(_, value_str):
    value = int(value_str)
    print(value)
    if value:
        deck_attached_event.set()
        print('Deck is attached!')
    else:
        print('Deck is NOT attached!')

# --- Globals for distance tracking ---
position_estimate = [0, 0]  # [x, y] global position estimate
total_distance = 0.0        # total path length traveled
last_position = None        # last recorded position (for incremental distance)
logging_active = True       # flag to control whether the log callback prints

def log_pos_callback(timestamp, data, logconf):
    global position_estimate, total_distance, last_position, logging_active

    # If logging has been disabled (e.g. after landing), do nothing
    if not logging_active:
        return

    x = data.get("stateEstimate.x", 0.0)
    y = data.get("stateEstimate.y", 0.0)

    # Compute incremental distance (Euclidean)
    if last_position is not None:
        dx = x - last_position[0]
        dy = y - last_position[1]
        step_distance = np.sqrt(dx**2 + dy**2)
        total_distance += step_distance

    # Update trackers
    last_position = [x, y]
    position_estimate = [x, y]

    # Print distance traveled
    print(f"📍Distance traveled: {total_distance:.3f} m (X={x:.2f}, Y={y:.2f})")

"-----------------------------------------------------------------------------------------------------------------------"

# --- Helper functions ---
def is_close(distance, threshold=BOUNDARY_THRESHOLD):
    """Check if an obstacle is within a certain threshold distance."""
    return distance is not None and distance < threshold


def detect_edge(mr):
    """Detect pad edge when downward distance differs."""
    if mr.down is None:
        return False
    return abs(mr.down) < EDGE_TOL

"-----------------------------------------------------------------------------------------------------------------------"

# --- Search Sweep Function ---
def search_sweep(mc, mr, velocity=VELOCITY, width=SWEEP_WIDTH, depth=SWEEP_DEPTH, step_size=0.2):
    """
    Snake-pattern search:
      - width: the length of each sweep row (meters)
      - depth: total distance to cover perpendicular to the sweep (meters)
      - step_size: how far to move between rows
    """

    print(f"\n🔍 Starting  {width:.2f} x {depth:.2f} m snake-pattern search for first pad edge...")
    plt.ion()
    plt.figure(figsize=(6, 6))
    plt.title("Initial Pad Search Sweep")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True)

    # Number of rows (how many back-and-forth sweeps)
    num_rows = int(np.ceil(width / step_size))   # use ceil to avoid skipping a partial last row
    direction = 1

    for i in range(num_rows):
        print(f"➡️ Sweep row {i + 1}/{num_rows} ({'→' if direction > 0 else '←'})")

        # Time to travel across the width at the given velocity
        travel_time = depth / velocity
        step_start = time.time()

        while time.time() - step_start < travel_time:
            # Move along the sweep axis; change axes here if your coordinate frame
            # uses Y for sweep and X for stepping. This uses X for sweep (forward/back).
            mc.start_linear_motion(velocity * direction, 0, 0)
            time.sleep(0.05)

            if detect_edge(mr):
                x, y = position_estimate
                mc.stop()
                print(f"🟥 Edge detected at X={x:.2f}, Y={y:.2f}")
                plt.scatter(x, y, c="red", s=50, label="First Edge")
                plt.pause(0.01)

                # <<< Added hover pause
                print("⏸ Hovering for 2 seconds...")
                hover_start = time.time()
                while time.time() - hover_start < HOVER_PAUSE:
                    mc.start_linear_motion(0, 0, 0)  # actively hold position
                    time.sleep(0.1)
                mc.stop()
                # <<< End hover pause

                return (x, y)

            mc.stop()

        # Move one row over (perpendicular to sweep) before the next sweep
        if i < num_rows - 1:
            # Here we step along the perpendicular axis (Y) by step_size
            mc.start_linear_motion(0, step_size, 0)
            time.sleep(step_size / velocity)
            mc.stop()
            direction *= -1

    print(f"⚠️ No edges detected in {width:.2f} x {depth:.2f} m area.")
    plt.show(block=False)
    return None

"-----------------------------------------------------------------------------------------------------------------------"

# --- Landing Pad Edge Detection ---
def detect_pad_edges(mc, mr, velocity=VELOCITY, grid_size=GRID_SIZE, step_size=0.1):
    """
    Performs a grid search around the first edge to collect up to EDGE_POINTS edges,
    or stops after 30 seconds if not enough are found.
    """
    edges = []
    plt.ion()
    plt.figure(figsize=(6, 6))
    plt.title("Landing Pad Edge Detection")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True)

    print("\n🚀 Moving to detect pad edges...")

    # Track elapsed time
    start_time = time.time()
    time_limit = 30.0  # seconds

    # === Phase 2: Snake-pattern search for remaining edges ===
    print(f"\n🐍 Starting {grid_size:.2f} x {grid_size:.2f} m snake-pattern edge scan...")
    num_steps = int(np.ceil(grid_size / step_size))
    direction = 1

    while len(edges) < EDGE_POINTS and (time.time() - start_time < time_limit):
        for i in range(num_steps):
            for j in range(num_steps):
                # Check time limit inside the loop
                if time.time() - start_time >= time_limit:
                    print("⏰ Time limit reached during edge scan.")
                    break

                # Sweep along Y for this step
                mc.start_linear_motion(0, direction * velocity, 0)
                time.sleep(step_size / velocity)
                mc.stop()

                if detect_edge(mr):
                    x, y = position_estimate
                    # Only record distinct edge points
                    if len(edges) == 0 or np.linalg.norm(np.array([x, y]) - np.array(edges[-1])) > 0.05:
                        edges.append((x, y))
                        plt.scatter(x, y, c='red', s=50)
                        plt.pause(0.01)
                        print(f"🟥 Edge #{len(edges)} detected at X={x:.2f}, Y={y:.2f}")

                        # <<< Hover pause for 2 seconds
                        print("⏸ Hovering for 2 seconds...")
                        hover_start = time.time()
                        while time.time() - hover_start < HOVER_PAUSE:
                            mc.start_linear_motion(0, 0, 0)
                            time.sleep(0.1)
                        mc.stop()

                        if len(edges) >= EDGE_POINTS:
                            print(f"✅ {EDGE_POINTS} edges found — stopping scan.")
                            break

            if len(edges) >= EDGE_POINTS or (time.time() - start_time >= time_limit):
                break

            # Move forward (X) one step before next strip
            mc.start_linear_motion(velocity, 0, 0)
            time.sleep(step_size / velocity)
            mc.stop()
            direction *= -1

        if len(edges) >= EDGE_POINTS or (time.time() - start_time >= time_limit):
            break

    # === Phase 3: Compute and plot pad center ===
    if len(edges) >= EDGE_POINTS:
        edges_np = np.array(edges)
        x_center, y_center = np.mean(edges_np[:, 0]), np.mean(edges_np[:, 1])
        plt.scatter(x_center, y_center, c='green', marker='x', s=100, label='Pad Center')
        plt.legend()
        plt.pause(0.01)
        print(f"🎯 Estimated pad center at X={x_center:.2f}, Y={y_center:.2f}")
        plt.show(block=False)
        return edges, (x_center, y_center)
    else:
        print(f"⚠️ Only {len(edges)} edges found in 30 seconds — initiating landing.")
        plt.show(block=False)

        # Land safely after 30 seconds timeout
        print("🛬 Time limit reached — initiating safe landing.")
        mc.land()
        time.sleep(2)

        # Stop logging output after landing
        global logging_active
        logging_active = False
        print("🛑 Logging stopped after timeout landing.")

        return edges, None

"-----------------------------------------------------------------------------------------------------------------------"

# --- Find Center and Land ---
def center_and_land(mc, pad_center):
    global logging_active

    if not pad_center:
        print("⚠️ No pad center available for landing.")
        return
    x_center, y_center = pad_center
    dx = x_center - position_estimate[0]
    dy = y_center - position_estimate[1]

    print(f"📍 Moving to pad center: ΔX={dx:.2f}, ΔY={dy:.2f}")
    mc.move_distance(dx, dy, 0)
    print("🛬 Landing...")
    mc.land()
    time.sleep(2)
    print("✅ Landed successfully.")

    # Stop logging printouts
    logging_active = False
    print("🛑 Logging stopped after landing.")

"-----------------------------------------------------------------------------------------------------------------------"

if __name__ == "__main__":

    # === Initialize the low-level drivers ===
    cflib.crtp.init_drivers()
    cf = Crazyflie(rw_cache="./cache")

    print("🔌 Connecting to Crazyflie...")
    with SyncCrazyflie(URI, cf=cf) as scf:
        # === Deck check ===
        scf.cf.param.add_update_callback(group='deck', name='bcFlow2', cb=param_deck_flow)
        if not deck_attached_event.wait(timeout=5):
            print("❌ No flow deck detected — aborting mission.")
            sys.exit(1)

        print("✅ Flow deck detected.")
        scf.cf.platform.send_arming_request(True)
        time.sleep(1)

        # === Configure and start logging ===
        logconf = LogConfig(name="Position", period_in_ms=50)
        logconf.add_variable("stateEstimate.x", "float")
        logconf.add_variable("stateEstimate.y", "float")

        try:
            scf.cf.log.add_config(logconf)
            logconf.data_received_cb.add_callback(log_pos_callback)
            logconf.start()
            print("📡 Logging started (tracking position).")
        except KeyError as e:
            print(f"⚠️ Failed to add log config: {e}")
            sys.exit(1)
        except AttributeError as e:
            print(f"⚠️ Logging not supported by firmware: {e}")
            sys.exit(1)

        # === Begin mission ===
        with MotionCommander(scf, default_height=HOVER_HEIGHT) as mc:
            with Multiranger(scf) as mr:
                try:
                    print("\n🚁 Hovering to measure baseline pad height...")

                    # Wait until the downward sensor gives a valid reading
                    baseline = None
                    timeout = time.time() + 5  # 5-second timeout
                    while baseline is None and time.time() < timeout:
                        baseline = mr.down
                        time.sleep(0.1)

                    if baseline is None:
                        print(f"⚠️ Could not read baseline height (sensor returned None). Using default {HOVER_HEIGHT:.1f} m.")
                        baseline = HOVER_HEIGHT
                    else:
                        print(f"📏 Baseline height: {baseline:.2f} m")

                    time.sleep(HOVER_PAUSE)

                    # === Phase 1: Search sweep for edge ===
                    first_edge = search_sweep(mc, mr)

                    if first_edge:
                        print("✅ Edge found — starting pad edge detection...")
                        edges, pad_center = detect_pad_edges(mc, mr)
                        center_and_land(mc, pad_center)
                    else:
                        print("⚠️ No pad detected after sweep — landing.")
                        mc.land()
                        # ensure logging is stopped if we land here too
                        logging_active = False

                except KeyboardInterrupt:
                    print("\n⚠️ Emergency stop requested — landing immediately.")
                    mc.stop()
                    mc.land()
                    time.sleep(2)
                    print("🛑 Drone safely landed and disarmed.")
                except Exception as e:
                    print(f"\n❌ Mission aborted due to error: {e}")
                    mc.stop()
                    mc.land()
                    time.sleep(2)
                finally:
                    plt.ioff()
                    plt.show()

        # === Stop logging safely ===
        try:
            logconf.stop()
            print("📴 Logging stopped cleanly.")
        except Exception:
            print("⚠️ Could not stop log config properly (may already be stopped).")

        print("✅ Mission complete. Disconnected from Crazyflie.")
