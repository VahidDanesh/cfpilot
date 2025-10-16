"""
Simple test to verify position data is being received correctly
"""
import time
import cflib.crtp
from cflib.utils import uri_helper
import sys
from pathlib import Path


try:
    import cfpilot
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from cfpilot.controller import CrazyflieController


class PositionTest:
    def __init__(self):
        self.controller = CrazyflieController()
        self.uri = self.controller.config['connection']['uri']
        self.position_count = 0
        
    def state_print(self):
        # Read current state directly

        latest = self.controller.get_latest_data()
        current_x = latest.get('stateEstimate.x', -1.0)
        current_y = latest.get('stateEstimate.y', -1.0)
        current_z = latest.get('stateEstimate.z', -1.0)
        yaw = latest.get('stabilizer.yaw', -1.0)
        print(f"Current Position: x={current_x:.3f}, y={current_y:.3f}, z={current_z:.3f}, yaw={yaw:.1f}°")

    def run_test(self, duration=10):
        """Test position readout for specified duration"""
        print("="*60)
        print("Position Readout Test")
        print(f"Duration: {duration} seconds")
        print("="*60)
        
        print(f"\nConnecting to {self.uri}...")
        self.controller.connect(self.uri)
        if not self.controller.wait_for_connection(timeout=10.0):
            print("❌ Connection failed!")
            return
        
        print("✅ Connected!")
                
        print(f"\n📊 Receiving position data for {duration} seconds...")
        print("(Printing every 1 second)\n")
        current_time = time.time()
        end_time = current_time + duration
        while time.time() < end_time:
            self.state_print()
            self.position_count += 1
            time.sleep(0.1)
        
        print(f"\n✅ Test complete!")
        print(f"Total position updates received: {self.position_count}")
        print(f"Update rate: {self.position_count / duration:.1f} Hz")
        
        self.controller.disconnect()


def main():
    cflib.crtp.init_drivers()
    
    test = PositionTest()
    
    try:
        test.run_test(duration=10)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted")
        test.controller.disconnect()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        test.controller.disconnect()


if __name__ == "__main__":
    main()
