# test_arm.py
from maxarm_pc import MaxArmClient

PORT = "COM2165"  # <<< change to your real port

arm = MaxArmClient(PORT)

print("Homing...")
print(arm.home(2000))

print("Moving...")
print(arm.move_xyz(50, -150, 160, 1000))
