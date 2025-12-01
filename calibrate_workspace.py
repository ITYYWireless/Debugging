import cv2
import time
from maxarm_pc import MaxArmClient

# === CONFIG ===
PORT = "COM2165"
CAM_INDEX = 1  

# Starting position
START_X = 0
START_Y = -150
START_Z = 180

# Jog step size (mm)
STEP_XY = 5
STEP_Z  = 5

MOVE_TIME_MS = 200  # small moves


def main():
	arm = MaxArmClient(PORT)
	print("Homing...")
	arm.home(2000)

	# Current position
	x = START_X
	y = START_Y
	z = START_Z

	# Move to start
	print(f"Moving to start: X={x}, Y={y}, Z={z}")
	arm.move_xyz(x, y, z, time_ms=200)
	time.sleep(0.6)

	# Track min/max visited
	min_x = max_x = x
	min_y = max_y = y
	min_z = max_z = z

	cap = cv2.VideoCapture(CAM_INDEX)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

	print("\nControls:")
	print("  W/S : +Y / -Y")
	print("  A/D : -X / +X")
	print("  R/F : +Z / -Z")
	print("  C   : print min/max")
	print("  Q   : quit\n")

	while True:
		ret, frame = cap.read()
		if not ret:
			continue

		h, w, _ = frame.shape

		# Crosshair
		cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
		cv2.line(frame, (0, h // 2), (w, h // 2), (255, 255, 255), 1)

		# Show XYZ
		cv2.putText(frame, f"X={x:.1f} Y={y:.1f} Z={z:.1f}",
		            (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

		# Min/max
		cv2.putText(frame, f"Xmin={min_x:.1f} Xmax={max_x:.1f}",
		            (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 1)
		cv2.putText(frame, f"Ymin={min_y:.1f} Ymax={max_y:.1f}",
		            (10,75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 1)
		cv2.putText(frame, f"Zmin={min_z:.1f} Zmax={max_z:.1f}",
		            (10,95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 1)

		cv2.putText(frame, "W/S=Y  A/D=X  R/F=Z  C=minmax  Q=quit",
		            (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

		cv2.imshow("Workspace Calibration", frame)

		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break

		moved = False

		# Y axis
		if key == ord('w'):
			y += STEP_XY; moved = True
		elif key == ord('s'):
			y -= STEP_XY; moved = True

		# X axis
		if key == ord('d'):
			x += STEP_XY; moved = True
		elif key == ord('a'):
			x -= STEP_XY; moved = True

		# Z axis
		if key == ord('r'):
			z += STEP_Z; moved = True
		elif key == ord('f'):
			z -= STEP_Z; moved = True

		# Print min/max snapshot
		if key == ord('c'):
			print("\n=== CURRENT MIN/MAX VISITED ===")
			print(f"X_MIN = {min_x:.1f}")
			print(f"X_MAX = {max_x:.1f}")
			print(f"Y_MIN = {min_y:.1f}")
			print(f"Y_MAX = {max_y:.1f}")
			print(f"Z_MIN = {min_z:.1f}")
			print(f"Z_MAX = {max_z:.1f}")
			print("================================\n")

		# Move the arm if requested
		if moved:
			arm.move_xyz(x, y, z, time_ms=MOVE_TIME_MS)
			time.sleep(MOVE_TIME_MS / 1000.0)

			min_x = min(min_x, x)
			max_x = max(max_x, x)
			min_y = min(min_y, y)
			max_y = max(max_y, y)
			min_z = min(min_z, z)
			max_z = max(max_z, z)

	cap.release()
	cv2.destroyAllWindows()
	arm.close()


if __name__ == "__main__":
	main()
