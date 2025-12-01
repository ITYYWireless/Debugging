import cv2
import subprocess
import time
import numpy as np
import mss


# Start scrcpy as a background process
scrcpy_process = subprocess.Popen([
    r"C:\Users\PC\Desktop\Robot Arm\scrcpy-win64-v3.3.3\scrcpy.exe",
    "--video-source=camera",
    "--camera-id=0",
    "--max-size=640",
    "--video-bit-rate=2M"
])

# Allow scrcpy to open
time.sleep(2)

CAPTURE_REGION = {
	"left": 0,
	"top": 100,
	"width": 640,
	"height": 480,
}

with mss.mss() as sct:
	while True:
		# Grab the region
		img = np.array(sct.grab(CAPTURE_REGION))

		# mss gives BGRA; drop alpha channel
		frame = img[:, :, :3]

		cv2.imshow("Phone Camera (Window Capture)", frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

cv2.destroyAllWindows()