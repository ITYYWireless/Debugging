import cv2
import numpy as np

CAM_INDEX = 1   # your camera index

# Tunable constants
MIN_PHONE_AREA = 20000    # minimum contour area to consider as phone
SMOOTH_FRAMES   = 5       # how many frames to keep last result when detection drops


def find_phone_screen(frame):
	"""
	Detect the phone screen outline by taking the largest contour and
	fitting a rotated rectangle (works for vertical or sideways phone).
	Returns pts (4 corner points) or None, and the edge image.
	"""

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)

	# Edge detection
	edges = cv2.Canny(blur, 70, 150)

	# Close gaps
	kernel = np.ones((5, 5), np.uint8)
	edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

	# Find contours
	contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return None, edges

	# Find largest contour over area threshold
	best = None
	best_area = 0
	for c in contours:
		area = cv2.contourArea(c)
		if area > MIN_PHONE_AREA and area > best_area:
			best = c
			best_area = area

	if best is None:
		return None, edges

	# Fit rotated rectangle to that contour
	rect = cv2.minAreaRect(best)        # (center, (w,h), angle)
	box = cv2.boxPoints(rect)           # 4 corner points (float)
	box = np.int32(box)                 # convert to int

	return box, edges


def main():
	cap = cv2.VideoCapture(CAM_INDEX)
	if not cap.isOpened():
		print("Could not open camera")
		return

	prev_pts = None
	missed = 0

	while True:
		ret, frame = cap.read()
		if not ret:
			continue

		pts, edges = find_phone_screen(frame)

		if pts is not None:
			prev_pts = pts
			missed = 0
		else:
			# If we temporarily lose detection, keep the last result for a few frames
			if prev_pts is not None and missed < SMOOTH_FRAMES:
				pts = prev_pts
				missed += 1

		if pts is not None:
			# Draw BLUE contour line
			cv2.polylines(frame, [pts], True, (255, 0, 0), 3)

			# Draw RED corners
			for (x, y) in pts:
				cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)

			cv2.putText(frame, "PHONE DETECTED", (10, 40),
			            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
		else:
			cv2.putText(frame, "NO PHONE", (10, 40),
			            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

		cv2.imshow("Camera", frame)
		cv2.imshow("Edges", edges)

		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
