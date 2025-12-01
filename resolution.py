import cv2

cam = cv2.VideoCapture(0)

resolutions = [
    (1920,1080),
    (2560,1440),
    (3840,2160),
    (1280,720),
    (640,480),
    (320,240)
]

print("\nChecking supported resolutions:\n")
for w,h in resolutions:
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    ret, frame = cam.read()
    actual_w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"Requested {w}x{h} → Got {int(actual_w)}x{int(actual_h)}")

cam.release()
