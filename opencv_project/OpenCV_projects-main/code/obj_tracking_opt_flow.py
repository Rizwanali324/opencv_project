import cv2
import numpy as np

# Start capturing video
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Cannot read video frame")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec
out = cv2.VideoWriter('tracking_output.mp4', fourcc, 20.0, (frame.shape[1], frame.shape[0]))

# Select the ROI for tracking
roi = cv2.selectROI(frame, False)
x, y, w, h = roi
track_window = (x, y, w, h)

# Set up the ROI for tracking
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # Apply CAMShift to get the new location
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)

    # Draw it on the image
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    img2 = cv2.polylines(frame, [pts], True, 255, 2)

    cv2.imshow('CAMShift Tracking', img2)
    out.write(img2)  # Write the frame to the video file

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release() # Release the VideoWriter object
cv2.destroyAllWindows()
