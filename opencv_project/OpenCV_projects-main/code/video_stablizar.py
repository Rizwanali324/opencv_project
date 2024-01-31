import cv2
import numpy as np

def on_trackbar(val):
    global feature_params, lk_params
    feature_params['maxCorners'] = cv2.getTrackbarPos('Max Corners', 'Original | Stabilized')
    lk_params['winSize'] = (cv2.getTrackbarPos('Win Size', 'Original | Stabilized'),) * 2

# Load a shaky video
cap = cv2.VideoCapture('OpenCV_projects-main/data/shaking.mp4')

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read the first frame from the video
ret, prev_frame = cap.read()
if not ret:
    print("Error: Cannot read video file.")
    exit()

# Convert frame to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Shi-Tomasi corner detection parameters
feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

# Set up the Lucas-Kanade optical flow parameters
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create window and trackbars
cv2.namedWindow('Original | Stabilized')
cv2.createTrackbar('Max Corners', 'Original | Stabilized', feature_params['maxCorners'], 500, on_trackbar)
cv2.createTrackbar('Win Size', 'Original | Stabilized', lk_params['winSize'][0], 50, on_trackbar)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect features in the first frame
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    # Calculate optical flow
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_pts, None, **lk_params)

    # Select good points
    good_prev_pts = prev_pts[status == 1]
    good_next_pts = next_pts[status == 1]

    # Ensure there are enough points to estimate a transformation
    if len(good_prev_pts) >= 3:
        # Estimate transformation matrix
        matrix, _ = cv2.estimateAffinePartial2D(good_prev_pts, good_next_pts)

        # Apply the transformation to stabilize the frame
        stabilized_frame = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))
    else:
        stabilized_frame = frame.copy()

    # Concatenate original and stabilized frame
    concatenated_frame = np.hstack((frame, stabilized_frame))

    # Display the concatenated frame
    cv2.imshow("Original | Stabilized", concatenated_frame)

    # Update previous frame and previous points
    prev_gray = frame_gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
