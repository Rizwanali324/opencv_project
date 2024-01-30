"""

# Code

import cv2
import numpy as np


def nothing(x):
    # Callback function for trackbar, does nothing but required by OpenCV
    pass

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)
# Create a window
cv2.namedWindow('Real-time Edge Detection')

# Create trackbars for adjusting Canny thresholds
cv2.createTrackbar('Min Threshold', 'Real-time Edge Detection', 100, 500, nothing)
cv2.createTrackbar('Max Threshold', 'Real-time Edge Detection', 150, 500, nothing)

# Get frame width, height, and FPS from the capture object
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object to save the video in MP4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change codec to 'mp4v'
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width * 2, frame_height))


while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Corrected color conversion
# Apply Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

     # Get the current positions of the trackbars
    min_threshold = cv2.getTrackbarPos('Min Threshold', 'Real-time Edge Detection')
    max_threshold = cv2.getTrackbarPos('Max Threshold', 'Real-time Edge Detection')

    # Apply Canny edge detection using trackbar values
    edges = cv2.Canny(blurred, min_threshold, max_threshold)

    # Convert 'edges' to a 3-channel image so it can be stacked with 'frame'
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Stack both frames for side-by-side display
    combined = np.hstack((frame, edges_colored))

    # Display the result
    cv2.imshow('Real-time Edge Detection', combined)
    # Write the frame to the output video file
    out.write(combined)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
cv2.destroyAllWindows()"""




import cv2
import numpy as np
def nothing(x):
    # Callback function for trackbar, does nothing but required by OpenCV
    pass

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)
# Create a window
cv2.namedWindow('Real-time Edge Detection')

# Create trackbars for adjusting Canny thresholds
cv2.createTrackbar('Min Threshold', 'Real-time Edge Detection', 100, 500, nothing)
cv2.createTrackbar('Max Threshold', 'Real-time Edge Detection', 150, 500, nothing)


# Get frame width, height, and FPS from the capture object
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object to save the video in MP4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change codec to 'mp4v'
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width * 2, frame_height))

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Corrected color conversion
# Apply Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

     # Get the current positions of the trackbars
    min_threshold = cv2.getTrackbarPos('Min Threshold', 'Real-time Edge Detection')
    max_threshold = cv2.getTrackbarPos('Max Threshold', 'Real-time Edge Detection')

    # Apply Canny edge detection using trackbar values
    edges = cv2.Canny(blurred, min_threshold, max_threshold)

    # Convert 'edges' to a 3-channel image so it can be stacked with 'frame'
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Stack both frames for side-by-side display
    combined = np.hstack((frame, edges_colored))

    # Display the result
    cv2.imshow('Real-time Edge Detection', combined)

    # Write the frame to the output video file
    out.write(combined)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
