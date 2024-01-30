"""import cv2
import numpy as np

def nothing(x):
    # Callback function for trackbar (does nothing but necessary for createTrackbar)
    pass

def region_of_interest(img):
    height, width = img.shape
    polygon = np.array([
        [(0, height), (width, height), (width, height//2), (0, height//2)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Load the image or video frame
image = cv2.imread('images.png')  # Replace with your image path
cv2.namedWindow('Lane Line Detection')

# Create trackbars for Gaussian blur and Canny thresholds
cv2.createTrackbar('Gaussian Kernel Size', 'Lane Line Detection', 5, 30, nothing)
cv2.createTrackbar('Min Threshold', 'Lane Line Detection', 50, 200, nothing)
cv2.createTrackbar('Max Threshold', 'Lane Line Detection', 150, 300, nothing)

while True:
    # Read trackbar positions
    kernel_size = cv2.getTrackbarPos('Gaussian Kernel Size', 'Lane Line Detection')
    min_threshold = cv2.getTrackbarPos('Min Threshold', 'Lane Line Detection')
    max_threshold = cv2.getTrackbarPos('Max Threshold', 'Lane Line Detection')

    # Ensure kernel size is odd
    if kernel_size % 2 == 0: kernel_size += 1

    # Process the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    edges = cv2.Canny(blur, min_threshold, max_threshold)
    roi = region_of_interest(edges)

    # Apply Hough Line Transform
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, 100, minLineLength=40, maxLineGap=5)

    # Draw lines on a copy of the original image
    line_image = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    # Stack original and line-detected images side by side
    combined_image = np.hstack((image, line_image))

    # Show the result
    cv2.imshow('Lane Line Detection', combined_image)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
"""





import cv2
import numpy as np

def nothing(x):
    # Callback function for trackbar
    pass

def region_of_interest(img):
    height, width = img.shape[:2]
    # Define a polygon for the ROI, adjust these values based on your needs
    polygon = np.array([
        [(50, height), (width - 50, height), (width - 50, height // 2), (50, height // 2)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [polygon], 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Start capturing video from the webcam
cap = cv2.VideoCapture("test2.mp4 ")

# Create a window
cv2.namedWindow('Lane Line Detection')

# Create trackbars for color thresholding and Canny thresholds
cv2.createTrackbar('Lower Threshold', 'Lane Line Detection', 0, 255, nothing)
cv2.createTrackbar('Upper Threshold', 'Lane Line Detection', 255, 255, nothing)
cv2.createTrackbar('Min Canny Threshold', 'Lane Line Detection', 50, 200, nothing)
cv2.createTrackbar('Max Canny Threshold', 'Lane Line Detection', 150, 300, nothing)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to (500, 500)
    frame = cv2.resize(frame, (500, 500))

    # Read trackbar positions
    lower_thresh = cv2.getTrackbarPos('Lower Threshold', 'Lane Line Detection')
    upper_thresh = cv2.getTrackbarPos('Upper Threshold', 'Lane Line Detection')
    min_canny_threshold = cv2.getTrackbarPos('Min Canny Threshold', 'Lane Line Detection')
    max_canny_threshold = cv2.getTrackbarPos('Max Canny Threshold', 'Lane Line Detection')

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply color thresholding
    _, thresholded = cv2.threshold(gray, lower_thresh, upper_thresh, cv2.THRESH_BINARY)

    # Apply Canny edge detection
    edges = cv2.Canny(thresholded, min_canny_threshold, max_canny_threshold)

    # Apply region of interest masking
    masked_edges = region_of_interest(edges)

    # Apply Hough Line Transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 100, minLineLength=40, maxLineGap=5)

    # Draw lines on a copy of the original frame
    line_image = frame.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    # Stack original and line-detected frames side by side
    combined_image = np.hstack((frame, line_image))

    # Show the result
    cv2.imshow('Lane Line Detection', combined_image)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
cv2.destroyAllWindows()
