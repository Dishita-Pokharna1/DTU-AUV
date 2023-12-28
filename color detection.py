import cv2
import numpy as np


# Function to detect a specific color range in an image
def detect_color(image, lower_bound, upper_bound):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define a mask using the specified color range
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Bitwise-AND mask and original image
    result = cv2.bitwise_and(image, image, mask=mask)

    return result


# Read an image
image = cv2.imread('image.jpg')

# Check if the image is loaded successfully
if image is None:
    print("Error: Could not read the image.")
    exit()

# Define the color range for blue in HSV
lower_bound = np.array([100, 50, 50])  # Example: lower bound for blue
upper_bound = np.array([130, 255, 255])  # Example: upper bound for blue

# Call the detect_color function
result_image = detect_color(image, lower_bound, upper_bound)

# Display the original and result images
cv2.imshow('Original Image', image)
cv2.imshow('Color Detection Result', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()