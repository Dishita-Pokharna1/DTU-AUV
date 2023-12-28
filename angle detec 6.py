import cv2
#cv2 for image processing
#numpy for numerical operations
import numpy as np


image = cv2.imread('1234.jpeg')

# COLOR
while True:
    img = cv2.imread("1234.jpeg")
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 51, 0])
    upper = np.array([26, 255, 255])

    mask = cv2.inRange(imgHSV, lower, upper)
    final_image = mask.copy()
    imgResult = cv2.bitwise_and(img, img, mask=mask)
    break
    #find binary mask using cv2 find contours

contours, _ = cv2.findContours(final_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#min area threshold set to filter out small contours
min_area = 100
main_path_contour = None
#approx contour using polydp to reduce number of vertices

for contour in contours:
    if cv2.contourArea(contour) > min_area:
        main_path_contour = contour
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approximated_contour = cv2.approxPolyDP(contour, epsilon, True)


        cv2.drawContours(image, [approximated_contour], -1, (0, 0, 255), 4)

        if len(approximated_contour) > 7:
            x3, y3 = approximated_contour[2][0]
            x7, y7 = approximated_contour[6][0]

            cv2.circle(image, (x3, y3), 8, (0, 255, 0), -1)
            cv2.circle(image, (x7, y7), 8, (0, 255, 0), -1)


            cv2.line(image, (x3, y3), (x7, y7), (255, 0, 0), 2)

            slope = abs(y7 - y3) / abs(x7 - x3)

            angle_radians = np.abs(np.arctan(slope))
            angle_degrees = np.degrees(angle_radians)
            cv2.putText(image, f' Arc tangent: {angle_degrees:.2f} degrees', (x3, y3 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 0, 0), 3)

cv2.imshow('Original and Approximated Paths with A, B, Line, and Angle', image)
cv2.imshow("Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()