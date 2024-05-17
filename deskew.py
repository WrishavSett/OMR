import cv2
from deskew import determine_skew

image = cv2.imread('1.jpg')
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
angle = determine_skew(grayscale)
print(angle)
# rotated = rotate(image, angle, (0, 0, 0))
# cv2.imwrite('output.png', rotated)