import cv2
import numpy as np

def detect_anchors(image):
  """Detects anchors in an OMR image.

  Args:
    image: A numpy array representing the OMR image.

  Returns:
    A list of numpy arrays representing the detected anchors.
  """

  # Convert the image to grayscale.
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Blur the image to reduce noise.
  blur = cv2.GaussianBlur(gray, (5, 5), 0)

  # Apply edge detection.
  edges = cv2.Canny(blur, 100, 200)

  # Find contours in the image.
  contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  # Filter out contours that are too small or too large.
  filtered_contours = []
  for contour in contours:
    # print(contour)
    # area = cv2.contourArea(contour)
    # if 100 < area < 10000:
      filtered_contours.append(contour)

  # Return the filtered contours.
  return filtered_contours

# Load the image.
image = cv2.imread("1.jpg")

# Detect the anchors.
anchors = detect_anchors(image)

# Draw the anchors on the image.
for anchor in anchors:
  cv2.rectangle(image, (anchor[0][0], anchor[0][1]), (anchor[2][0], anchor[2][1]), (0, 255, 0), 2)

# Show the image.
cv2.imshow("Anchors", image)
cv2.waitKey(0)
cv2.destroyAllWindows()