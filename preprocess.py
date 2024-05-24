import cv2
import math
import numpy as np

def process_image(image):
  shape = image.shape
  if(len(shape)>2):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]
  image = cv2.resize(image, (800, 1200))
  return image

def rotate_image(image, angle,centerpoints=None):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  print(image_center)
  if centerpoints != None:
    image_center=tuple(np.array(centerpoints)/1)
    print(image_center)

  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def getskew(sorted_peaks):
  base = abs(sorted_peaks[0][1] - sorted_peaks[1][1])
  height = abs(sorted_peaks[0][0] - sorted_peaks[1][0])
  theta = 90 - math.degrees(math.atan(height/base))
  # print(height,base)
  # print(theta)
  if sorted_peaks[0][0] < sorted_peaks[1][0]:
    print("positive Skew")
    return theta
  else:
    print("negative Skew")
    return -theta

def checkalignment(peaks_top,peaks_bottom):
  if(len(peaks_top) != 4 and len(peaks_bottom)!= 2):
    return False
  return True
