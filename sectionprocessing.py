import matplotlib.pyplot as plt
import cv2
from skimage import io
from skimage.feature import match_template
from skimage.feature import peak_local_max # new import!
import numpy as np
from config import *

def getmetadataforblock(anchor,block):
  metadata = {}
  metadata["name"] = block["name"]
  metadata["anchorx2startx"] = block["start"]["x"] - anchor["start"]["x"]
  metadata["anchory2starty"] = block["start"]["y"] - anchor["start"]["y"]
  metadata["anchorx2endx"] = block["end"]["x"] - anchor["start"]["x"]
  metadata["anchory2endy"] = block["end"]["y"] - anchor["start"]["y"]
  metadata["anchor2options"] = []
  blockchildren = block["children"]
  for options in blockchildren:
    optionmetadata = {}
    optionmetadata["optionname"] = options["name"]
    optionmetadata["anchorx2optionstartx"] = options["start"]["x"] - anchor["start"]["x"]
    optionmetadata["anchory2optionstarty"] = options["start"]["y"] - anchor["start"]["y"]
    optionmetadata["anchorx2optionendx"] = options["end"]["x"] - anchor["start"]["x"]
    optionmetadata["anchory2optionendy"] = options["end"]["y"] - anchor["start"]["y"]
    metadata["anchor2options"].append(optionmetadata)
  return metadata

def getactualcoordinates(calculatedanchorx,calculatedanchory,q12md):
  x1 = int(calculatedanchorx + q12md["anchorx2startx"])
  x2 = int(calculatedanchorx + q12md["anchorx2endx"])
  y1 = int(calculatedanchory + q12md["anchory2starty"])
  y2 = int(calculatedanchory + q12md["anchory2endy"])
  q = {
      "region": [y1,y2,x1,x2], # for resized image of 800,1200
  }
  for option in q12md["anchor2options"]:
    x1 = int(calculatedanchorx + option["anchorx2optionstartx"])
    x2 = int(calculatedanchorx + option["anchorx2optionendx"])
    y1 = int(calculatedanchory + option["anchory2optionstarty"])
    y2 = int(calculatedanchory + option["anchory2optionendy"])
    q[option["optionname"]] = [y1,y2,x1,x2]
  return q


def get_roi_match(image_roi,template,top=True):
  result = match_template(image_roi, template,pad_input=True) #added the pad_input bool
  peaks = peak_local_max(result,min_distance=10,threshold_rel=0.8) # find our peaks
  if(top == False):
    peaks[:,0] = peaks[:,0]+1100
  return peaks

def get_image_sections(image,coords):
  region = image[coords["region"][0]:coords["region"][1],coords["region"][2]:coords["region"][3]]
  a = image[coords["a"][0]:coords["a"][1],coords["a"][2]:coords["a"][3]]
  b = image[coords["b"][0]:coords["b"][1],coords["b"][2]:coords["b"][3]]
  c = image[coords["c"][0]:coords["c"][1],coords["c"][2]:coords["c"][3]]
  d = image[coords["d"][0]:coords["d"][1],coords["d"][2]:coords["d"][3]]
  return region,[a,b,c,d]

def get_image_sectionsv2(image,coords):
  optionimages = []
  for key in coords.keys():
    if key == "region":
      region = image[coords[key][0]:coords[key][1],coords[key][2]:coords[key][3]]
    else:
      tmpoptionimage = image[coords[key][0]:coords[key][1],coords[key][2]:coords[key][3]]
      optionimages.append(tmpoptionimage)
  return region,optionimages

def get_result(region,options,show_region=False):
  result = []
  multiselect = 0
  noselect = 0
  for i in options:
    result.append(np.mean(255-i))
  result_arr = np.array(result)
  for result_arr_element in result_arr:
    if (result_arr_element > RESULT_THRESH):
      multiselect += 1
    if (result_arr_element < RESULT_THRESH):
      noselect += 1
  # print(result_arr)
  if show_region:
    plt.imshow(region)
    plt.show()
  option_index = np.argmax(result_arr)
  if multiselect > 1:
    option_index = 4
  if noselect != (OPTIONS-1) and multiselect <=1:
    option_index = 5
  return OPTIONS_MAP[option_index]

def get_roll(region,options,show_region=False):
  result = []
  countselected = 0
  multiselect = 0
  noselect = 0
  for i in options:
    result.append(np.mean(255-i))
  result_arr = np.array(result)
  # print(result_arr)
  for result_arr_element in result_arr:
    if (result_arr_element > ROLL_THRESH):
        multiselect += 1
    if (result_arr_element < RESULT_THRESH):
        noselect += 1
  if show_region:
    plt.imshow(region)
    plt.show()
  option_index = np.argmax(result_arr)
  if countselected > 1:
    multiselect = 10
  if noselect != (ROLLLENGTH-1) and multiselect <=1:
    option_index = 11
  return OPTIONS_MAP_ROLL[option_index]


def showsectionandgetregion(image,data,calculatedanchorx,calculatedanchory,sortedanchor,anchornumber,sectionnumber):
    print(calculatedanchorx,calculatedanchory)
    q12md = getmetadataforblock(sortedanchor[anchornumber],data[sectionnumber])
    q = getactualcoordinates(calculatedanchorx,calculatedanchory,q12md)
    region,options = get_image_sections(image,q)
    selected_result = get_result(region,options,True)
    print(q12md["name"],selected_result,q["region"])