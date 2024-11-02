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


def get_roi_match(image_roi,template,bottom_search_point=None,top=True):
  result = match_template(image_roi, template,pad_input=True) #added the pad_input bool
  peaks = peak_local_max(result,min_distance=10,threshold_rel=0.8) # find our peaks
  if(top == False):
    peaks[:,0] = peaks[:,0]+bottom_search_point
  return peaks

def get_image_sections(image,coords):
  region = image[coords["region"][0]:coords["region"][1],coords["region"][2]:coords["region"][3]]
  a = image[coords["a"][0]:coords["a"][1],coords["a"][2]:coords["a"][3]]
  b = image[coords["b"][0]:coords["b"][1],coords["b"][2]:coords["b"][3]]
  c = image[coords["c"][0]:coords["c"][1],coords["c"][2]:coords["c"][3]]
  d = image[coords["d"][0]:coords["d"][1],coords["d"][2]:coords["d"][3]]
  return region,[a,b,c,d]

def preprocess_section_image_before_crop(image,threshold):
  _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
  kernel = np.ones((3, 3), np.uint8)
  eroded_adjusted = cv2.erode(binary, kernel, iterations=2)
  result = cv2.bitwise_not(eroded_adjusted)
  return result


def get_image_sectionsv2(image,coords):
  optionimages = []
  for key in coords.keys():
    if key == "region":
      region = image[coords[key][0]:coords[key][1],coords[key][2]:coords[key][3]]
      region = preprocess_section_image_before_crop(region,175)
    else:
      tmpoptionimage = image[coords[key][0]:coords[key][1],coords[key][2]:coords[key][3]]
      tmpoptionimage = preprocess_section_image_before_crop(tmpoptionimage,175)
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
    print(f"For each Option - mean {np.mean(255-i)} | length {len(i)} | division {np.mean(255-i)/len(i)}")
    result.append(np.mean(255-i))
  result_arr = np.array(result)
  # print(result_arr)
  for result_arr_element in result_arr:
    if (result_arr_element > ROLL_THRESH):
        multiselect += 1
    if (result_arr_element < ROLL_THRESH):
        noselect += 1
  if show_region:
    plt.imshow(region)
    plt.show()
  option_index = np.argmax(result_arr)
  if multiselect > 1:
    option_index = 10
  if noselect != (ROLLLENGTH-1) and multiselect <=1:
    option_index = 11
  return OPTIONS_MAP_ROLL[option_index]

# def count_large_values(arr, threshold=2):
#   mean = np.mean(arr)
#   std_dev = np.std(arr)
#   # Calculate Z-scores for each element
#   z_scores = (arr - mean) / std_dev
#   # Count the number of elements with Z-score above the threshold
#   large_values_count = np.sum(z_scores > threshold)
#   return z_scores,large_values_count

def check_single_selectionv2(arr, threshold):
  # Count the number of elements greater than the threshold
  checked_option = np.where(arr > threshold)[0]
  return checked_option

# def no_selected_outlier(data):
#     data = np.array(data)
#     Q1 = np.percentile(data, 25)
#     Q3 = np.percentile(data, 75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.75 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     outliers = data[(data < lower_bound) | (data > upper_bound)]
#     return len(outliers)

# def check_single_selection(data,multiplicative_factor):
#   mean = np.mean(data)
#   std_dev = np.std(data)
#   threshold = mean + multiplicative_factor * std_dev
#   outlier_indices = np.where(data > threshold)[0]
#   return outlier_indices

# def find_IQR(data):
#   Q1 = np.percentile(data, 25)
#   Q3 = np.percentile(data, 75)
#   IQR = Q3 - Q1
#   return IQR

# def check_no_selected(data):
#   original_IQR = find_IQR(data)
#   max_index = np.argmax(data)  
#   data_without_max = np.delete(data, max_index)
#   new_IQR = find_IQR(data_without_max)
#   print(f" Old IQR - {original_IQR} and new IQR {new_IQR}")
#   if (original_IQR > (2 * new_IQR)):
#     return False
#   else:
#     return True
  

# def get_section_data(region,options,OPTIONS_MAP_DATA,threshold,show_region=False):
#   result = []
#   for i in options:
#     print(f"For each Option - mean {np.mean(255-i)} | option {i.shape} | division {np.mean(255-i)/len(i)}")
#     result.append(np.mean(255-i))
#   result_arr = np.array(result)
#   print(result_arr)
#   selected_indices = check_single_selectionv2(result_arr,1.5)
#   if len(selected_indices) == 1:
#     # Only one option selected
#     print(f"Single option with index {selected_indices[0]}")
#     option_index = selected_indices[0]
#   else:
#     threshold = np.percentile(result_arr, 90)
#     outlier_indices = np.where(result_arr > threshold)[0]
#     if len(outlier_indices) == 0:
#       option_index = len(OPTIONS_MAP_DATA)-2
#       print(f"Multi options with indexs {option_index} - threshold {threshold}")
#     else:
#       option_index = len(OPTIONS_MAP_DATA)-1
#       print(f"No options with index {option_index} - threshold {threshold}")
#   if show_region:
#     plt.imshow(region)
#     plt.show()
#   return result_arr,OPTIONS_MAP_DATA[option_index]
def get_section_datav_using_median3(region,options,OPTIONS_MAP_DATA,threshold,show_region=False):
  result = []
  for i in options:
    print(f"For each Option - median {np.median(255-i)} | option {i.shape} | division {np.median(255-i)/len(i)}")
    result.append(np.median(255-i))
  result_arr = np.array(result)

  print(result_arr)
  selected_indices = check_single_selectionv2(result_arr,threshold)

  if len(selected_indices) == 1:
    # Only one option selected
    print(f"Single option with index {selected_indices[0]}")
    option_index = selected_indices[0]
  elif len(selected_indices) == 0:
    option_index = len(OPTIONS_MAP_DATA)-1
    print(f"No options with index {option_index} - threshold {threshold}")
  else:
    option_index = len(OPTIONS_MAP_DATA)- 2
    print(f"Multiple options with index {option_index} - threshold {threshold}")

  if show_region:
    plt.imshow(region)
    plt.show()
  return result_arr,OPTIONS_MAP_DATA[option_index]


def get_section_datav2(region,options,OPTIONS_MAP_DATA,threshold,show_region=False):
  result = []
  for i in options:
    print(f"For each Option - mean {np.mean(255-i)} | option {i.shape} | division {np.mean(255-i)/len(i)}")
    result.append(np.mean(255-i))
  result_arr = np.array(result)

  print(result_arr)
  selected_indices = check_single_selectionv2(result_arr,threshold)

  if len(selected_indices) == 1:
    # Only one option selected
    print(f"Single option with index {selected_indices[0]}")
    option_index = selected_indices[0]
  elif len(selected_indices) == 0:
    option_index = len(OPTIONS_MAP_DATA)-1
    print(f"No options with index {option_index} - threshold {threshold}")
  else:
    option_index = len(OPTIONS_MAP_DATA)- 2
    print(f"Multiple options with index {option_index} - threshold {threshold}")

  if show_region:
    plt.imshow(region)
    plt.show()
  return result_arr,OPTIONS_MAP_DATA[option_index]

def get_thresholdv2(rawdata,percentile):
  non_zero_elements = sorted(set(filter(lambda x: x != 0, rawdata)))
  data = np.array(non_zero_elements)
  Q1 = np.percentile(data, 25)
  Q3 = np.percentile(data, 75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  actual_data = data[(data > lower_bound) | (data < upper_bound)]
  threshold = np.percentile(actual_data, percentile)
  return threshold
  

def get_threshold(data,percentile):
  ## Remove outliers and then get the threshold
  data = np.array(data)
  Q1 = np.percentile(data, 25)
  Q3 = np.percentile(data, 75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  actual_data = data[(data > lower_bound) | (data < upper_bound)]
  threshold = np.percentile(actual_data, percentile)
  return threshold

def get_section_means(options):
  result = []
  for i in options:
    result.append(np.mean(255-i))
  result_arr = np.array(result)
  return result_arr

def get_section_median(options):
  result = []
  for i in options:
    result.append(np.median(255-i))
  result_arr = np.array(result)
  return result_arr

def showsectionandgetregion(image,data,calculatedanchorx,calculatedanchory,sortedanchor,anchornumber,sectionnumber):
    print(calculatedanchorx,calculatedanchory)
    q12md = getmetadataforblock(sortedanchor[anchornumber],data[sectionnumber])
    q = getactualcoordinates(calculatedanchorx,calculatedanchory,q12md)
    region,options = get_image_sections(image,q)
    selected_result = get_result(region,options,True)
    print(q12md["name"],selected_result,q["region"])