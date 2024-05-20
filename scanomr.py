import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io
from skimage.feature import match_template
from skimage.feature import peak_local_max # new import!
import os
import pandas as pd


OPTIONS_MAP = {0:"a",1:"b",2:"c",3:"d",4:"Not Selected"}
OPTIONS_MAP_ROLL = {0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",10:"-NS/MS-"}


peaks_top = 0
peaks_bottom = 0
calculatedanchorx = 0
calculatedanchory = 0
epsilon = 25
data = None

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

def process_image(image):
  shape = image.shape
  if(len(shape)>2):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]
  image = cv2.resize(image, (800, 1200))
  return image


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
  countselected = 0
  for i in options:
    result.append(np.mean(255-i))
  result_arr = np.array(result)
  for result_arr_element in result_arr:
    if (result_arr_element > 80):
      countselected += 1
  # print(result_arr)
  if show_region:
    plt.imshow(region)
  option_index = np.argmax(result_arr)
  if countselected > 1:
    option_index = 4
  return OPTIONS_MAP[option_index]

def get_roll(region,options,show_region=False):
  result = []
  countselected = 0
  for i in options:
    result.append(np.mean(255-i))
  result_arr = np.array(result)
  # print(result_arr)
  for result_arr_element in result_arr:
    if (result_arr_element > 100):
      countselected += 1
  if show_region:
    plt.imshow(region)
    plt.show()
  option_index = np.argmax(result_arr)
  print(option_index)
  if countselected > 1:
    option_index = 10
  return OPTIONS_MAP_ROLL[option_index]

def createmetadatfile(anchornumber,data):
  metadatalist = []
  for data_element in data:
    if data_element["name"] != "anchor":
      block_metadata = getmetadataforblock(data[anchornumber],data_element)
      metadatalist.append(block_metadata)
  with open("metadata.json", "w") as outfile:
    json.dump(metadatalist, outfile)

    

def showsectionandgetregion(image,data,calculatedanchorx,calculatedanchory,sortedanchor,anchornumber,sectionnumber):
    print(calculatedanchorx,calculatedanchory)
    q12md = getmetadataforblock(sortedanchor[anchornumber],data[sectionnumber])
    q = getactualcoordinates(calculatedanchorx,calculatedanchory,q12md)
    region,options = get_image_sections(image,q)
    selected_result = get_result(region,options,True)
    print(q12md["name"],selected_result,q["region"])

def readjson(filename):
    global data
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def getsortedanchor(data):
  allanchors = []
  for elements in data:
    if elements["name"] == "anchor":
      allanchors.append(elements)
  sortedanchor = sorted(allanchors, key=lambda d: d["start"]["x"])
  return sortedanchor

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


def processoneimage(payloaddata,template,image,anchornumber):
    sortedanchor = getsortedanchor(payloaddata)
    template = process_image(template)
    template = template[int(sortedanchor[anchornumber]["start"]["y"]):int(sortedanchor[anchornumber]["end"]["y"]),int(sortedanchor[anchornumber]["start"]["x"]):int(sortedanchor[anchornumber]["end"]["x"])]
    # plt.imshow(template)
    # plt.show()
    # image = io.imread("1.jpg")
    image = process_image(image)

    image_roi_top = image[0:180,:]
    image_roi_bottom = image[1100:,:]

    peaks_bottom = get_roi_match(image_roi_bottom,template,top=False)
    peaks_top = get_roi_match(image_roi_top,template,top=True)
    allpeaks = np.concatenate((peaks_top, peaks_bottom), axis=0)
    sorted_peaks_tops = allpeaks[allpeaks[:, 1].argsort()]
    calculatedanchorx = sorted_peaks_tops[anchornumber][1] - 7
    calculatedanchory = sorted_peaks_tops[anchornumber][0] - 7

    # plt.imshow(image)
    # plt.plot(peaks_bottom[:,1], peaks_bottom[:,0], 'o', markeredgecolor='r', markerfacecolor='none', markersize=2)
    # plt.plot(peaks_top[:,1], peaks_top[:,0], 'o', markeredgecolor='r', markerfacecolor='none', markersize=2)
    # print(calculatedanchorx,calculatedanchory)
    # showsectionandgetregion(image,data,calculatedanchorx,calculatedanchory,sortedanchor,anchornumber,35)

    datadict = {}
    rollnumber = ""
    for i in range(6,16):
      q12md = getmetadataforblock(sortedanchor[anchornumber],data[i])
      q = getactualcoordinates(calculatedanchorx,calculatedanchory,q12md)
      region,options = get_image_sectionsv2(image,q)
      selected_result = get_roll(region,options)
      rollnumber += selected_result
    datadict["imagename"] = rollnumber
    for i in range(16,len(data)):
        q12md = getmetadataforblock(sortedanchor[anchornumber],data[i])
        q = getactualcoordinates(calculatedanchorx,calculatedanchory,q12md)
        region,options = get_image_sectionsv2(image,q)
        selected_result = get_result(region,options)
        datadict[q12md['name']] = selected_result
        # print(f"{q12md['name']}-{selected_result}")
    dataframe = pd.DataFrame([datadict])
    return dataframe
    

if __name__ == "__main__":
    anchornumber = 2
    data = readjson('payload.json')
    createmetadatfile(anchornumber,data)
    metadata = None
    with open('metadata.json', 'r') as f:
      metadata = json.load(f)
    template = io.imread("4.jpg")
    df_concat = pd.DataFrame()
    for images in os.listdir("./imgdata"):
        image = io.imread(os.path.join("imgdata",images))
        print(images)
        df = processoneimage(data,template,image,anchornumber)
        df_concat = pd.concat([df_concat, df], axis=0)
    df_concat.to_csv('output.csv', index=False)

