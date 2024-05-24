import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io
from skimage.feature import match_template
from skimage.feature import peak_local_max # new import!
import os
import pandas as pd
from preprocess import *
from sectionprocessing import *
from dataprocessing import *

from flask import Flask, request, jsonify
from PIL import Image
import io as pyio
app = Flask(__name__)

peaks_top = 0
peaks_bottom = 0
calculatedanchorx = 0
calculatedanchory = 0
epsilon = 25
data = None

def getsortedanchor(data):
  allanchors = []
  for elements in data:
    if elements["name"] == "anchor":
      allanchors.append(elements)
  sortedanchor = sorted(allanchors, key=lambda d: d["start"]["x"])
  return sortedanchor

def processoneimage(data,template,image,anchornumber):
    sortedanchor = getsortedanchor(data)
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
    sorted_peaks = allpeaks[allpeaks[:, 1].argsort()]
    calculatedanchorx = sorted_peaks[anchornumber][1] - 7
    calculatedanchory = sorted_peaks[anchornumber][0] - 7

    theta = getskew(sorted_peaks)
    image = rotate_image(image,-theta,(sorted_peaks[0][1],sorted_peaks[0][0]))
    # plt.imshow(image)
    # plt.plot(peaks_bottom[:,1], peaks_bottom[:,0], 'o', markeredgecolor='r', markerfacecolor='none', markersize=2)
    # plt.plot(peaks_top[:,1], peaks_top[:,0], 'o', markeredgecolor='r', markerfacecolor='none', markersize=2)
    # print(calculatedanchorx,calculatedanchory)
    # showsectionandgetregion(image,data,calculatedanchorx,calculatedanchory,sortedanchor,anchornumber,35)

    if checkalignment(peaks_top,peaks_bottom) == False:
       raise Exception("Image is not properly aligned")
    
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

def process_image_api(image):
  anchornumber = 2
  data = readjson('payload.json')
  template = io.imread("./imgdata/4.jpg")
  df = processoneimage(data,template,image,anchornumber)
  return df

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Read the image file
        image_bytes = file.read()
        image = Image.open(pyio.BytesIO(image_bytes))
        
        # Process the image to get DataFrame
        df = process_image_api(np.array(image))
        
        # Convert DataFrame to JSON and return
        return df.to_json(orient="records")
    

if __name__ == "__main__":
   app.run(debug=True)
    # anchornumber = 2
    # data = readjson('payload.json')
    # createmetadatfile(anchornumber,data)
    # metadata = None
    # with open('metadata.json', 'r') as f:
    #   metadata = json.load(f)
    # template = io.imread("./imgdata/4.jpg")
    # print(type(template))
    # df_concat = pd.DataFrame()
    # for images in os.listdir("./imgdata"):
    #     image = io.imread(os.path.join("imgdata",images))
    #     df = processoneimage(data,template,image,anchornumber)
    #     df_concat = pd.concat([df_concat, df], axis=0)
    # df_concat.to_csv('output.csv', index=False)

