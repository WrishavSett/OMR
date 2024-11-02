import json
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage import io
from skimage.feature import match_template
from skimage.feature import peak_local_max # new import!
import os
import pandas as pd
from preprocess import *
from sectionprocessing import *
from dataprocessing import *
from config import *
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

def getsortedanchor(data,key="x"):
  allanchors = []
  accesskey = "name"
  for elements in data:
    if "type" in elements:
       accesskey = "type"
    if elements[accesskey] == "anchor":
      allanchors.append(elements)
  sortedanchor = sorted(allanchors, key=lambda d: d["start"][key])
  return sortedanchor

def get_template(template,sortedanchor,anchornumber):
  temp = template[int(sortedanchor[anchornumber]["start"]["y"]):int(sortedanchor[anchornumber]["end"]["y"]),int(sortedanchor[anchornumber]["start"]["x"]):int(sortedanchor[anchornumber]["end"]["x"])]
  return temp

def get_template_search_area(data):
  sorted_anchor = getsortedanchor(data,"y")
  start_point = sorted_anchor[0]["end"]["y"] + 50
  end_point = sorted_anchor[len(sorted_anchor)-1]["end"]["y"] - 50
  return (int(start_point),int(end_point))

def get_only_options_from_children(data_element):
  for x in data_element["children"]:
    if x["type"] == "number":
      data_element["children"].remove(x)
  return data_element

def createmetadatfile(anchornumber,data):
  sortedanchor = getsortedanchor(data)
  metadatalist = []
  for data_element in data:
    if data_element["type"] != "anchor":
      block_metadata = getmetadataforblock(sortedanchor[anchornumber],data_element)
      metadatalist.append(block_metadata)
  with open("metadata.json", "w") as outfile:
    json.dump(metadatalist, outfile)


def processoneimagefrommetadata(data,template,image,anchornumber,typeconfig):
    sortedanchor = getsortedanchor(data)
    template = process_image(template)
    template = get_template(template,sortedanchor,anchornumber) #template[int(sortedanchor[anchornumber]["start"]["y"]):int(sortedanchor[anchornumber]["end"]["y"]),int(sortedanchor[anchornumber]["start"]["x"]):int(sortedanchor[anchornumber]["end"]["x"])]
    image = process_image(image)

    top_search_point,bottom_search_point = get_template_search_area(data)
    image_roi_top = image[0:top_search_point,:]
    image_roi_bottom = image[bottom_search_point:,:]
    print("Search Points",top_search_point,bottom_search_point)
    peaks_bottom = get_roi_match(image_roi_bottom,template,bottom_search_point,top=False)
    peaks_top = get_roi_match(image_roi_top,template,top=True)
    allpeaks = np.concatenate((peaks_top, peaks_bottom), axis=0)
    sorted_peaks = allpeaks[allpeaks[:, 1].argsort()]
    calculatedanchorx = sorted_peaks[anchornumber][1] - calculatedanchorxbuffer
    calculatedanchory = sorted_peaks[anchornumber][0] - calculatedanchorybuffer

    # if checkalignment(peaks_top,peaks_bottom) == False:
    #    raise Exception("Image is not properly aligned")
    

    print(" Sorted Anchor ,", sortedanchor[anchornumber])
    print(" Calculated value ", (calculatedanchorx,calculatedanchory))
    datadict = {}
    allarr = []
    # Computing mean for the entire image
    for i in range(0,len(data)):
      print(data[i]["name"])
      if "children" in data[i]:
        data_element = get_only_options_from_children(data[i])
        q12md = getmetadataforblock(sortedanchor[anchornumber],data_element)
        q = getactualcoordinates(calculatedanchorx,calculatedanchory,q12md)
        region,options = get_image_sectionsv2(image,q)
        result_arr = get_section_means(options)
        allarr.extend(result_arr)


    # Setting the global threshold for the data
    threshold = get_thresholdv2(allarr,10)
    print(f" The golabl Threshold is {threshold} ")


    # Computing the selected options
    for i in range(0,len(data)):
      print(data[i]["name"])
      flag = False
      if "children" in data[i]:
        data_element = get_only_options_from_children(data[i])
        q12md = getmetadataforblock(sortedanchor[anchornumber],data_element)
        q = getactualcoordinates(calculatedanchorx,calculatedanchory,q12md)
        region,options = get_image_sectionsv2(image,q)
        print(typeconfig[data_element["type"]]["OPTIONS"])
        result_arr,selected_result = get_section_datav2(region,options,\
                                          typeconfig[data_element["type"]]["OPTIONS"],\
                                          threshold) #get_roll(region,options) #get_result(region,options)
        # rollnumber += selected_result
        if selected_result == "RR":
          flag = True
        datadict[q12md['name']] = {"type":data_element["type"],"result":selected_result,"flag":flag,"coord":q}
    # dataframe = pd.DataFrame([datadict])
    # print(allarr)
    return datadict

def process_image_api(image,typeconfig):
  anchornumber = 2
  data = readjson('D:/Rohit/OMR/Research/payload.json')
  template = io.imread("D:/Rohit/OMR/Research/imgdatanewformat/4.jpg")
  df = processoneimagefrommetadata(data,template,image,anchornumber,typeconfig)
  return df

@app.route('/upload', methods=['POST'])
def upload_file():
    typeconfig = {
    "Question" : {
        "OPTIONS":{0:"a",1:"b",2:"c",3:"d",4:"RR",5:"RR"},
        "LENGTH":4,
        },
    "hall_ticket_no_parent" : {
        "OPTIONS":{0:"1",1:"2",2:"3",3:"4",4:"5",5:"6",6:"7",7:"8",8:"9",9:"10",10:"RR",11:"RR"},
        "LENGTH":10,
        },
    "test_booklet_parent":{
        "OPTIONS":{0:"1",1:"2",2:"3",3:"4",4:"5",5:"6",6:"7",7:"8",8:"9",9:"10",10:"RR",11:"RR"},
        "LENGTH":10,
    },
    "Form_no_parent":{
        "OPTIONS":{0:"1",1:"2",2:"3",3:"4",4:"5",5:"6",6:"7",7:"8",8:"9",9:"10",10:"RR",11:"RR"},
        "LENGTH":10,
    }
}
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
        df = process_image_api(np.array(image),typeconfig)
        
        # Convert DataFrame to JSON and return
        return df#.to_json(orient="records")

if __name__ == "__main__":
  app.run(debug=True)


    # anchornumber = 2
    # data = readjson('payload.json')

    # # Don't uncomment
    # # createmetadatfile(anchornumber,data)
    # # metadata = None
    # # with open('metadataimgdatanewformat.json', 'r') as f:
    # #   metadata = json.load(f)

    # template = io.imread("./imgdatanewformat/4.jpg")
    # print(type(template))
    # df_concat = pd.DataFrame()
    # start_time = time.time()
    # for images in os.listdir("./imgdatanewformat"):
    #     image = io.imread(os.path.join("imgdatanewformat",images))
    #     df = processoneimagefrommetadata(data,template,image,anchornumber,TYPE_CONFIG)
    #     df_concat = pd.concat([df_concat, df], axis=0)
    # df_concat.to_csv('output_sh_fmd_2.csv', index=False)
    # print("--- %s seconds ---" % (time.time() - start_time))


