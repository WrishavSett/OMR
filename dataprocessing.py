import json
from sectionprocessing import getmetadataforblock
from scanomr import getsortedanchor

def createmetadatfile(anchornumber,data):
  sortedanchor = getsortedanchor(data)
  metadatalist = []
  for data_element in data:
    if data_element["type"] != "anchor":
      block_metadata = getmetadataforblock(sortedanchor[anchornumber],data_element)
      metadatalist.append(block_metadata)
  with open("metadata.json", "w") as outfile:
    json.dump(metadatalist, outfile)



def readjson(filename):
    global data
    with open(filename, 'r') as f:
        data = json.load(f)
    return data