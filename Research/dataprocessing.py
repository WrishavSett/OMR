import json
from sectionprocessing import getmetadataforblock
from scanomr import getsortedanchor
def readjson(filename):
    global data
    with open(filename, 'r') as f:
        data = json.load(f)
    return data