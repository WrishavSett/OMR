import json
import requests

# Read JSON data from two files
with open('D:\Rohit\OMR\Research\payload.json', 'r') as file_a:
    data_a = json.load(file_a)

# Construct the JSON body
payload = {
    'template': data_a,  # Assuming data_a is for the key "template"
    'template_image': "/app/data/imgdatanewformat/4.jpg",  # Assuming data_b is for the key "template_image"
    'data_path' : "/app/data/imgdatanewformat",
    'type_config': {
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
}

# Send POST request
url = 'http://185.199.53.224:8000/upload'
response = requests.post(url, json=payload)

# Print response from server
print(response.status_code)
print(response.json())
