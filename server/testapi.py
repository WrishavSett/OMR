import json
import requests

# Read JSON data from two files
with open('D:\Rohit\OMR\Research\\test_payload.json', 'r') as file_a:
    data_a = json.load(file_a)

# Construct the JSON body
payload = {
    'template': data_a,  # Assuming data_a is for the key "template"
    "template_image": "/app/data/DCG26/default/1727348883371-BE24-01-01001.jpg",
    "data_path": "/app/data/DCG26/batch3",
    "t_name": "1727348883371-BE24-01-01001.jpg",
    "type_config": {
        "question": {
            "OPTIONS": {
                "0": "A",
                "1": "B",
                "2": "C",
                "3": "D",
                "4": "RR",
                "5": "RR"
            },
            "LENGTH": 4
        },
        "roll_number": {
            "OPTIONS": {
                "0": "0",
                "1": "1",
                "2": "2",
                "3": "3",
                "4": "4",
                "5": "5",
                "6": "6",
                "7": "7",
                "8": "8",
                "9": "9",
                "10": "RR",
                "11": "RR"
            },
            "LENGTH": 10
        },
        "Center Code": {
            "OPTIONS": {
                "0": "B",
                "1": "1",
                "2": "RR",
                "3": "RR"
            },
            "LENGTH": 2
        }
    },
    "batch_name": "batch"

}

# payload = {
#     'template': data_a,  # Assuming data_a is for the key "template"
#     'template_image': "/app/data/imgdatanewformat/4.jpg",  # Assuming data_b is for the key "template_image"
#     'data_path' : "/app/data/imgdatanewformat",
#     "t_name" : "Templane Name",
#     "batch_name" : "Batch Name",
#     'type_config': {
#     "Question" : {
#         "OPTIONS":{0:"a",1:"b",2:"c",3:"d",4:"RR",5:"RR"},
#         "LENGTH":4,
#         },
#     "hall_ticket_no_parent" : {
#         "OPTIONS":{0:"1",1:"2",2:"3",3:"4",4:"5",5:"6",6:"7",7:"8",8:"9",9:"10",10:"RR",11:"RR"},
#         "LENGTH":10,
#         },
#     "test_booklet_parent":{
#         "OPTIONS":{0:"1",1:"2",2:"3",3:"4",4:"5",5:"6",6:"7",7:"8",8:"9",9:"10",10:"RR",11:"RR"},
#         "LENGTH":10,
#     },
#     "Form_no_parent":{
#         "OPTIONS":{0:"1",1:"2",2:"3",3:"4",4:"5",5:"6",6:"7",7:"8",8:"9",9:"10",10:"RR",11:"RR"},
#         "LENGTH":10,
#     }
# }
# }

# Send POST request
url = 'http://157.173.222.15:8000/upload'
response = requests.post(url, json=payload)

# Print response from server
print(response.status_code)
print(response.json())
