OPTIONS_MAP = {0:"a",1:"b",2:"c",3:"d",4:"Multiple Selected",5:"No Selected"}
OPTIONS_MAP_ROLL = {0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",10:"MS",11:"NS"}

ROLLLENGTH = 10
OPTIONS = 4


RESULT_THRESH = 80
ROLL_THRESH = 100
DATA_THRESH = 50

calculatedanchorxbuffer = 7
calculatedanchorybuffer = 7

# Shantanu Sir Config
# TYPE_CONFIG = {
#     "Question" : {
#         'OPTIONS':{0:"a",1:"b",2:"c",3:"d",4:"Multiple Selected",5:"No Selected"},
#         'LENGTH':4,
#         },
#     "hall_ticket_no_parent" : {
#         'OPTIONS':{0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",10:"MS",11:"NS"},
#         'LENGTH':10,
#         },
#     "test_booklet_parent":{
#         'OPTIONS':{0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",10:"MS",11:"NS"},
#         'LENGTH':10,
#     },
#     "Form_no_parent":{
#         'OPTIONS':{0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",10:"MS",11:"NS"},
#         'LENGTH':10,
#     }
# }

# Palit Sir Config
TYPE_CONFIG = {
    "Question" : {
        'OPTIONS':{0:"a",1:"b",2:"c",3:"d",4:"Multiple Selected",5:"No Selected"},
        'LENGTH':4,
        },
    "Rollnumber" : {
        'OPTIONS':{0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",10:"MS",11:"NS"},
        'LENGTH':10,
        },
    "Questionpaper" : {
        'OPTIONS':{0:"a",1:"b"},
        'LENGTH':2,
    }
}