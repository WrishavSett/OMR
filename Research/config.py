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
#         'OPTIONS':{0:"1",1:"2",2:"3",3:"4",4:"5",5:"6",6:"7",7:"8",8:"9",9:"10",10:"MS",11:"NS"},
#         'LENGTH':10,
#         },
#     "test_booklet_parent":{
#         'OPTIONS':{0:"1",1:"2",2:"3",3:"4",4:"5",5:"6",6:"7",7:"8",8:"9",9:"10",10:"MS",11:"NS"},
#         'LENGTH':10,
#     },
#     "Form_no_parent":{
#         'OPTIONS':{0:"1",1:"2",2:"3",3:"4",4:"5",5:"6",6:"7",7:"8",8:"9",9:"10",10:"MS",11:"NS"},
#         'LENGTH':10,
#     }
# }

# Palit Sir Config
TYPE_CONFIG = {
    "question" : {
        'OPTIONS':{0:"a",1:"b",2:"c",3:"d",4:"Multiple Selected",5:"No Selected"},
        'LENGTH':4,
        },
    "roll_number" : {
        'OPTIONS':{0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",10:"MS",11:"NS"},
        'LENGTH':10,
        },
    "question_paper_set" : {
        'OPTIONS':{0:"a",1:"b"},
        'LENGTH':2,
    }
}


#plit sir 2 config
# TYPE_CONFIG = {
#     "Question" : {
#         'OPTIONS':{0:"a",1:"b",2:"c",3:"d",4:"Multiple Selected",5:"No Selected"},
#         'LENGTH':4,
#         },
#     "Rollnumber" : {
#         'OPTIONS':{0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",10:"MS",11:"NS"},
#         'LENGTH':10,
#         },
#     "Centercode" : {
#         'OPTIONS':{0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",10:"MS",11:"NS"},
#         'LENGTH':10,
#         },
#     "BookletSerialNo" : {
#         'OPTIONS':{0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",10:"MS",11:"NS"},
#         'LENGTH':10,
#         },
#     "Catagory" : {
#         'OPTIONS':{0:"1",1:"MS",2:"NS"},
#         'LENGTH':1,
#         },
#     "Gender" : {
#         'OPTIONS':{0:"1",1:"MS",2:"NS"},
#         'LENGTH':1,
#         },
#     "lang1" : {
#         'OPTIONS':{0:"1",1:"MS",2:"NS"},
#         'LENGTH':1,
#         },
#     "lang2" : {
#         'OPTIONS':{0:"1",1:"MS",2:"NS"},
#         'LENGTH':1,
#         },
#     "Sub_Grp" : {
#         'OPTIONS':{0:"1",1:"MS",2:"NS"},
#         'LENGTH':1,
#         }
# }