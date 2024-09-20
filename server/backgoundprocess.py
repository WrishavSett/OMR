#https://vmois.dev/python-flask-background-thread/
import logging
import threading
from confluent_kafka import Producer
import os
import json
from omrprocessing import processoneimagefrommetadata
import config
from PIL import Image
import numpy as np
from skimage import io
from dotenv import load_dotenv

class OMRProcessThread(threading.Thread):
    def __init__(self,template,template_image,data_path,type_config,processed_omr_result_id):
        super().__init__()
        load_dotenv()
        self._stop_event = threading.Event()
        configproducer = {'bootstrap.servers': os.getenv('DOCKER_HOST_IP')+':9092'}
        self.producer = Producer(configproducer)
        self.path = data_path # "D:\Rohit\OMR\Research\imgdatanewformat"
        self.template = template
        self.template_image = template_image
        self.type_config = type_config
        self.processed_omr_result_id = processed_omr_result_id

    def stop(self) -> None:
        self.producer.flush()
        self._stop_event.set()

    def _stopped(self) -> bool:
        return self._stop_event.is_set() 

    def delivery_callback(self,err, msg):
        if err:
            print('ERROR: Message failed delivery: {}'.format(err))
        else:
            print("Produced event to topic {topic}: key = {key:12} value = {value:12}".format(
                topic=msg.topic(), key=msg.key().decode('utf-8'), value=msg.value().decode('utf-8')))
    
    def process_image_api(self,image):
        anchornumber = 2
        data = self.template # readjson('D:/Rohit/OMR/Research/payload.json')
        template =  io.imread(self.template_image) #  io.imread("D:/Rohit/OMR/Research/imgdatanewformat/4.jpg")
        df = processoneimagefrommetadata(data,template,image,anchornumber,self.type_config)
        return df

    def is_valid_image_extension(self,file_name):
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        return any(file_name.lower().endswith(ext) for ext in valid_extensions)

    def run(self) -> None:
        for file in os.listdir(self.path):
            if self.is_valid_image_extension(file):
                image = Image.open(os.path.join(self.path,file))
                rslt = self.process_image_api(np.array(image))
            key = str(self.processed_omr_result_id)
            # key = str(self.template_name)+ "_" + str(self.batch_name) + "_" + str(file) + "_" + str(self.processed_omr_result_id)
            self.producer.produce("testtopic",json.dumps(rslt).encode('utf-8') , key, callback=self.delivery_callback)
            logging.info(f"File {file} send to kafka")
