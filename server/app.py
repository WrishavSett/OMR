import os
import logging
import signal
from flask import Flask, request, jsonify
from flask_cors import CORS
from backgoundprocess import OMRProcessThread

logging.basicConfig(level=logging.INFO, force=True)

app = Flask(__name__)
CORS(app)
# notification_thread.start()

# def sigint_handler(signum, frame):
#     notification_thread.stop()
#     # wait until thread is finished
#     if notification_thread.is_alive():
#         notification_thread.join()
#     logging.info("CLosing thread")

# signal.signal(signal.SIGINT, sigint_handler)

@app.route('/upload', methods=['POST'])
def upload_file():
    body = request.json
    # Check if the required fields are present and not null
    required_fields = ['template', 'template_image', 'type_config']
    missing_fields = [field for field in required_fields if field not in body or body[field] is None]
    
    if missing_fields:
        return jsonify({'error': f"Missing or null fields: {', '.join(missing_fields)}"}), 400


    data_path = body["data_path"] #"D:\Rohit\OMR\Research\imgdatanewformat"
    notification_thread = OMRProcessThread(body["template"],body["template_image"],data_path,body["type_config"]) # BackgroundThreadFactory.create('notification')
    notification_thread.start()
    logging.info("Starting thread from new ")
    return jsonify({'success': 'OK'})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
    # app.run(debug=True)
