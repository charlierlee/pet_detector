import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv


def process_frame(frame):
    RED = (0, 0, 255)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    path = 'cascades/haarcascade_frontalface_default.xml'
    face_detector = cv2.CascadeClassifier(path)
    face_rects = face_detector.detectMultiScale(gray,
            scaleFactor=1.1,
            minNeighbors=5, 
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE)
    print(f'found {len(face_rects)} face(s)')
    for rect in face_rects:
        cv2.rectangle(frame, rect, RED, 2)
    cv2.imshow('window', frame)

load_dotenv('.env')
camera_url = os.environ.get('CAMERA_URL')
cap = cv2.VideoCapture(camera_url)
EXTENSION = 'jpg'
file_name_format = "{:%Y%m%d_%H%M%S.%f}-{:f}-{:s}.{:s}"
success, image = cap.read()

#################### Setting up parameters ################
seconds = 2
fps = cap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
multiplier = fps * seconds
#################### Initiate Process ################
thres_1 = 3100 #3100 # Chosen threshold to detect face
thres_2 = 3400


while success:
    frameId = int(round(cap.get(1))) #current frame number, rounded b/c sometimes you get frame intervals which aren't integers...this adds a little imprecision but is likely good enough
    success, frame = cap.read()

    if frameId % multiplier == 0:

        process_frame(frame)

    #    time.sleep(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()