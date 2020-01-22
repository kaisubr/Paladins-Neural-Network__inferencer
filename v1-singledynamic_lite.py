import time
import os 
import cv2
import numpy as np 
import tensorflow as tf 
import sys 
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util 
import ObjectDetector
import ObjectDetectorDetectionAPI
from ObjectDetectorLite import ObjectDetectorLite

# and some of screen imports
from mss import mss
import mss.tools
import keyboard
import threading
import random
from win32api import GetSystemMetrics


CWD_PATH = os.getcwd() 
NUM_CLASSES = 1

PATH_TO_LABELS = os.path.join(CWD_PATH, 'saved_inference_graph_models', 'labelmap.pbtxt') 
PATH_TO_FROZEN_GRAPH = os.path.join(CWD_PATH, 'saved_inference_graph_models', 'tflite_graph-v4.pb')  # "/content/drive/My Drive/Colab Notebooks/EnemyDetection/inference_graph/tflite_graph.pb"
PATH_TO_LITE_MODEL   = os.path.join(CWD_PATH, 'saved_inference_graph_models', 'detect-v4.tflite') # "/content/drive/My Drive/Colab Notebooks/EnemyDetection/inference_graph/detect.tflite"

IMAGE_NAME = 'j_294-4_noxml-complex.png'
PATH_TO_IMAGE = os.path.join(CWD_PATH, 'samples', IMAGE_NAME)

THRESHOLD = 0.1

## Ready!
print("\n\n\nPALADINS ARTIFICIAL NEURAL NETWORK - INFERENCER V4.4-1: Ready!")
print("Like any other tool, I'll probably need a disclaimer, don't I. *sigh*. DISCLAIMER: The script provided is for research & demonstration purposes only. Please remember this requires a trained model, which you can try making yourself! I am not responsible for any damage (such as being banned from the game) caused through the use of this tool. When making modifications, please follow the LICENSE attached, and remember take the time to read some of my other references' works (see README). Thanks. \n\n")

## Runner
def test(image): 
    detector = ObjectDetectorLite(PATH_TO_LITE_MODEL, PATH_TO_LABELS, NUM_CLASSES, THRESHOLD)
    result = detector.detect(image, 0.2)
    print(result)

    for obj in result:
            print('coordinates: {} {}. class: "{}". confidence: {:.2f}'.
                        format(obj[0], obj[1], obj[3], obj[2]))

            cv2.rectangle(image, obj[0], obj[1], (0, 255, 0), 2)
            cv2.putText(image, '{}: {:.2f}'.format(obj[3], obj[2]),
                        (obj[0][0], obj[0][1] - 5),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    cv2.imshow("window", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
    
# testimage = cv2.cvtColor(cv2.imread(PATH_TO_IMAGE), cv2.COLOR_BGR2RGB)
# test(testimage)

cwd = os.getcwd() 

w = 0
h = 0
top = 0
left = 0

def update_screen_resolution():
    global w, h, top, left
    w, h = GetSystemMetrics(0), GetSystemMetrics(1)
    top, left = int(h/2 - 150), int(w/2 - 150)

def takeshot_center():
    # The screen part to capture
    
    # 720/2 - 150
    print(str(top) + ", " + str(left))
    monitor = {"top": top, "left": left, "width": 300, "height": 300}
    #fname = cwd + '\j_' + str(i) + '-' + str(version) + '.png'
    img = mss.mss().grab(monitor)
    
    #https://stackoverflow.com/questions/51488275/cant-record-screen-using-mss-and-cv2
    img = np.array(img) # Retrieve as BGRA
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB) # Reshape to RGB

    test(img)
    #mss.tools.to_png(img.rgb, img.size, output=fname)   

# Run once, dynamically!
def run():
    takeshot_center()

update_screen_resolution()
run()