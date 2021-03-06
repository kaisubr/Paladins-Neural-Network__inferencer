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
import multiprocessing
import random
import queue
import win32gui
import win32api, win32con
import pyautogui
import ctypes
from ctypes import wintypes
import time
from win32api import GetSystemMetrics
# from pynput.mouse import Button, Controller


CWD_PATH = os.getcwd() 
NUM_CLASSES = 1

PATH_TO_LABELS = os.path.join(CWD_PATH, 'saved_inference_graph_models', 'labelmap.pbtxt') 
PATH_TO_FROZEN_GRAPH = os.path.join(CWD_PATH, 'saved_inference_graph_models', 'tflite_graph-v4.pb')  # "/content/drive/My Drive/Colab Notebooks/EnemyDetection/inference_graph/tflite_graph.pb"
PATH_TO_LITE_MODEL   = os.path.join(CWD_PATH, 'saved_inference_graph_models', 'detect-v4.tflite') # "/content/drive/My Drive/Colab Notebooks/EnemyDetection/inference_graph/detect.tflite"

IMAGE_NAME = 'j_294-4_noxml-complex.png'
PATH_TO_IMAGE = os.path.join(CWD_PATH, 'samples', IMAGE_NAME)

THRESHOLD = 0.15
pyautogui.PAUSE = 0

## Ready!
print("\n\n\nPALADINS ARTIFICIAL NEURAL NETWORK - INFERENCER V4.4-1: Ready!")
print("Like any other tool, I'll probably need a disclaimer, don't I. *sigh*. DISCLAIMER: The script provided is for research & demonstration purposes only. Please remember this requires a trained model, which you can try making yourself! I am not responsible for any damage (such as being banned from the game) caused through the use of this tool. When making modifications, please follow the LICENSE attached, and remember take the time to read some of my other references' works (see README). Thanks. \n\n")

cwd = os.getcwd() 
w, h, top, left = 0, 0, 0, 0

st, en, count, tot_elapsed = 0, 0, 0, 0
shot_start, shot_end = 0, 0
st = time.perf_counter()
detector = ObjectDetectorLite(PATH_TO_LITE_MODEL, PATH_TO_LABELS, NUM_CLASSES, THRESHOLD)
isRunning = False
# mouse = Controller()

# https://docs.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-mouseinput
# https://pythonprogramming.net/direct-input-game-python-plays-gta-v/
# https://gist.github.com/Aniruddha-Tapas/1627257344780e5429b10bc92eb2f52a
# https://stackoverflow.com/questions/14489013/simulate-python-keypresses-for-controlling-a-game

wintypes.ULONG_PTR = wintypes.WPARAM
user32 = ctypes.WinDLL('user32', use_last_error=True)
INPUT_MOUSE    = 0
class MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx",          wintypes.LONG),
                ("dy",          wintypes.LONG),
                ("mouseData",   wintypes.DWORD),
                ("dwFlags",     wintypes.DWORD),
                ("time",        wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))

class KEYBDINPUT(ctypes.Structure):
    _fields_ = (("wVk",         wintypes.WORD),
                ("wScan",       wintypes.WORD),
                ("dwFlags",     wintypes.DWORD),
                ("time",        wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))

    def __init__(self, *args, **kwds):
        super(KEYBDINPUT, self).__init__(*args, **kwds)
        # some programs use the scan code even if KEYEVENTF_SCANCODE
        # isn't set in dwFflags, so attempt to map the correct code.
        if not self.dwFlags & KEYEVENTF_UNICODE:
            self.wScan = user32.MapVirtualKeyExW(self.wVk,
                                                 MAPVK_VK_TO_VSC, 0)

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (("uMsg",    wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD))

class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = (("ki", KEYBDINPUT),
                    ("mi", MOUSEINPUT),
                    ("hi", HARDWAREINPUT))
    _anonymous_ = ("_input",)
    _fields_ = (("type",   wintypes.DWORD),
                ("_input", _INPUT))

LPINPUT = ctypes.POINTER(INPUT)

def _check_count(result, func, args):
    if result == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    return args


user32.SendInput.errcheck = _check_count
user32.SendInput.argtypes = (wintypes.UINT, # nInputs
                             LPINPUT,       # pInputs
                             ctypes.c_int)  # cbSize

## Move mouse helper. THIS WORKS FOR DX11!
def moveTo(x, y, inGame):
    print("Ok...")
    
    # find dx and dy
    if (not inGame):
        cur = None
        cur = win32gui.GetCursorPos() # tuple
        curX, curY = cur[0], cur[1]
    else:
        curX, curY = int(w/2), int(h/2)

    print("MOUSE (START) AT " + str(curX) + ", " + str(curY) + " ; MODEL (END) AT " + str(x) + ", " + str(y))
    
    moveX = x-curX
    moveY = y-curY

    print("... dx = " + str(moveX) + "; dy = " + str(moveY))

    # MOUSE SPEED must be default.
    # Does not depend on MOUSE ACCELERATION.
    obj = INPUT(type=INPUT_MOUSE,
              mi=MOUSEINPUT(dx=moveX,
                            dy=moveY, 
                            mouseData=0,
                            dwFlags=0x0001,
                            time=0))
    user32.SendInput(1, ctypes.pointer(obj), ctypes.sizeof(obj))

    # extra = ctypes.c_ulong(0)
    # ii_ = Input_I()
    # ii_.mi = MouseInput(100, 100, 0, 0, 0, ctypes.pointer(extra))
    # obj = Input( ctypes.c_ulong(1), ii_ )
    # ctypes.windll.user32.SendInput(1, ctypes.pointer(obj), ctypes.sizeof(obj))


## Runner. The queue has images to parse
def run_model(show, queue): 
    global threads, detector, st, en, count, tot_elapsed# , mouse

    if (not queue.empty()):
        image = queue.get()
        print(str(queue.qsize()) + " remaining")

        result = detector.detect(image, THRESHOLD)
        print(result)

        found = False

        for obj in result:
                print('coordinates: {} {}. class: "{}". confidence: {:.2f}'.
                            format(obj[0], obj[1], obj[3], obj[2]))
                
                if (not found):
                    bzx, bzy = int(w/2 - 150), int(h/2 - 150)
                    print(obj[0]) # top left

                    # coordinates within the bounding box 
                    bx_topleft, by_topleft = int(obj[0][0]), int(obj[0][1])
                    bx_bottomright, by_bottomright = int(obj[1][0]), int(obj[1][1])
                    
                    bx = int((bx_topleft + bx_bottomright)/2)
                    by = int((by_topleft + by_bottomright)/2)

                    # real screen coordinates 
                    x = bzx + bx
                    y = bzy + by

                    # This will move
                    # mouse.position = (x, y)
                    # ctypes.windll.user32.SetCursorPos(x, y)
                    # win32api.SetCursorPos((x, y))
                    # pyautogui.moveTo(x, y, duration=0.0)
                    moveTo(x, y, True) # True = inGame, meaning move dx/dy from center of game, NOT from the current cursor position.

                    # This will click
                    ctypes.windll.user32.mouse_event(2, 0, 0, 0,0) # left down
                    ctypes.windll.user32.mouse_event(4, 0, 0, 0,0) # left up

                    Found = True

                if (show):
                    cv2.rectangle(image, obj[0], obj[1], (0, 255, 0), 2)
                    cv2.putText(image, '{}: {:.2f}'.format(obj[3], obj[2]),
                                (obj[0][0], obj[0][1] - 5),
                                cv2.FONT_HERSHEY_PLAIN, 0.85, (0, 255, 0), 2)
        if (show):
            cv2.imshow(time.strftime("%H:%M:%S", time.localtime()), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

        # if (runagain):
        #     start_program(True)
        #     return
            # threads.pop()
            # t = threading.Thread(target=run, args=(True,))
            # t.start()
            # threads.append(t)
        en = time.perf_counter()
        tot_elapsed += (en - st)
        count += 1
        st = time.perf_counter()

        # I imagine on GPUs it would be magnitudes faster
        # On CPU (turbo boost disabled, ~2.1 GHz) processed about 3.7 FPS. 
        # On CPU (low power mode, ~0.8 GHz) processed about 1.7 FPS.
        print("Successful run. Avg " + str(tot_elapsed/count) + " sec, processing " + str(count/tot_elapsed) + " fps\n")
    print("returning.")
    return
    
# run_modelimage = cv2.cvtColor(cv2.imread(PATH_TO_IMAGE), cv2.COLOR_BGR2RGB)
# run_model(run_modelimage)


def update_screen_resolution():
    global w, h, top, left
    w, h = GetSystemMetrics(0), GetSystemMetrics(1)
    top, left = int(h/2 - 150), int(w/2 - 150)

def grab_screenshot(queue):
    # The screen part to capture
    global isRunning

    while True:
        if isRunning:
            # shot_start = time.perf_counter()
            print("New shot.")
            
            # 720/2 - 150
            print(str(top) + ", " + str(left))
            monitor = {"top": top, "left": left, "width": 300, "height": 300}
            #fname = cwd + '\j_' + str(i) + '-' + str(version) + '.png'
            img = mss.mss().grab(monitor)
            
            #https://stackoverflow.com/questions/51488275/cant-record-screen-using-mss-and-cv2
            img = np.array(img) # Retrieve as BGRA
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB) # Reshape to RGB

            
            queue.put(img)
            # shot_end = time.perf_counter()
            # print("Shot finished in " + str(shot_end - shot_start) + " sec")

            # threading.Timer(0.0, run_model, args=(False, queue,)).start()
            run_model(False, queue)
            
            # queue.join()
            # run_model(img, False)
            #mss.tools.to_png(img.rgb, img.size, output=fname)   

        
    return

# Deprecated, see __main__ 
def start_program():
    global running, st, en, count, tot_elapsed

    if (True):
        en = time.perf_counter()
        tot_elapsed += (en - st)
        count += 1
        st = time.perf_counter()

        print("Successful run. Avg " + str(tot_elapsed/count) + " sec, processing " + str(count/tot_elapsed) + " fps\n")
    
    print("New shot.")
    grab_screenshot(True)
    # threading.Timer(0.2, run).start() 

    return

def flipRunning(event):
    global isRunning, st, en, count, tot_elapsed
    # print(event.name)

    if (event.name == 'caps lock'):
        update_screen_resolution()
        if (not isRunning): 
            isRunning = True
            st, en, count, tot_elapsed = 0, 0, 0, 0
            st = time.perf_counter()
        else :
            isRunning = False

keyboard.on_press(flipRunning)

# Run
if __name__=="__main__":
    update_screen_resolution()

    queue = queue.Queue() # multiprocessing.JoinableQueue()

    grab_screenshot(queue)

    moveTo(0,0, False)
    # grabsct = multiprocessing.Process(target=grab_screenshot, args=(queue, ))
    # modeler = multiprocessing.Process(target=run_model, args=(False, queue, ))

    # starting our processes
    # grabsct.start()
    # modeler.start()
# t = threading.Thread(target=run, args=(True,))
# t.start()
# threads.append(t)