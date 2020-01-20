
import time 
setup = time.perf_counter()

import os 
import cv2
import numpy as np 
import tensorflow as tf 
import sys 
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util 
# https://medium.com/datadriveninvestor/mobile-object-detector-with-tensorflow-lite-9e2c278922d0
# https://github.com/QuantuMobileSoftware/mobile_detector


CWD_PATH = os.getcwd() 

PATH_TO_LABELS = os.path.join(CWD_PATH, 'saved_inference_graph_models', 'labelmap.pbtxt') 
NUM_CLASSES = 1

PATH_TO_FROZEN_GRAPH = os.path.join(CWD_PATH, 'saved_inference_graph_models', 'tflite_graph-v4.pb')  # "/content/drive/My Drive/Colab Notebooks/EnemyDetection/inference_graph/tflite_graph.pb"
PATH_TO_LITE_MODEL   = os.path.join(CWD_PATH, 'saved_inference_graph_models', 'detect-v4.tflite') # "/content/drive/My Drive/Colab Notebooks/EnemyDetection/inference_graph/detect.tflite"

IMAGE_NAME = 'j_248-4-1.png'
PATH_TO_IMAGE = os.path.join(CWD_PATH, 'samples', IMAGE_NAME)

from abc import ABC, abstractmethod
class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, frame, threshold=0.0):
        pass

class ObjectDetectorDetectionAPI(ObjectDetector):
    def __init__(self, graph_path=PATH_TO_FROZEN_GRAPH):
        """
            Builds Tensorflow graph, load model and labels
        """

        # model_path = path.join(basepath, graph_path)

        # Load Tensorflow model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Load lebel_map
        self._load_label(PATH_TO_LABELS, NUM_CLASSES, use_disp_name=True)

        with self.detection_graph.as_default():
            self.sess = tf.Session(graph=self.detection_graph)
            # Definite input and output Tensors for detection_graph
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def close(self):
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()

    def detect(self, frame, threshold=0.1):
        """
            Predicts person in frame with threshold level of confidence
            Returns list with top-left, bottom-right coordinates and list with labels, confidence in %
        """
        frames = np.expand_dims(frame, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores,
                 self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: frames})

        # Find detected boxes coordinates
        return [self._boxes_coordinates(frame,
                            np.squeeze(boxes[0]),
                            np.squeeze(i[2]).astype(np.int32),
                            np.squeeze(i[3]),
                            min_score_thresh=threshold,
                            ) for i in zip(frames, boxes, classes, scores)][0]


    def _boxes_coordinates(self,
                            image,
                            boxes,
                            classes,
                            scores,
                            max_boxes_to_draw=20,
                            min_score_thresh=.5):
        """
          This function groups boxes that correspond to the same location
          and creates a display string for each detection and overlays these
          on the image.
          Args:
            image: uint8 numpy array with shape (img_height, img_width, 3)
            boxes: a numpy array of shape [N, 4]
            classes: a numpy array of shape [N]
            scores: a numpy array of shape [N] or None.  If scores=None, then
              this function assumes that the boxes to be plotted are groundtruth
              boxes and plot all boxes as black with no classes or scores.
            category_index: a dict containing category dictionaries (each holding
              category index `id` and category name `name`) keyed by category indices.
            use_normalized_coordinates: whether boxes is to be interpreted as
              normalized coordinates or not.
            max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
              all boxes.
            min_score_thresh: minimum score threshold for a box to be visualized
        """

        if not max_boxes_to_draw:
            max_boxes_to_draw = boxes.shape[0]
        number_boxes = min(max_boxes_to_draw, boxes.shape[0])
        person_boxes = []
        # person_labels = []
        for i in range(number_boxes):
            if scores is None or scores[i] > min_score_thresh:
                box = tuple(boxes[i].tolist())
                ymin, xmin, ymax, xmax = box

                im_height, im_width, _ = image.shape
                left, right, top, bottom = [int(z) for z in (xmin * im_width, xmax * im_width,
                                                             ymin * im_height, ymax * im_height)]

                person_boxes.append([(left, top), (right, bottom), scores[i],
                                     self.category_index[classes[i]]['name']])
        return person_boxes

    def _load_label(self, path, num_c, use_disp_name=True):
        """
            Loads labels
        """
        label_map = label_map_util.load_labelmap(path)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_c,
                                                                    use_display_name=use_disp_name)
        self.category_index = label_map_util.create_category_index(categories)

class ObjectDetectorLite(ObjectDetectorDetectionAPI):
    def __init__(self, model_path=PATH_TO_LITE_MODEL):
        """
            Builds Tensorflow graph, load model and labels
        """

        # Load lebel_map
        self._load_label(PATH_TO_LABELS, NUM_CLASSES, use_disp_name=True)

        # Define lite graph and Load Tensorflow Lite model into memory
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def detect(self, image, threshold=0.1):
        """
            Predicts person in frame with threshold level of confidence
            Returns list with top-left, bottom-right coordinates and list with labels, confidence in %
        """

        # Resize and normalize image for network input
        frame = cv2.resize(image, (300, 300))
        frame = np.expand_dims(frame, axis=0)
        frame = (2.0 / 255.0) * frame - 1.0
        frame = frame.astype('float32')

        # run model
        self.interpreter.set_tensor(self.input_details[0]['index'], frame)
        self.interpreter.invoke()

        # get results
        boxes = self.interpreter.get_tensor(
            self.output_details[0]['index'])
        classes = self.interpreter.get_tensor(
            self.output_details[1]['index'])
        scores = self.interpreter.get_tensor(
            self.output_details[2]['index'])
        num = self.interpreter.get_tensor(
            self.output_details[3]['index'])

        # Find detected boxes coordinates
        return self._boxes_coordinates(image,
                            np.squeeze(boxes[0]),
                            np.squeeze(classes[0]+1).astype(np.int32),
                            np.squeeze(scores[0]),
                            min_score_thresh=threshold)

    def close(self):
        pass


###################################################################################################
detector = ObjectDetectorLite()
image = cv2.cvtColor(cv2.imread(PATH_TO_IMAGE), cv2.COLOR_BGR2RGB)

s = time.perf_counter()

result = detector.detect(image, 0.2)

e = time.perf_counter()

print(result)

print("Start " + str(s) + "; end " + str(e) + ", thus:")
print("SETUP TOOK " + str( (s - setup) )  )
print("MODEL TOOK " + str( (e - s) ) + " ")


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

