# Paladins Neural Network

Paladins is a first-person shooter with complex game mechanics, such as deployables, reaviling, and crowd control effects.

The Paladins Neural Network detects enemy models and may be used to intelligently aim the player at a target.

How effective is it?
* On i7-6500U (CPU only), my model averaged 0.18 seconds for a single frame image processing.
* 0.8 mAP in 60k steps

Inference graphs were saved from EnemyDetection/inference_graph/saved_model OR frozen_inference_graph.pb if that's available.

Training information:
* Tesla T4 using Google Colab
* The process went through four versions before arriving at ssdlite mobilenet and processing through TFLite 
* 60k steps; 300 training at batch size 24 and L2 regularization. Single class (`model`).
* [300x300] images manually labeled (approx. 300 train / 67 test)

You can see the evolution of different versions in /someshots/.

I plan to upload the Colab .py file and provide a more thorough discussion later. I learned a lot through this experiment, but to combat cheating, I will not release .tflite, .pb, .pbtxt files. This method will be undetectable by EAC since the model only requires the input image.