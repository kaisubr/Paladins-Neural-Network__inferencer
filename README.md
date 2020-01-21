# Paladins Artificial Neural Network: Inferencing

This inferencer in this repository applies [the model that I created here](https://github.com/kaisubr/PANN__trainer-public) onto real-time game data! The inferencer's versioning system will append the model's version (i.e. v4.4-0).

Paladins is a first-person shooter with complex game mechanics, such as deployables, revealing, and crowd control effects.

The Paladins Artificial Neural Network uses a convolutional neural network that detects enemy models and may be used to intelligently aim the player at a target.
* The network can detect obfuscated enemies, such as revealed enemies behind walls
* In some cases, the model was able to predict almost completely obstructed bodies
* The model is able detect partial bodies (such as a torso but no legs)
* The model is able to differentiate enemies and allies in complicated environments

## How effective is it?
After several iterations, v4.4 performs as follows:
* On i7-6500U (CPU only), my model averaged 0.18 seconds for a single frame image processing.
* 0.806 mAP at 0.5 IOU in 60k steps.
* 0.701 mAP at [0.5...0.95] IOU, area = large, in 60k steps.
* Loss for final step was 1.8736447. This can be lowered by further training.

![alt text](/someshots/mAP.PNG "")

Inference graphs were saved from EnemyDetection/inference_graph/saved_model OR frozen_inference_graph.pb if that's available. For the ssdlite network, it was saved through Tensorflow toco `tflite_convert` (alternatively you can use `bazel`).

### What did the results look like?

#### In-game real-time input
v4.4-1 is still being worked on! I will likely continue using Python `mss` to take in data.

#### Single frame inputs
v4.4-0 was designed for single frame (300x300 image) inputs.
* It was able to detect nearly completely obstructed bodies in complex game environments.
    * The image below was discarded from training because it was too ambiguous to label, but the model was still able to detect the enemy located behind the ally. Notice that this was taken from a real Paladins game, with no 'bots' (such as 'Bot 9'), so overfitting to match label names above the enemy could not have occurred.
    * ![alt text](/someshots/j_294-4_noxml-complex-v4-detected.png "")
* It was able to detect partially obstructed bodies. The image below was also discarded from training because it was too ambiguous to label, but the model performed well.
    * ![alt text](/someshots/j_211-4-1_noxml-v4-detected.png "")
* It was able to detect obfuscated enemies and revealed enemies behind walls.
    * ![alt text](/someshots/j_248-4-1-v4-detected.png "") ![alt text](/someshots/j_346-4-1-v4-detected.png "")
* It could differentiate players from allies. Here, no enemies are detected:
    * ![alt text](/someshots/j_290-4-1_noxml-noenemies-v4-detected.png "")
* More results can be viewed in /someshots/.

I plan to upload the Colab .py file and provide a more thorough discussion later. I learned a lot through this experiment, but to combat cheating, I will not release .tflite, .pb, .pbtxt files. This method will be undetectable by EAC since the model only requires the input image.

<!-- 
Notes to self: .../raw contains raw data & xml files, along with 0noxml and 0rename 
Drive content/ contains Colab, raw data, xml files, config files, tfevent files, and four training versions
-->
