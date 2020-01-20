# Paladins Neural Network

Paladins is a first-person shooter with complex game mechanics, such as deployables, reaviling, and crowd control effects.

The Paladins Neural Network detects enemy models and may be used to intelligently aim the player at a target.
* The network can detect obfuscated enemies, such as revealed enemies behind walls
* The model is able to differentiate enemies and allies
* The model is able detect partial bodies (such as a torso but no legs)

## How effective is it?
After several iterations, v4.4 performs as follows:
* On i7-6500U (CPU only), my model averaged 0.18 seconds for a single frame image processing.
* 0.806 mAP at 0.5 IOU in 60k steps.
* 0.701 mAP at [0.5...0.95] IOU, area = large, in 60k steps.
* Loss for final step was 1.8736447. This can be lowered by further training.

Inference graphs were saved from EnemyDetection/inference_graph/saved_model OR frozen_inference_graph.pb if that's available.

### What did the results look like?
v4.4 performed as follows:
* It was able to detect obfuscated enemies and revealed enemies behind walls
    * ![alt text](/someshots/j_248-4-1-v4-detected.png "")
    * ![alt text](/someshots/j_346-4-1-v4-detected.png "")
* It was able to detect partially obstructed bodies
    * ![alt text](/someshots/j_211-4-1_noxml-v4-detected.png "")
* It could differentiate players from allies. Here, no enemies are detected:
    * ![alt text](/someshots/j_290-4-1_noxml-noenemies-v4-detected.png "")

### Training information
This is how I trained the network:
* Tesla T4 using Google Colab.
* The process went through four versions before arriving at ssdlite-mobilenet and processing through TFLite.
* I finetuned the model using the COCO dataset.
* 60k steps; 300 training at batch size 24 and L2 regularization. Single class (`model`).
* [300x300] images manually labeled (approx. 300 train / 67 test).

You can see the evolution of different versions in /someshots/.

I plan to upload the Colab .py file and provide a more thorough discussion later. I learned a lot through this experiment, but to combat cheating, I will not release .tflite, .pb, .pbtxt files. This method will be undetectable by EAC since the model only requires the input image.

### Changelog overview
Short discussion:
* v0: Initial training with 250 ish images. Default configuration, stopped after 12k steps.
* v1: Added 300 images, CSV ended up with about 1000 in training. Used train.py. Configuration steps 25000.
* v2: Split images with train/test, so 800 training and about 50 in testing. Also changed to model_main.py, and configuration classes = 1, regularization = 0.01, batch size = 3, eval_config num_examples = 33 images in the testing directory.
      This is working well. I'm worried about prediction times (https://stackoverflow.com/questions/46839073/tensorflow-object-detection-api-rcnn-is-slow-on-cpu-1-frame-per-min)
      Perhaps I might try to train using mobilenet (ssd_mobilenet)
      Other notes: eval.py might force drive update for Tensorboard?
      average_precision: 0.612540 for 23143 steps, very good! Freezing now.

      Compiling on my laptop took ~ 1 minute :(
      Yeah, we need another model.
* v3: Changed model to ssdlite_mobilenet_v2_coco_2018_05_09 in an attempt to improve speed, regularization = 0.004, image resize to 300x300. Cut after 5k steps since mAP was stuck at zero, loss decreasing slowly.
* v4: And that's when I got this genius idea: running mss with a 300x300 window in the middle of the 720p game running! 
      We need MORE DATA! So, now I have deprecated train to images/train_dep3; new and selected images moved to images/train and updated images/test!
      * v4.0: Trained to 17k steps using 720p & 300x300 in testing. 
      * v4.1: Removed all 720p images from testing. Added 150 more images to test, 50 more images to train.
      * v4.2: Continue training to 60k steps. Edit config to 62 images. Enabled shuffling.
      * v4.3: I tried to export using Open CV DNN. This revealed errors importing the model with batch norm layers.
      * v4.4: I realized I had to export as a TFLite graph instead. Works pretty well! Averaged 0.17-0.19 sec for processing.

### Other references
* [Racoon classifier](https://github.com/datitran/raccoon_dataset/tree/93938849301895fb73909842ba04af9b602f677a)
* [Racoon discussion](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)
* [TFLite](https://github.com/QuantuMobileSoftware/mobile_detector)
* [GFG shoe classifier](https://www.geeksforgeeks.org/ml-training-image-classifier-using-tensorflow-object-detection-api/)
* [Sliding windows](http://www.cs.utoronto.ca/~fidler/slides/CSC420/lecture17.pdf)
* [Detection frameworks](https://www.datacamp.com/community/tutorials/object-detection-guide)

<!-- 
Notes to self: .../raw contains raw data & xml files, along with 0noxml and 0rename 
Drive content/ contains Colab, raw data, xml files, config files, tfevent files, and four training versions
-->