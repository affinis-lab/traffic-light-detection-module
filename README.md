# traffic-light-module
Module for detecting traffic lights in the [CARLA autonomous driving simulator](http://carla.org/)  (version: 0.8.4).


- To train: 
  - In the [config](https://github.com/affinis-lab/traffic-light-detection-module/blob/master/config.json) file set _training_ -> _enabled_ to **true**
  - Put your annotations file in the **dataset** folder
  - In the [config](https://github.com/affinis-lab/traffic-light-detection-module/blob/master/config.json) file _set training_ -> _annot_file_name_ to the name of your annotations file
  - Put your images in the **dataset/images** folder
  - If necessary, adjust parameters in [config](https://github.com/affinis-lab/traffic-light-detection-module/blob/master/config.json) according to your problem/dataset
  - run main.py with **-c config.json**
  
- To evaluate:
  - In the [config](https://github.com/affinis-lab/traffic-light-detection-module/blob/master/config.json) file set _training_ -> _enabled_ to **false**
  - Put your annotations file in the **evaluation** folder
  - In the [config](https://github.com/affinis-lab/traffic-light-detection-module/blob/master/config.json) file _set training_ -> _annot_file_name_ to to the name of your annotations file containing images for evaluation
  - Put your images in the **evaluation/images** folder
  - If necessary, adjust parameters in [config](https://github.com/affinis-lab/traffic-light-detection-module/blob/master/config.json) according to your problem/dataset
  - run main.py with **-c config.json**
  
- To generate anchors:
  - run generate_anchors.py with **-c config.json**

- Soon to be added:
  - Prediction images
  - Real time traffic light detecting gifs
  - Several test images
