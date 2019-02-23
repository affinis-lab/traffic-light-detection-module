# traffic-light-detection-module

![out(2).png](https://github.com/affinis-lab/traffic-light-detection-module/blob/master/out/out2.png)

Module for detecting traffic lights in the [CARLA autonomous driving simulator](http://carla.org/)  (version: 0.8.4).

- Model for objection detection is based on tiny yolov2
- Training started with yolov2 coco pretrained weights
- It was first trained on the LISA traffic light detection dataset (~5800 images), and after that on the dataset collected from the CARLA simulator by myself (~1800 images).


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
  - Real time traffic light detecting gifs
  - Link to datasets used
  - Link to model weights
  
- Several examples of predictions, more can be found in the [out folder](https://github.com/affinis-lab/traffic-light-detection-module/tree/master/out)

![out(11).png](https://github.com/affinis-lab/traffic-light-detection-module/blob/master/out/out11.png)
![out(12).png](https://github.com/affinis-lab/traffic-light-detection-module/blob/master/out/out12.png)
![out(6).png](https://github.com/affinis-lab/traffic-light-detection-module/blob/master/out/out6.png)
![out(7).png](https://github.com/affinis-lab/traffic-light-detection-module/blob/master/out/out7.png)
![out(14).png](https://github.com/affinis-lab/traffic-light-detection-module/blob/master/out/out14.png)
![out(15).png](https://github.com/affinis-lab/traffic-light-detection-module/blob/master/out/out15.png)
![out(4).png](https://github.com/affinis-lab/traffic-light-detection-module/blob/master/out/out4.png)
