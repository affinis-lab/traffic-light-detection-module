import cv2
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder


def load_image_predict(image_path, image_h, image_w):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_h, image_w))
    image = image/255
    image = np.expand_dims(image, 0)

    return image


def load_carla_data(path, labels):
    le = LabelEncoder()
    le.fit_transform(labels)

    data = pd.read_csv(path, delimiter=",", header=None)

    dataset = {}

    objects_omitted = 0
    red = 0
    green = 0
    for record in data[1:][data.columns[:7]].values:
        tokens = record[5].split(",")

        xmin, ymin, xmax, ymax = float(tokens[1].split(":")[1]), float(tokens[2].split(":")[1]),\
                               float(tokens[3].split(":")[1]), float(tokens[4].split(":")[1].replace("}", ""))

        #omit small images
        if xmax < 15:
            objects_omitted += 1
            continue

        xmax += xmin
        ymax += ymin

        if "stop" in record[6]:
            obj_class = "stop"
            red += 1
        else:
            obj_class = "go"
            green += 1

        obj = {}
        obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], obj['class'] = xmin, ymin, xmax, ymax, obj_class

        image_path = record[0]

        #image_path = os.path.join("images", image_path)

        if image_path in dataset:
            dataset[image_path].append(obj)
        else:
            dataset[image_path] = [obj]

    print("Objects omitted", objects_omitted)
    print("Red light: ", red)
    print("Green light: ", green)

    instances = []

    for key in dataset.keys():
        inst = {}

        inst['image_path'] = key
        inst['object'] = dataset[key]

        instances.append(inst)

    return instances


def load_image(path):
    img = cv2.imread(os.path.join(path))

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if len(img.shape) == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img