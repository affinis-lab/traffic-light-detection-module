from keras.models import load_model
import os
import numpy as np

from yolo import YOLO, dummy_loss
from preprocessing import load_image_predict
from postprocessing import decode_netout


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_model(config):
    model = YOLO(
         config =config
    )
    model.load_weights(os.path.join(BASE_DIR, config['model']['saved_model_name']))
    return model


def get_model_from_file(config):
    path = os.path.join(BASE_DIR, 'checkpoints', config['model']['saved_model_name'])
    model = load_model(path, custom_objects={'custom_loss': dummy_loss})
    return model


def predict_with_model_from_file(config, model, image_path):
    image = load_image_predict(image_path, config['model']['image_h'], config['model']['image_w'])

    dummy_array = np.zeros((1, 1, 1, 1, config['model']['max_obj'], 4))
    netout = model.predict([image, dummy_array])[0]

    boxes = decode_netout(netout=netout, anchors=config['model']['anchors'],
                          nb_class=config['model']['num_classes'],
                          obj_threshold=config['model']['obj_thresh'],
                          nms_threshold=config['model']['nms_thresh'])
    return boxes