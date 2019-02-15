import argparse
import json
import cv2
import matplotlib.pyplot as plt

from predict import *
from postprocessing import draw_boxes


def main(args):
    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    train = True
    if train:
        yolo = YOLO(config)
        #yolo.load_weights(config['model']['saved_model_name'])
        yolo.train()
    else:
        #yolo = YOLO(config)
        p = "1.png"
        im = cv2.imread(p)

        # Directly from file
        # model = get_model_from_file(config)
        # netout = predict_with_model_from_file(config, model, p)

        # From YOLO class
        model = get_model(config)
        netout = model.predict(p)

        draw_boxes(im, netout, config['model']['classes'])

        plt.imshow(np.squeeze(im))
        plt.show()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description='Train and validate autonomous car module')

    arg_parser.add_argument(
        '-c',
        '--conf',
        help='path to the configuration file')

    main(arg_parser.parse_args())
