import json
import os
import argparse
import cv2

from postprocessing import draw_boxes
from predict import predict_with_model_from_file, get_model_from_file


BASE_DIR = os.path.dirname(__file__)
TEST_IMAGES_DIR = os.path.join(BASE_DIR, 'test_images')
OUT_IMAGES_DIR = os.path.join(BASE_DIR, 'out')


def detect_on_test_images(config):
    model = get_model_from_file(config)

    all_images = [f for f in os.listdir(TEST_IMAGES_DIR) if os.path.isfile(os.path.join(TEST_IMAGES_DIR, f))]
    img_num = 1
    for image_name in all_images:
        image_path = os.path.join(TEST_IMAGES_DIR, image_name)

        netout = predict_with_model_from_file(config, model, image_path)
        plt_image = draw_boxes(cv2.imread(image_path), netout, config['model']['classes'])

        #cv2.imshow('demo', plt_image)
        #cv2.waitKey(0)

        cv2.imwrite(os.path.join(OUT_IMAGES_DIR, 'out' + str(img_num) + '.png'), plt_image)
        img_num += 1


def main(args):
    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    detect_on_test_images(config)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '-c',
        '--conf',
        default='config.json',
        help='Path to configuration file')

    args = argparser.parse_args()
    main(args)