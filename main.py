import argparse
import json

from yolo import YOLO


def main(args):
    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    yolo = YOLO(config)
    if config['train']['enabled']:
        #yolo.load_weights(config['model']['saved_model_name'])
        yolo.train()
    else:
        yolo.evaluate()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description='Train and validate autonomous car module')

    arg_parser.add_argument(
        '-c',
        '--conf',
        help='path to the configuration file')

    main(arg_parser.parse_args())
