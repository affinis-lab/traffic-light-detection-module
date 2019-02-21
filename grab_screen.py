import copy
import json
import os
import argparse
import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api
import tensorflow as tf
from keras.models import load_model
from mss import mss

from yolo import dummy_loss
from postprocessing import decode_netout, draw_boxes


BASE_DIR = os.path.dirname(__file__)


def grab_screen_slower(region=None):
    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())


    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)


def grab_and_broadcast_screen(config):
    paused = False

    mon = {'top': 10, 'left': 10, 'width': 750, 'height': 680}
    sct = mss()

    dummy_array = np.zeros((1, 1, 1, 1, config['model']['max_obj'], 4))

    model = load_model("checkpoints/traffic-light-model.h5",
                       custom_objects={'custom_loss': dummy_loss, 'tf': tf})

    frame_num = 0
    while (True):

        if not paused:
            screen = np.array(sct.grab(mon))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

            plt_image = copy.deepcopy(screen)

            screen = cv2.resize(screen, (config['model']['image_h'], config['model']['image_w']))
            screen = screen / 255.
            screen = np.expand_dims(screen, 0)

            netout = model.predict([screen, dummy_array])[0]

            boxes = decode_netout(netout,
                                  obj_threshold=config['model']['obj_thresh'],
                                  nms_threshold=config['model']['nms_thresh'],
                                  anchors=config['model']['anchors'],
                                  nb_class=config['model']['num_classes'])

            plt_image = draw_boxes(plt_image, boxes, labels=config['model']['classes'])
            cv2.imshow('window', cv2.cvtColor(plt_image, cv2.COLOR_BGR2RGB))

            key = cv2.waitKey(10) & 0xff
            if key == 27:
                cv2.destroyAllWindows()
                break


            #save images when o or O is pressed
            if key == ord('o') or key == ord('O'):
                cv2.imwrite(os.path.join(BASE_DIR, 'out', str(frame_num)+".png"), cv2.cvtColor(plt_image, cv2.COLOR_BGR2RGB))
                frame_num += 1


def main(args):
    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    grab_and_broadcast_screen(config)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '-c',
        '--conf',
        default='config.json',
        help='Path to configuration file')

    args = argparser.parse_args()
    main(args)