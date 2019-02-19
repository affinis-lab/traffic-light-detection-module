import numpy as np
import keras
import cv2
import copy
import os
from imgaug import augmenters as iaa
from sklearn.preprocessing import LabelEncoder

from postprocessing import interval_overlap


BASE_DIR = os.path.dirname(__file__)
IMAGES_DIR = os.path.join(BASE_DIR, 'dataset', 'images')


def bbox_iou(box1, box2):
    # 0   ,1   ,2   ,3
    # xmin,ymin,xmax,ymax
    intersect_w = interval_overlap([box1[0], box1[2]], [box2[0], box2[2]])
    intersect_h = interval_overlap([box1[1], box1[3]], [box2[1], box2[3]])

    intersect = intersect_w * intersect_h

    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


class BatchGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, config, dataset, shuffle=True, jitter = True):
        'Initialization'
        self.config = config
        self.dataset = dataset

        self.image_h = config['model']['image_h']
        self.image_w = config['model']['image_w']
        self.n_channels = 3

        self.grid_h = config['model']['grid_h']
        self.grid_w = config['model']['grid_w']

        self.n_classes = config['model']['num_classes']
        self.labels = config['model']['classes']

        self.batch_size = config['train']['batch_size']
        self.max_obj = config['model']['max_obj']

        self.shuffle = shuffle
        self.jitter = jitter

        self.nb_anchors = int(len(config['model']['anchors']) / 2)

        self.anchors = [[0, 0, config['model']['anchors'][2 * i], config['model']['anchors'][2 * i + 1]] for i in
                        range(int(len(config['model']['anchors']) // 2))]

        self.on_epoch_end()

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        self.aug_pipe = iaa.Sequential(
            [
                # apply the following augmenters to most images
                # iaa.Fliplr(0.5), # horizontally flip 50% of all images
                # iaa.Flipud(0.2), # vertically flip 20% of all images
                # sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
                #sometimes(iaa.Affine(
                    # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    # rotate=(-5, 5), # rotate by -45 to +45 degrees
                    # shear=(-5, 5), # shear by -16 to +16 degrees
                    # order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    # cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                #)),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 3),
                           [
                               # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),
                                   # blur image using local means with kernel sizes between 2 and 7
                                   iaa.MedianBlur(k=(3, 11)),
                                   # blur image using local medians with kernel sizes between 2 and 7
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                               # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                               # search either for all edges or for directed edges
                               # sometimes(iaa.OneOf([
                               #    iaa.EdgeDetect(alpha=(0, 0.7)),
                               #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                               # ])),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               # add gaussian noise to images
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                   # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               ]),
                               # iaa.Invert(0.05, per_channel=True), # invert color channels
                               iaa.Add((-10, 10), per_channel=0.5),
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               # change brightness of images (50-150% of original value)
                               iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                               # iaa.Grayscale(alpha=(0.0, 1.0)),
                               # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                               # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(float(len(self.dataset)) / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'

        '''
        l_bound = index*self.config['BATCH_SIZE']
        r_bound = (index+1)*self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']
        '''

        le = LabelEncoder()
        le.fit_transform(self.labels)

        x_batch = np.zeros((self.batch_size, self.image_h, self.image_w, self.n_channels))
        b_batch = np.zeros((self.batch_size, 1, 1, 1, self.max_obj, 4))

        y_batch = np.zeros((self.batch_size, self.grid_h, self.grid_w, self.nb_anchors, 4 + 1 + self.num_classes()))  # desired network output

        #current_batch = self.dataset[l_bound:r_bound]
        current_batch = self.dataset[index * self.batch_size:(index + 1) * self.batch_size]

        instance_num = 0

        for instance in current_batch:
            img, object_annotations = self.prep_image_and_annot(instance, jitter=self.jitter)

            obj_num = 0

            # center of the bounding box is divided with the image width/height and grid width/height
            # to get the coordinates relative to a single element of a grid
            for obj in object_annotations:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['class'] in self.labels:
                    center_x = .5 * (obj['xmin'] + obj['xmax'])     # center of the lower side of the bb (by x axis)
                    center_x = center_x / (float(self.image_w) / self.grid_w)     # scaled to the grid unit (a value between 0 and GRID_W-1)
                    center_y = .5 * (obj['ymin'] + obj['ymax'])     # center of the lower side (by y axis)
                    center_y = center_y / (float(self.image_h) / self.grid_h)    # scaled to the grid unit (a value between 0 and GRID_H-1)

                    grid_x = int(np.floor(center_x))    # assigns the object to the matching
                    grid_y = int(np.floor(center_y))     # grid element according to (center_x, center_y)

                    if grid_x < self.grid_w and grid_y < self.grid_h:
                        center_w = (obj['xmax'] - obj['xmin']) / (float(self.image_w) / self.grid_w)
                        center_h = (obj['ymax'] - obj['ymin']) / (float(self.image_h) / self.grid_h)

                        box = [center_x, center_y, center_w, center_h]

                        # find the anchor that best predicts this box
                        best_anchor = -1
                        max_iou = -1

                        shifted_box = [0, 0, center_w, center_h]

                        for i in range(len(self.anchors)):
                            anchor = self.anchors[i]
                            iou = bbox_iou(shifted_box, anchor)

                            if max_iou < iou:
                                best_anchor = i
                                max_iou = iou

                        classes = [0, 0]

                        obj_label = int(le.transform([obj['class']]))

                        if obj_label == 0:
                            classes[0] = 1
                        else:
                            classes[1] = 1

                        img = self.normalize(img)

                        x_batch[instance_num] = img

                        b_batch[instance_num, 0, 0, 0, obj_num] = box
                        y_batch[instance_num, grid_y, grid_x, best_anchor] = [box[0], box[1], box[2], box[3], 1.0, classes[0], classes[1]]

                        obj_num += 1
                        obj_num %= self.max_obj

            instance_num += 1

        return [x_batch, b_batch], y_batch


    def prep_image_and_annot(self, dataset_instance, jitter):
        image_path = dataset_instance['image_path']
        image = self.load_image(os.path.join(IMAGES_DIR,image_path))

        h, w, c = image.shape

        if jitter:
            image = self.aug_pipe.augment_image(image)

        # resize the image to standard size
        image = cv2.resize(image, (self.image_h, self.image_w))

        object_annotations = copy.deepcopy(dataset_instance['object'])
        for obj in object_annotations:
            for attr in ['xmin', 'xmax']:
                obj[attr] = int(obj[attr] * float(self.image_w) / w)
                obj[attr] = max(min(obj[attr], self.image_w), 0)

            for attr in ['ymin', 'ymax']:
                obj[attr] = int(obj[attr] * float(self.image_h) / h)
                obj[attr] = max(min(obj[attr], self.image_h), 0)

        return image, object_annotations


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.dataset)


    def load_image(self, path):
        img = cv2.imread(os.path.join(IMAGES_DIR, path))

        try:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            else:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        except:
            print(path)

        return img


    def load_annotation(self, i):
        annots = []

        for obj in self.dataset[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.labels.index(obj['class'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)


    def normalize(self, image):
        return image/255.


    def num_classes(self):
        return len(self.labels)


    def size(self):
        return len(self.dataset)