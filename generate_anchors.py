import random
import argparse
import numpy as np
import cv2
import os
import json

from preprocessing import load_carla_data


argparser = argparse.ArgumentParser()

argparser.add_argument(
    '-c',
    '--conf',
    default='config.json',
    help='path to configuration file')

argparser.add_argument(
    '-a',
    '--anchors',
    default=2,
    help='number of anchors to use')


BASE_DIR = os.path.dirname(__file__)
IMAGES_DIR = os.path.join(BASE_DIR, 'dataset', 'images')


def IOU(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid

        if c_w >= w and c_h >= h:
            similarity = w*h/(c_w*c_h)
        elif c_w >= w and c_h <= h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape

    return np.array(similarities)


def avg_IOU(anns, centroids):
    n,d = anns.shape
    sum = 0.

    for i in range(anns.shape[0]):
        sum+= max(IOU(anns[i], centroids))

    return sum/n


def print_anchors(centroids):
    anchors = centroids.copy()

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    r = "anchors: ["
    for i in sorted_indices[:-1]:
        r += '%0.2f,%0.2f, ' % (anchors[i,0], anchors[i,1])

    r += '%0.2f,%0.2f' % (anchors[sorted_indices[-1:],0], anchors[sorted_indices[-1:],1])
    r += "]"

    print(r)


def run_kmeans(ann_dims, anchor_num):
    ann_num = ann_dims.shape[0]
    prev_assignments = np.ones(ann_num)*(-1)
    iterations = 0
    old_distances = np.zeros((ann_num, anchor_num))

    indices = [random.randrange(ann_dims.shape[0]) for i in range(anchor_num)]
    centroids = ann_dims[indices]
    anchor_dim = ann_dims.shape[1]

    while True:
        distances = []
        iterations += 1
        for i in range(ann_num):
            d = 1 - IOU(ann_dims[i], centroids)
            distances.append(d)
        distances = np.array(distances) # distances.shape = (ann_num, anchor_num)

        print("iteration {}: dists = {}".format(iterations, np.sum(np.abs(old_distances-distances))))

        #assign samples to centroids
        assignments = np.argmin(distances,axis=1)

        if (assignments == prev_assignments).all():
            return centroids

        #calculate new centroids
        centroid_sums=np.zeros((anchor_num, anchor_dim), np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]]+=ann_dims[i]
        for j in range(anchor_num):
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j) + 1e-6)

        prev_assignments = assignments.copy()
        old_distances = distances.copy()


def main(args):
    config_path = args.conf
    num_anchors = args.anchors

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    data = load_carla_data(os.path.join(BASE_DIR, 'dataset',config['train']['annot_file_name']), config['model']['classes'])

    print(len(data))

    np.random.shuffle(data)

    grid_w = config['model']['grid_w']
    grid_h = config['model']['grid_h']

    # run k_mean to find the anchors
    annotation_dims = []
    for image in data:
        img = cv2.imread(os.path.join(IMAGES_DIR,image['image_path']))
        h, w, c = img.shape
        cell_w = w/grid_w
        cell_h = h/grid_h

        for obj in image['object']:
            relative_w = (float(obj['xmax']) - float(obj['xmin']))/cell_w
            relatice_h = (float(obj["ymax"]) - float(obj['ymin']))/cell_h
            annotation_dims.append(tuple(map(float, (relative_w,relatice_h))))

    annotation_dims = np.array(annotation_dims)
    centroids = run_kmeans(annotation_dims, num_anchors)

    # write anchors to file
    print('\naverage IOU for', num_anchors, 'anchors:', '%0.2f' % avg_IOU(annotation_dims, centroids))
    print_anchors(centroids)


if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)