from keras.models import Model, load_model
from keras.layers import Reshape, Lambda, Conv2D, Input, MaxPooling2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback
from keras.optimizers import SGD, Adam, RMSprop
import tensorflow as tf
import os
import numpy as np

from postprocessing import decode_netout, interval_overlap, compute_overlap, compute_ap
from preprocessing import load_image_predict, load_carla_data
from utils import BatchGenerator


BASE_DIR = os.path.dirname(__file__)
ANNOT_DIR = os.path.join(BASE_DIR, 'dataset')


class TinyYoloFeature:
    """Tiny yolo feature extractor"""
    def __init__(self, input_size, config):
        input_image = Input(shape=(input_size, input_size, 3))

        # Layer 1
        x = Conv2D(16, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2 - 5
        for i in range(0,4):
            x = Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same', name='conv_' + str(i+2), use_bias=False)(x)
            x = BatchNormalization(name='norm_' + str(i+2))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

        # Layer 7
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_' + str(7), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str(7))(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 8
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_' + str(8), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str(8))(x)
        x = LeakyReLU(alpha=0.1, name = 'last')(x)

        self.feature_extractor = Model(input_image, x)


        pretrained = load_model('checkpoints\\' + config['model']['saved_model_name'], custom_objects={'custom_loss': dummy_loss, 'tf': tf})
        pretrained = pretrained.get_layer('model_1')

        idx = 0
        for layer in self.feature_extractor.layers:
            print(layer.name)
            layer.set_weights(pretrained.get_layer(index=idx).get_weights())
            idx += 1

        frozen = [1, 2, 3, 4, 5, 6, 7]

        for l in frozen:
            self.feature_extractor.get_layer("conv_" + str(l)).trainable = False
            self.feature_extractor.get_layer("norm_" + str(l)).trainable = False

        self.feature_extractor.summary()


class YOLO(object):
    def __init__(self, config):

        self.config = config

        self.image_h = config['model']['image_h']
        self.image_w = config['model']['image_w']

        self.grid_h, self.grid_w = config['model']['grid_h'], config['model']['grid_w']

        self.labels = config['model']['classes']
        self.nb_class = int(len(self.labels))
        self.nb_box = int(len(config['model']['anchors'])/2)
        self.class_wt = np.ones(self.nb_class, dtype='float32')
        self.anchors = config['model']['anchors']

        self.max_box_per_image = config['model']['max_obj']
        self.batch_size = config['train']['batch_size']

        self.object_scale = config['model']['obj_scale']
        self.no_object_scale = config['model']['no_obj_scale']
        self.coord_scale = config['model']['coord_scale']
        self.class_scale = config['model']['class_scale']

        self.obj_thresh = config['model']['obj_thresh']
        self.nms_thresh = config['model']['nms_thresh']

        self.warmup_batches = config['train']['warmup_batches']
        self.debug = config['train']['debug']

        ##########################
        # Make the model
        ##########################

        # make the feature extractor layers
        input_image = Input(shape=(self.image_h, self.image_w, 3))
        self.true_boxes = Input(shape=(1, 1, 1, self.max_box_per_image, 4))

        self.feature_extractor = TinyYoloFeature(self.image_h, config).feature_extractor
        features = self.feature_extractor(input_image)

        # Object detection layer
        output = Conv2D(self.nb_box * (4 + 1 + self.nb_class),
                        (1, 1), strides=(1, 1),
                        padding='same',
                        name='DetectionLayer',
                        activation='linear',
                        kernel_initializer='lecun_normal')(features)

        output = Reshape((self.grid_h, self.grid_w, self.nb_box, 4 + 1 + self.nb_class))(output)
        output = Lambda(lambda args: args[0])([output, self.true_boxes])

        self.model = Model([input_image, self.true_boxes], output)
        self.model.summary()

        pretrained = load_model('checkpoints\\' + config['model']['saved_model_name'], custom_objects={'custom_loss': self.custom_loss, 'tf': tf})
        self.model.get_layer('DetectionLayer').set_weights(
            pretrained.get_layer('DetectionLayer').get_weights())


    def load_weights(self, model_path):
        model = load_model(model_path, custom_objects={'custom_loss': self.custom_loss, 'tf': tf})

        idx = 0
        for layer in self.model.layers:
            layer.set_weights(model.get_layer(index=idx).get_weights())
            idx += 1


    def predict(self, image_path):
        image = load_image_predict(image_path, self.image_h, self.image_w)

        dummy_array = np.zeros((1, 1, 1, 1, self.max_box_per_image, 4))
        netout = self.model.predict([image, dummy_array])[0]

        boxes = decode_netout(netout=netout, anchors = self.anchors, nb_class=self.nb_class,
                              obj_threshold=self.obj_thresh, nms_threshold=self.nms_thresh)
        return boxes


    def train(self):
        data = load_carla_data(os.path.join(ANNOT_DIR, self.config['train']['annot_file_name']), self.config['model']['classes'])

        np.random.shuffle(data)

        train_instances, validation_instances = data[:1655], data[1655:]

        np.random.shuffle(train_instances)
        np.random.shuffle(validation_instances)

        train_generator = BatchGenerator(self.config, train_instances, jitter=True)
        validation_generator = BatchGenerator(self.config, validation_instances, jitter=False)

        checkpoint = ModelCheckpoint(
            'checkpoints\\model.{epoch:02d}-{val_loss:.2f}.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='auto',
            period=1
        )

        checkpoint_all = ModelCheckpoint(
            'checkpoints\\all_models.{epoch:02d}-{loss:.2f}.h5',
            monitor='loss',
            verbose=1,
            save_best_only=True,
            mode='auto',
            period=1
        )

        # optimizer = RMSprop(lr=1e-3,rho=0.9, epsilon=1e-08, decay=0.0)
        # optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        optimizer = SGD(lr=1e-5, momentum=0.9, decay=0.0005)

        self.model.compile(loss=self.custom_loss, optimizer=optimizer)  #, metrics=['accuracy'])

        self.model.summary()

        history = self.model.fit_generator(generator=train_generator,
                                      steps_per_epoch=len(train_generator),
                                      epochs=self.config['train']['nb_epochs'],
                                      verbose=1,
                                      validation_data=validation_generator,
                                      validation_steps=len(validation_generator),
                                      callbacks=[checkpoint, checkpoint_all],# map_evaluator_cb],  # checkpoint, tensorboard
                                      max_queue_size=10,
                                      workers=3
                                      )


    def evaluate(self):
        data = load_carla_data(os.path.join(ANNOT_DIR, self.config['train']['annot_file_name']),
                               self.config['model']['classes'])

        np.random.shuffle(data)

        validation_instances = data#[1400:]

        validation_generator = BatchGenerator(self.config, validation_instances, jitter=False)

        map_evaluator_cb = self.MAP_evaluation(self, validation_generator,
                                               save_best=True,
                                               save_name='checkpoints\\best-mAP.h5',
                                               # os.path.join(BASE_DIR,'best_mAP\\weights.{epoch:02d}-{val_loss:.2f}.h5'),
                                               tensorboard=None,
                                               iou_threshold=0.4)

        self.model.compile(loss=self.custom_loss, optimizer=SGD(lr=1e-5, momentum=0.9, decay=0.0005))

        self.model.summary()

        history = self.model.fit_generator(generator=validation_generator,
                                           steps_per_epoch=len(validation_generator),
                                           epochs=1,
                                           verbose=1,
                                           callbacks=[map_evaluator_cb]
                                           )


    def normalize(self, image):
        return image / 255.


    def custom_loss(self, y_true, y_pred):
        mask_shape = tf.shape(y_true)[:4]

        cell_x = tf.to_float(
            tf.reshape(tf.tile(tf.range(self.grid_w), [self.grid_h]), (1, self.grid_h, self.grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

        cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [self.batch_size, 1, 1, self.nb_box, 1])

        coord_mask = tf.zeros(mask_shape)
        conf_mask = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)

        seen = tf.Variable(0.)
        total_loss = tf.Variable(0.)
        total_recall = tf.Variable(0.)
        total_boxes = tf.Variable(self.grid_h*self.grid_w*self.nb_box*self.batch_size)

        """
        Adjust prediction
        """
        ### adjust x and y
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

        ### adjust w and h tf.exp(
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(self.anchors, [1, 1, 1, self.nb_box, 2])

        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])

        ### adjust class probabilities
        pred_box_class = y_pred[..., 5:]

        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., 0:2]  # relative position to the containing cell

        ### adjust w and h
        true_box_wh = y_true[..., 2:4]  # number of cells accross, horizontally and vertically

        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins = true_box_xy - true_wh_half
        true_maxes = true_box_xy + true_wh_half

        pred_wh_half = pred_box_wh / 2.
        pred_mins = pred_box_xy - pred_wh_half
        pred_maxes = pred_box_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        true_box_conf = iou_scores * y_true[..., 4]

        ### adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 5:], -1)

        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.coord_scale

        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = self.true_boxes[..., 0:2]
        true_wh = self.true_boxes[..., 2:4]

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        #conf_mask = conf_mask + tf.to_float(best_ious < 0.5) * (1 - y_true[..., 4]) * self.no_object_scale

        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        #conf_mask = conf_mask + y_true[..., 4] * self.object_scale

        conf_mask_neg = tf.to_float(best_ious < 0.50) * (1 - y_true[..., 4]) * self.no_object_scale
        conf_mask_pos = y_true[..., 4] * self.object_scale

        ### class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * tf.gather(self.class_wt, true_box_class) * self.class_scale

        """
        Warm-up training
        """
        no_boxes_mask = tf.to_float(coord_mask < self.coord_scale / 2.)
        seen = tf.assign_add(seen, 1.)

        true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, self.warmup_batches + 1),
                                                       lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
                                                                true_box_wh + tf.ones_like(true_box_wh) * \
                                                                np.reshape(self.anchors, [1, 1, 1, self.nb_box, 2]) * \
                                                                no_boxes_mask,
                                                                tf.ones_like(coord_mask)],
                                                       lambda: [true_box_xy,
                                                                true_box_wh,
                                                                coord_mask])

        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        #nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
        nb_conf_box_neg = tf.reduce_sum(tf.to_float(conf_mask_neg > 0.0))
        nb_conf_box_pos = tf.subtract(tf.to_float(total_boxes), nb_conf_box_neg) #tf.reduce_sum(tf.to_float(conf_mask_pos > 0.0))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

        true_box_wh = tf.sqrt(true_box_wh)
        pred_box_wh = tf.sqrt(pred_box_wh)

        loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf_neg = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask_neg) / (nb_conf_box_neg + 1e-6) / 2.
        loss_conf_pos = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask_pos) / (nb_conf_box_pos + 1e-6) / 2
        loss_conf = loss_conf_neg + loss_conf_pos
        #loss_conf = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

        loss = tf.cond(tf.less(seen, self.warmup_batches + 1),
                       lambda: loss_xy + loss_wh + loss_conf + loss_class + 10,
                       lambda: loss_xy + loss_wh + loss_conf + loss_class)

        if self.debug:
            nb_true_box = tf.reduce_sum(y_true[..., 4])
            nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.32) * tf.to_float(pred_box_conf > 0.3))

            current_recall = nb_pred_box / (nb_true_box + 1e-6)
            total_recall = tf.assign_add(total_recall, current_recall)

            total_loss = tf.assign_add(total_loss, loss)

            loss = tf.Print(loss, [loss_xy], message='\nLoss XY \t', summarize=1000)
            loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
            loss = tf.Print(loss, [nb_conf_box_neg], message='Nb Conf Box Negative \t', summarize=1000)
            loss = tf.Print(loss, [nb_conf_box_pos], message='Nb Conf Box Positive \t', summarize=1000)
            loss = tf.Print(loss, [loss_conf_neg], message='Loss Conf Negative \t', summarize=1000)
            loss = tf.Print(loss, [loss_conf_pos], message='Loss Conf Positive \t', summarize=1000)
            loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
            loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
            loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
            loss = tf.Print(loss, [total_loss / seen], message='Average Loss \t', summarize=1000)
            loss = tf.Print(loss, [nb_pred_box], message='Number of pred boxes \t', summarize=1000)
            loss = tf.Print(loss, [nb_true_box], message='Number of true boxes \t', summarize=1000)
            loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
            loss = tf.Print(loss, [total_recall / seen], message='Average Recall \t', summarize=1000)


        return loss


    class MAP_evaluation(Callback):
        """ Evaluate a given dataset using a given model.
            code originally from https://github.com/fizyr/keras-retinanet
            # Arguments
                generator       : The generator that represents the dataset to evaluate.
                model           : The model to evaluate.
                iou_threshold   : The threshold used to consider when a detection is positive or negative.
                score_threshold : The score confidence threshold to use for detections.
                save_path       : The path to save images with visualized detections to.
            # Returns
                A dict mapping class names to mAP scores.
        """

        def __init__(self,
                     yolo,
                     generator,
                     iou_threshold=0.5,
                     score_threshold=0.3,
                     save_path=None,
                     period=1,
                     save_best=False,
                     save_name=None,
                     tensorboard=None):

            self.yolo = yolo
            self.generator = generator
            self.iou_threshold = iou_threshold
            self.save_path = save_path
            self.period = period
            self.save_best = save_best
            self.save_name = save_name
            self.tensorboard = tensorboard

            self.bestMap = 0

            self.model = self.yolo.model

            if not isinstance(self.tensorboard, TensorBoard) and self.tensorboard is not None:
                raise ValueError("Tensorboard object must be a instance from keras.callbacks.TensorBoard")


        def on_epoch_end(self, epoch, logs={}):
            print(epoch)
            #% self.period == 0 and self.period != 0:
            mAP, average_precisions = self.evaluate_mAP()
            print('\n')
            for label, average_precision in average_precisions.items():
                print(self.yolo.labels[label], '{:.4f}'.format(average_precision))
            print('mAP: {:.4f}'.format(mAP))

            if self.save_best and self.save_name is not None and mAP > self.bestMap:
                print(
                    "mAP improved from {} to {}, saving model to {}.".format(self.bestMap, mAP, self.save_name))
                self.bestMap = mAP
                print(self.save_name)
                self.model.save(self.save_name)
                self.model.save_weights('checkpoints\\best-mAP.h5')
            else:
                print("mAP did not improve from {}.".format(self.bestMap))

            if self.tensorboard is not None and self.tensorboard.writer is not None:
                import tensorflow as tf
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = mAP
                summary_value.tag = "val_mAP"
                self.tensorboard.writer.add_summary(summary, epoch)


        def evaluate_mAP(self):
            average_precisions = self._calc_avg_precisions()
            mAP = sum(average_precisions.values()) / len(average_precisions)

            return mAP, average_precisions


        def _calc_avg_precisions(self):
            # gather all detections and annotations
            all_detections = [[None for i in range(self.generator.num_classes())] for j in
                              range(self.generator.size())]
            all_annotations = [[None for i in range(self.generator.num_classes())] for j in
                               range(self.generator.size())]

            for i in range(self.generator.size()):
                raw_image = self.generator.load_image(self.generator.dataset[i]['image_path'])
                raw_height, raw_width, _ = raw_image.shape
                # make the boxes and the labels
                pred_boxes = self.yolo.predict(os.path.join(ANNOT_DIR, 'images', self.generator.dataset[i]['image_path']))

                score = np.array([box.score for box in pred_boxes])
                pred_labels = np.array([box.label for box in pred_boxes])

                if len(pred_boxes) > 0:
                    pred_boxes = np.array([[box.xmin * raw_width, box.ymin * raw_height, box.xmax * raw_width,
                                            box.ymax * raw_height, box.score] for box in pred_boxes])
                else:
                    pred_boxes = np.array([[]])

                    # sort the boxes and the labels according to scores
                score_sort = np.argsort(-score)
                pred_labels = pred_labels[score_sort]
                pred_boxes = pred_boxes[score_sort]

                # copy detections to all_detections
                for label in range(self.generator.num_classes()):
                    all_detections[i][label] = pred_boxes[pred_labels == label, :]

                annotations = self.generator.load_annotation(i)

                # copy detections to all_annotations
                for label in range(self.generator.num_classes()):
                    all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

            # compute mAP by comparing all detections and all annotations
            average_precisions = {}

            for label in range(self.generator.num_classes()):
                false_positives = np.zeros((0,))
                true_positives = np.zeros((0,))
                scores = np.zeros((0,))
                num_annotations = 0.0

                for i in range(self.generator.size()):
                    detections = all_detections[i][label]
                    annotations = all_annotations[i][label]
                    num_annotations += annotations.shape[0]
                    detected_annotations = []

                    for d in detections:
                        scores = np.append(scores, d[4])

                        if annotations.shape[0] == 0:
                            false_positives = np.append(false_positives, 1)
                            true_positives = np.append(true_positives, 0)
                            continue

                        overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                        assigned_annotation = np.argmax(overlaps, axis=1)
                        max_overlap = overlaps[0, assigned_annotation]

                        if max_overlap >= self.iou_threshold and assigned_annotation not in detected_annotations:
                            false_positives = np.append(false_positives, 0)
                            true_positives = np.append(true_positives, 1)
                            detected_annotations.append(assigned_annotation)
                        else:
                            false_positives = np.append(false_positives, 1)
                            true_positives = np.append(true_positives, 0)

                # no annotations -> AP for this class is 0 (is this correct?)
                if num_annotations == 0:
                    average_precisions[label] = 0
                    continue

                # sort by score
                indices = np.argsort(-scores)
                false_positives = false_positives[indices]
                true_positives = true_positives[indices]

                # compute false positives and true positives
                false_positives = np.cumsum(false_positives)
                true_positives = np.cumsum(true_positives)

                # compute recall and precision
                recall = true_positives / num_annotations
                precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

                # compute average precision
                average_precision = compute_ap(recall, precision)
                average_precisions[label] = average_precision

            return average_precisions


def dummy_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(y_pred))