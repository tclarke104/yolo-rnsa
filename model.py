from keras.models import Model
from keras.layers import Reshape, Conv2D, Input, MaxPooling2D, BatchNormalization, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from preprocessing import BatchGenerator
import random
import numpy as np
from preprocessing import get_test_image

LABELS = ['PN']

# the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)


class Yolo:

    def __init__(self, config, debug=True):
        self.debug = debug
        self.config = config
        self.input_image = Input(shape=(self.config['IMAGE_H'], self.config['IMAGE_W'], 3))
        self.true_boxes = Input(shape=(1, 1, 1, self.config['TRUE_BOX_BUFFER'], 4))
        self.model = self.initialize_model()

        self.image_h = self.config['IMAGE_H']
        self.image_w = self.config['IMAGE_W']
        self.grid_h = self.config['GRID_H']
        self.grid_w = self.config['GRID_W']
        self.nb_box = self.config['BOX']
        self.num_class = self.config['CLASS']
        self.class_wt = self.config['CLASS_WEIGHTS']
        self.obj_threshold = self.config['OBJ_THRESHOLD']
        self.nms_threshold = self.config['NMS_THRESHOLD']
        self.anchors = self.config['ANCHORS']
        self.no_object_scale = self.config['NO_OBJECT_SCALE']
        self.object_scale = self.config['OBJECT_SCALE']
        self.coord_scale = self.config['COORD_SCALE']
        self.class_scale = self.config['CLASS_SCALE']
        self.batch_size = self.config['BATCH_SIZE']
        self.warmup_batches = self.config['WARM_UP_BATCHES']
        self.true_box_buffer = self.config['TRUE_BOX_BUFFER']
        self.orig_size = self.config['ORIG_SIZE']

    def space_to_depth_x2(self, x):
        return tf.space_to_depth(x, block_size=2)

    def initialize_model(self):

        # Layer 1
        x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(self.input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False)(x)
        x = BatchNormalization(name='norm_2')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 3
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False)(x)
        x = BatchNormalization(name='norm_3')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 4
        x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4', use_bias=False)(x)
        x = BatchNormalization(name='norm_4')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 5
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False)(x)
        x = BatchNormalization(name='norm_5')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 7
        x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7', use_bias=False)(x)
        x = BatchNormalization(name='norm_7')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 8
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8', use_bias=False)(x)
        x = BatchNormalization(name='norm_8')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 9
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9', use_bias=False)(x)
        x = BatchNormalization(name='norm_9')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 10
        x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10', use_bias=False)(x)
        x = BatchNormalization(name='norm_10')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 11
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11', use_bias=False)(x)
        x = BatchNormalization(name='norm_11')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 12
        x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12', use_bias=False)(x)
        x = BatchNormalization(name='norm_12')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 13
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13', use_bias=False)(x)
        x = BatchNormalization(name='norm_13')(x)
        x = LeakyReLU(alpha=0.1)(x)

        skip_connection = x

        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 14
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14', use_bias=False)(x)
        x = BatchNormalization(name='norm_14')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 15
        x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15', use_bias=False)(x)
        x = BatchNormalization(name='norm_15')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 16
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16', use_bias=False)(x)
        x = BatchNormalization(name='norm_16')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 17
        x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17', use_bias=False)(x)
        x = BatchNormalization(name='norm_17')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 18
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18', use_bias=False)(x)
        x = BatchNormalization(name='norm_18')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 19
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19', use_bias=False)(x)
        x = BatchNormalization(name='norm_19')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 20
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20', use_bias=False)(x)
        x = BatchNormalization(name='norm_20')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 21
        skip_connection = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_21', use_bias=False)(
            skip_connection)
        skip_connection = BatchNormalization(name='norm_21')(skip_connection)
        skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
        skip_connection = Lambda(self.space_to_depth_x2)(skip_connection)

        x = concatenate([skip_connection, x])

        # Layer 22
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22', use_bias=False)(x)
        x = BatchNormalization(name='norm_22')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 23
        x = Conv2D(self.config['BOX'] * (4 + 1 + self.config['CLASS']),
                   (1, 1), strides=(1, 1),
                   padding='same',
                   name='conv_23')(x)
        output = Reshape((self.config['GRID_H'],
                          self.config['GRID_W'],
                          self.config['BOX'],
                          4 + 1 + self.config['CLASS']))(x)

        # small hack to allow true_boxes to be registered when Keras build the model
        # for more information: https://github.com/fchollet/keras/issues/2790
        # output = Lambda(lambda args: args[0])([output, true_boxes])

        output = Lambda(lambda args: args[0])([output, self.true_boxes])

        model = Model([self.input_image, self.true_boxes], output)

        return model

    def train(self,
              image_fps,
              image_annotations,
              number_of_images_to_use=20,
              use_all_imgs=False):

        if use_all_imgs:
            image_fps_list = list(image_fps)
        else:
            image_fps_list = list(image_fps[:number_of_images_to_use])

        # split dataset into training vs. validation dataset
        # split ratio is set to 0.9 vs. 0.1 (train vs. validation, respectively)
        sorted(image_fps_list)
        random.seed(42)
        random.shuffle(image_fps_list)

        validation_split = 0.1
        split_index = int((1 - validation_split) * len(image_fps_list))

        image_fps_train = image_fps_list[:split_index]
        image_fps_val = image_fps_list[split_index:]
        image_annotations_train = {k: v for k, v in image_annotations.items() if k in image_fps_train}
        image_annotations_val = {k: v for k, v in image_annotations.items() if k in image_fps_val}

        train_generator = BatchGenerator(self.config, image_fps_train, image_annotations_train)
        test_generator = BatchGenerator(self.config, image_fps_val, image_annotations_val)

        optimizer = Adam(lr=0.5e-8, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        # optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
        # optimizer = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08, decay=0.0)

        file_path = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

        self.model.compile(loss=self.custom_loss, optimizer=optimizer, metrics=['accuracy'])

        checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        self.model.fit_generator(train_generator,
                                 steps_per_epoch=len(train_generator),
                                 epochs=10000,
                                 verbose=1,
                                 validation_data=test_generator,
                                 validation_steps=len(test_generator),
                                 callbacks=[checkpoint])

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
        total_recall = tf.Variable(0.)

        """
        Adjust prediction
        """
        ### adjust x and y
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

        ### adjust w and h
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
        conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * self.no_object_scale

        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., 4] * self.object_scale

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
        nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

        loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

        loss = tf.cond(tf.less(seen, self.warmup_batches + 1),
                       lambda: loss_xy + loss_wh + loss_conf + loss_class + 10,
                       lambda: loss_xy + loss_wh + loss_conf + loss_class)

        if self.debug:
            nb_true_box = tf.reduce_sum(y_true[..., 4])
            nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))

            current_recall = nb_pred_box / (nb_true_box + 1e-6)
            total_recall = tf.assign_add(total_recall, current_recall)

            loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
            loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
            loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
            loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
            loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
            loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
            loss = tf.Print(loss, [total_recall / seen], message='Average Recall \t', summarize=1000)

        return loss

    def predict(self):
        test_img = get_test_image(image_fp)

        test_img = resize(test_img, (416, 416, 3))
        test_img = test_img.reshape((1, 416, 416, 3))

        out = model.predict([test_img, dummy_array])

        boxes = decode_netout(out, ANCHORS, CLASS, 0.1)
        print(out[0])