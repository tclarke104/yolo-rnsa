import glob
import os
import pydicom
import numpy as np
from skimage.transform import resize
from utils import BoundBox, bbox_iou
import cv2
from keras.utils import Sequence


def get_dicom_fps(dicom_dir):
    dicom_fps = glob.glob(dicom_dir + '/' + '*.dcm')
    return list(set(dicom_fps))


def parse_dataset(dicom_dir, anns):
    image_fps = get_dicom_fps(dicom_dir)
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows():
        fp = os.path.join(dicom_dir, row['patientId'] + '.dcm')
        image_annotations[fp].append(row)
    return image_fps, image_annotations


def get_test_image(image_fp):
    ds = pydicom.read_file(image_fp)
    image = ds.pixel_array

    if len(image.shape) != 3 or image.shape[2] != 3:
        image = np.stack((image,) * 3, -1)
    return image


def get_image_and_annotation(patient_index, image_fps, image_annotations):
    # Load image from file path
    ds = pydicom.read_file(image_fps[patient_index])
    image = ds.pixel_array

    # Convert image from grayscale to RGB
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = np.stack((image,) * 3, -1)

    # Get image annotations
    annotation = image_annotations[image_fps[patient_index]]

    return image, annotation


def convert_annotation_to_mask(annotation, orig_size=1024):
    # Original DICOM image size: 1024 x 1024
    count = len(annotation)
    if count == 0:
        mask = np.zeros((orig_size, orig_size, 1), dtype=np.uint8)
        class_ids = np.zeros((1,), dtype=np.int32)
    else:
        mask = np.zeros((orig_size, orig_size, count), dtype=np.uint8)
        class_ids = np.zeros((count,), dtype=np.int32)
        for i, a in enumerate(annotation):
            if a['Target'] == 1:
                x = int(a['x'])
                y = int(a['y'])
                w = int(a['width'])
                h = int(a['height'])
                mask_instance = mask[:, :, i].copy()
                cv2.rectangle(mask_instance, (x, y), (x + w, y + h), 255, -1)
                mask[:, :, i] = mask_instance
                class_ids[i] = 1
    return mask.astype(np.bool), class_ids.astype(np.int32)


class BatchGenerator(Sequence):
    def __init__(self, config,
                 image_fps,
                 annotations,
                 shuffle=True,
                 jitter=True,
                 norm=None):
        '''Creates a generator that supplies training and validation sets in discrete batches
        config: a dictionary of constants that tells properties of image
        image_fps: a list of file paths to each of the training images
        annotations: a dictionary contain the labels for each image. Indexed by file path I believe
        shuffle: bool for whether we should shuffle between epochs
        '''
        self.generator = None

        self.image_fps = image_fps
        self.config = config

        self.shuffle = shuffle
        self.jitter = jitter
        self.norm = norm
        self.ORIG_SIZE = 1024
        self.image_annotations = annotations

        self.anchors = [BoundBox(0, 0, config['ANCHORS'][2 * i], config['ANCHORS'][2 * i + 1]) for i in
                        range(int(len(config['ANCHORS']) // 2))]

        if shuffle: np.random.shuffle(self.image_fps)

    def __len__(self):
        # handles the len() function so keras can makes something of it
        return int(np.ceil(float(len(self.image_fps)) / self.config['BATCH_SIZE']))

    def size(self):
        return len(self.image_fps)

    def num_classes(self):
        return len(self.config['LABELS'])

    def __getitem__(self, idx):
        '''allows for us to index into our generator using [] notation'''

        # calculate the indeces of pictures that we will select from.
        l_bound = idx * self.config['BATCH_SIZE']
        r_bound = (idx + 1) * self.config['BATCH_SIZE']

        instance_count = 0

        # initialize the output
        # x_batch is the batch of images of size (batch_size, image_width, image_height, num_channels)
        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_W'], self.config['IMAGE_H'], 3))
        # not entirely sure what b_batch is right now. Possibly broken
        b_batch = np.zeros((r_bound - l_bound, 1, 1, 1, self.config['TRUE_BOX_BUFFER'], 4))
        # y_batch is training labels of size (batch_size, grid_width, grid_height, num_anchors, num_labels)
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_W'], self.config['GRID_H'], len(self.anchors),
                            1 + 4 + self.config['CLASS']))

        # check to make sure that the batch_size isn't greater than the remaining number of pics
        if r_bound > len(self.image_fps):
            r_bound = len(self.image_fps)
            l_bound = r_bound - self.config['BATCH_SIZE']
            if l_bound < 0:
                l_bound = 0

        # iterate over every sample in our batch range
        for index, example in enumerate(range(l_bound, r_bound)):
            true_box_index = 0

            # load images and samples based on index
            image, annotations = self.load_image_and_annotations(example)

            # check to see if the current images has any real bounding boxes
            has_box = annotations[0]['Target']

            if has_box:
                # create list of class BoundingBox for each annotation of the image
                bboxes = self.create_bboxes(annotations)
                # creates the 13x13x5x6 label for each image as well as b_batch
                label, b_batch, instance_count = self.convert_bboxes_to_out(bboxes, b_batch, instance_count,
                                                                            true_box_index)
                # don't know what the true boxes and b_batch really are could be broken
                true_box_index += 1
                true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']
            else:
                # if no bboxes in annotations just create 13x13x5x6 grid of zeros
                label = np.zeros(
                    (self.config['GRID_W'], self.config['GRID_H'], len(self.anchors), 1 + 4 + self.config['CLASS']))

            # append labels to training set
            x_batch[index] = image
            y_batch[index] = label
            instance_count += 1

        return [x_batch, b_batch], y_batch

    def load_image_and_annotations(self, patient_index, norm=True):
        '''
        method for loading images from disk and selecting their annotation from
        the annotation dictionary.
        patient_index: int that represents the index of the image in image_fps
        '''

        # Load image from file path
        ds = pydicom.read_file(self.image_fps[patient_index])
        image = ds.pixel_array

        # Convert image from grayscale to RGB
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)

        # if norm:
        # image = np.true_divide(image, 255)

        # Get image annotations from image_annotatiosn dictionary based of file path
        annotation = self.image_annotations[self.image_fps[patient_index]]

        # downsample image to desired size (416,416,3) and return annotation
        # return resize(image, (self.config['IMAGE_W'], self.config['IMAGE_H'], 3)), annotation
        return resize(image, (self.config['IMAGE_W'], self.config['IMAGE_H'], 3)), annotation

    def create_bboxes(self, annotations):
        bboxes = []
        for annotation in annotations:
            annotation.x *= (self.config['IMAGE_W'] / self.config['ORIG_SIZE'])
            annotation.y *= (self.config['IMAGE_H'] / self.config['ORIG_SIZE'])
            annotation.height *= (self.config['IMAGE_W'] / self.config['ORIG_SIZE'])
            annotation.width *= (self.config['IMAGE_H'] / self.config['ORIG_SIZE'])
            bboxes.append(BoundBox(annotation.x,
                                   annotation.y,
                                   annotation.x + annotation.width,
                                   annotation.y + annotation.height))
        return bboxes

    def convert_bboxes_to_out(self, bboxes, b_batch, instance_count, true_box_index):
        # intialize the 13x13x5x6 output label for the current image
        out_label = np.zeros(
            (self.config['GRID_W'], self.config['GRID_H'], len(self.anchors), 1 + 4 + self.config['CLASS']))

        # compute the width and height in pixels of an output grid cell
        width_grid_cell = float(self.config['IMAGE_W'] / self.config['GRID_W'])
        height_grid_cell = float(self.config['IMAGE_H'] / self.config['GRID_H'])

        # iterate over each bbox in our list of bboxes
        for bbox in bboxes:
            # calculate the center pixel of the bbox
            center_x = .5 * (bbox.xmin + bbox.xmax)
            center_y = .5 * (bbox.ymin + bbox.ymax)

            # change center coordinates into percent hxw of corresponding grid cell
            center_x = center_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
            center_y = center_y = center_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])

            # determine which grid cell the center lies in
            grid_x = int(np.floor(center_x) / width_grid_cell)
            grid_y = int(np.floor(center_y) / height_grid_cell)

            if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:

                obj_indx = 0

                # normalize height and width relative to size of grid cell
                center_w = (bbox.xmax - bbox.xmin) / (
                            float(self.config['IMAGE_W']) / self.config['GRID_W'])  # unit: grid cell
                center_h = (bbox.ymax - bbox.ymin) / (
                            float(self.config['IMAGE_H']) / self.config['GRID_H'])  # unit: grid cell

                # save true bbox size
                box = [center_x, center_y, center_w, center_h]

                # find the anchor that best predicts this box
                best_anchor = -1
                max_iou = -1

                # in order to compare to anchor box we make temporary box and
                # have them share same top left coordinate
                shifted_box = BoundBox(0,
                                       0,
                                       center_w,
                                       center_h)

                for i in range(len(self.anchors)):
                    # calcualte IOU of each each anchor and our bbox to determine
                    # which one fits our bbox best
                    anchor = self.anchors[i]
                    iou = bbox_iou(shifted_box, anchor)

                    if max_iou < iou:
                        best_anchor = i
                        max_iou = iou
                # create the label for our anchor box
                label = np.array([center_x, center_y, center_w, center_h, 1., 1])
                # assign the label to the correct anchor
                out_label[grid_x, grid_y, best_anchor] = label
                # add another true_box to the b_batch don't understand really
                b_batch[instance_count, 0, 0, 0, true_box_index] = box

        return out_label, b_batch, instance_count

    def on_epoch_end(self):
        '''allows keras to handle and shuffle images at end of each epoch'''
        if self.shuffle: np.random.shuffle(self.image_fps)
