import os
import pandas as pd
from preprocessing import parse_dataset
import numpy as np
from model import Yolo


LABELS = ['PN']
# Root directory of the project
ROOT_DIR = os.path.abspath('/Users/travisclarke/kaggle-data/')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')

if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR)
os.chdir(ROOT_DIR)

train_dicom_dir = os.path.join(ROOT_DIR, 'stage_1_train_images')
test_dicom_dir = os.path.join(ROOT_DIR, 'stage_1_test_images')

raw_annotations = pd.read_csv(os.path.join(ROOT_DIR, 'stage_1_train_labels.csv'))

image_fps, image_annotations = parse_dataset(train_dicom_dir, anns=raw_annotations)

config = dict(
    IMAGE_H=416,
    IMAGE_W=416,
    GRID_H=13,
    GRID_W=13,
    BOX=5,
    CLASS=len(LABELS),
    CLASS_WEIGHTS=np.ones(len(LABELS), dtype='float32'),
    OBJ_THRESHOLD=0.3,#0.5
    NMS_THRESHOLD=0.3,#0.45
    ANCHORS=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
    NO_OBJECT_SCALE=1.0,
    OBJECT_SCALE=5.0,
    COORD_SCALE=1.0,
    CLASS_SCALE=1.0,
    BATCH_SIZE=2,
    WARM_UP_BATCHES=0,
    TRUE_BOX_BUFFER=50,
    ORIG_SIZE=1024
)

model = Yolo(config, debug=False)

model.train(image_fps, image_annotations, number_of_images_to_use=2)
