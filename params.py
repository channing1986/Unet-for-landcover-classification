__author__ = 'Changlin'
__version__ = 0.1

import os

# ----------------------------------------------------
# DATA I/O

# TRAIN_DIR = 'G:/DataSet/GRSS2019Contest/Track1-RGB/'
# LABEL_DIR = 'G:/DataSet/GRSS2019Contest/Train-Track1-Truth/Track1-Truth/'
# TRAIN_DIR = 'G:/DataSet/GRSS2019Contest/Track1-RGB_pathces_512/img_patch/'
# LABEL_DIR = 'G:/DataSet/GRSS2019Contest/Track1-RGB_pathces_512/label_patch/'
CHECKPOINT_DIR = './checkpoint/'
# TEST_DIR = 'G:/DataSet/GRSS2019Contest/Validate-Track1/Track1/'
OUTPUT_DIR = './prediction/'
# OUTPUT_DIR_TRACK3='../data/validate/Track3-Submission-4_weightedPatchMerge/'
# SINGLEVIEW_TEST_MODEL = '../weights/181206-unet-height-weights.60.hdf5'
# #SEMANTIC_TEST_MODEL = '../weights/181219-unet-semantic-weights.40.hdf5'
# SEMANTIC_TEST_MODEL = '../weights/190214-unet256-semantic-weight.80.hdf5'
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# if not os.path.isdir(OUTPUT_DIR_TRACK3):
#     os.makedirs(OUTPUT_DIR_TRACK3)

if not os.path.isdir(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# CONTINUE_TRAINING = True
# CONTINUE_SEMANTIC_MODEL_FILE = SEMANTIC_TEST_MODEL
# CONTINUE_SINGLEVIEW_MODEL_FILE = SINGLEVIEW_TEST_MODEL


CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'weights.{epoch:02d}.hdf5')

CLASS_FILE_STR = '_CLS'
DEPTH_FILE_STR = '_AGL'
IMG_FILE_STR = '_RGB'
IMG_FILE_EXT = 'tif'
LABEL_FILE_EXT = IMG_FILE_EXT

CLSPRED_FILE_STR = '_CLS'
AGLPRED_FILE_STR = '_AGL'

# for MSI image training
MEAN_VALS_FILE = './data/msi_mean.json'
MAX_VAL = 65536  # MSI images are int16, so dividing by this instead of 255

# ----------------------------------------------------
# MODEL TRAINING/TESTING

GPUS = '0'  # GPU indices to restrict usage
NUM_CHANNELS = 3
BATCH_SZ = 6
#IMG_SZ = (1024, 1024)  # this code assumes all images in the training set have the same numbers of rows and columns
PATCH_SZ = [512, 512,3]
IGNORE_VALUE = -10000  # nan is set to this for ignore purposes later
NUM_CATEGORIES = 5  # for semantic segmentation
MODEL_SAVE_PERIOD = 1  # how many epochs between model checkpoint saves
NUM_EPOCHS = 200  # total number of epochs to train with
BINARY_CONF_TH = 0.4

BLOCK_IMAGES = False
BLOCK_SZ = (1024, 1024)
BLOCK_MIN_OVERLAP = 20

OPTIMIZER = 'Adam'
SEMANTIC_LOSS = 'categorical_crossentropy' 
LEARN_RATE=1E-4

BACKBONE = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'

# ----------------------------------------------------
# FLAGS

SEMANTIC_MODE = 0
SINGLEVIEW_MODE = 1

TRAIN_MODE = 0
TEST_MODE = 1

# ----------------------------------------------------
# LABEL MANIPULATION

CONVERT_LABELS = False

LAS_LABEL_GROUND = 2
LAS_LABEL_TREES = 5
LAS_LABEL_ROOF = 6
LAS_LABEL_WATER = 9
LAS_LABEL_BRIDGE_ELEVATED_ROAD = 17
LAS_LABEL_VOID = 65

TRAIN_LABEL_GROUND = 0
TRAIN_LABEL_TREES = 1
TRAIN_LABEL_BUILDING = 2
TRAIN_LABEL_WATER = 3
TRAIN_LABEL_BRIDGE_ELEVATED_ROAD = 4
TRAIN_LABEL_VOID = NUM_CATEGORIES

LABEL_MAPPING_LAS2TRAIN = {}
LABEL_MAPPING_LAS2TRAIN[LAS_LABEL_GROUND] = TRAIN_LABEL_GROUND
LABEL_MAPPING_LAS2TRAIN[LAS_LABEL_TREES] = TRAIN_LABEL_TREES
LABEL_MAPPING_LAS2TRAIN[LAS_LABEL_ROOF] = TRAIN_LABEL_BUILDING
LABEL_MAPPING_LAS2TRAIN[LAS_LABEL_WATER] = TRAIN_LABEL_WATER
LABEL_MAPPING_LAS2TRAIN[LAS_LABEL_BRIDGE_ELEVATED_ROAD] = TRAIN_LABEL_BRIDGE_ELEVATED_ROAD
LABEL_MAPPING_LAS2TRAIN[LAS_LABEL_VOID] = TRAIN_LABEL_VOID

LABEL_MAPPING_TRAIN2LAS = {}
LABEL_MAPPING_TRAIN2LAS[TRAIN_LABEL_GROUND] = LAS_LABEL_GROUND
LABEL_MAPPING_TRAIN2LAS[TRAIN_LABEL_TREES] = LAS_LABEL_TREES
LABEL_MAPPING_TRAIN2LAS[TRAIN_LABEL_BUILDING] = LAS_LABEL_ROOF
LABEL_MAPPING_TRAIN2LAS[TRAIN_LABEL_WATER] = LAS_LABEL_WATER
LABEL_MAPPING_TRAIN2LAS[TRAIN_LABEL_BRIDGE_ELEVATED_ROAD] = LAS_LABEL_BRIDGE_ELEVATED_ROAD
