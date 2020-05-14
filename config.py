__date__   = '14/05/2020'
__author__ = 'Lukas Mrvecka'
__email__  = 'lukas.mrvecka.st@vsb.cz'

IMAGE_PATH = r'D:\Documents\KITTI\images\training\image'
CALIB_PATH = r'D:\Documents\KITTI\calib\training\calib'
LABEL_PATH = r'D:\Documents\KITTI\label\training\label'
BB3_FOLDER = r'D:\Documents\KITTI\bb3_files'

# ground truth specification
RADIUS = 2
CIRCLE_RATIO = 0.3
BOUNDARIES = 0.25

# amount of data from dataset which will be loaded
# -1 for all data
IMAGE_EXTENSION = 'png'

#GROUND PLANE EXTRACTOR
RANSAC_ITERATIONS = 10000

# only images with specific extension will be loaded
IMAGE_EXTENSION = 'png'
# images are scaled to fixed size for the network
IMG_WIDTH = 256
IMG_HEIGHT = 128
IMG_CHANNELS = 3 # based on channel number, image will be loaded colored or grayscaled



# ------------  TRAINING ------------

# save model every x epochs
# model weights and json configuration will be saved
SAVE_MODEL_EVERY = 10
# update learning rate on this epoch, 
# current learning rate is divided by 10
UPDATE_LEARNING_RATE = [100, 200, 300, 800]

# type of optimizer, adam (default) or sgd
OPTIMIZER = "adam"

# update learning rate when one of losses goes below specified value
# -1 for ignore settings
# if updating, learning rate and update edge is divided by 10
# example UPDATE_EDGE = 0.01 
UPDATE_EDGE = -1


# determine the size of dataset
# -1 for whole dataset
DATA_AMOUNT = 20

# specify data which will be used for training or testing !!!! only one file
# if not empty, DATA_AMOUNT is ignored and BATCH_SIZE = 1
# if empty, DATA_AMOUNT and BATCH_SIZE are used
# example: SPECIFIC_DATA = "000008"
SPECIFIC_DATA = ""

# NETWORK RELATED SETTINGS
USE_GPU = True
# specify gpu name which should be used
# if empty program find gpu and use it if available
# if not empty, program check if device exists and use it or find another one if exists
DEVICE_NAME = ""
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
MAX_ERROR = 0.001

# number of iterations per epoch
ITERATIONS = 100
WEIGHT_FACTOR = 2.0




# ------------  TESTING ------------

# during testing this property MUST be set, this is the image i want to test 
SPECIFIC_TEST_DATA = "000008"

# pixels with probability lower than specified value will be set to 0
RESULT_TRESHOLD = 60

