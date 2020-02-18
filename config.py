
MODEL_PATH = r"C:\Users\Lukas\Documents\Object detection\model\my_net"
FROZEN_MODEL_PATH = r'C:\Users\Lukas\Documents\Object detection\model\frozen_model.pb'
# optional: if not specified program look for base dir + \[images,calib,label] + \training + \[image,calib,label]
IMAGE_PATH = r'D:\Documents\KITTI\images\training\image'
CALIB_PATH = r'D:\Documents\KITTI\calib\training\calib'
LABEL_PATH = r'D:\Documents\KITTI\label\training\label'
BB3_FOLDER = r'D:\Documents\KITTI\bb3_files'
PGP_FOLDER = r'D:\Documents\KITTI\pgp_file'

#percent
TRAINING_SPLIT = 70

# amount of data from dataset which will be loaded
# -1 for all data
DATA_AMOUNT = 8

IMAGE_EXTENSION = 'png'


#GROUND PLANE EXTRACTOR

RANSAC_ITERATIONS = 10000


# NETWORK RELATED SETTINGS
IS_TRAINING = True
IMG_WIDTH = 256
IMG_HEIGHT = 128
IMG_CHANNELS = 3 # based on channel number, image will be loaded colored or grayscaled

DEVICE_NAME = "/gpu:0"
LEARNING_RATE = 0.00001
BATCH_SIZE = 8
MAX_ERROR = 0.001
ITERATIONS = 100
WEIGHT_FACTOR = 2.0

# NON-MAXIMA SUPPRESSION

NMS_TRESHOLD = 200
RESULT_TRESHOLD = 60

