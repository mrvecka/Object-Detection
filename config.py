BASE_PATH = r'D:\Documents\KITTI'


# optional: if not specified program look for base dir + \[images,calib,label] + \training + \[image,calib,label]
IMAGE_PATH = r'D:\Documents\KITTI\images\training\image'
CALIB_PATH = r'D:\Documents\KITTI\calib\training\calib'
LABEL_PATH = r'D:\Documents\KITTI\label\training\label'
BB3_FOLDER = r'bb3_files'

#percent
TRAINING_SPLIT = 70

# amount of data from dataset which will be loaded
START_FROM = 0
DATA_AMOUNT = 400

CONTINUOUS_LOADER = True

IMAGE_EXTENSION = 'png'


#GROUND PLANE EXTRACTOR

RANSAC_ITERATIONS = 10000


# NETWORK RELATED SETTINGS
IS_TRAINING = True
IMG_WIDTH = 256
IMG_HEIGHT = 128
IMG_CHANNELS = 3 # based on channel number, image will be loaded colored or grayscaled

IMG_ORIG_WIDTH = 0
IMG_ORIG_HEIGHT = 0

DEVICE_NAME = "/gpu:0"
LEARNING_RATE = 0.001
BATCH_SIZE = 8
EPOCHS = 5
ITERATIONS = 10

# NON-MAXIMA SUPPRESSION

NMS_TRESHOLD = 200

