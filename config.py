
#MODEL_PATH_JSON = r"C:\Users\Lukas\Documents\Object detection 2.0\model\my_net_json\object_detection_model.json"
MODEL_WEIGHTS = r"C:\Users\Lukas\Documents\Object detection 2.0\model\model_weights\model_weights.h5"
MODEL_PATH_H5 = r"C:\Users\Lukas\Documents\Object detection 2.0\model\model\model.h5"
#MODEL_PATH_PB = r"C:\Users\Lukas\Documents\Object detection 2.0\model\my_net_pb\object_detection_model"

# save model every x epochs
# model weights and json configuration will be saved
SAVE_MODEL_EVERY = 10
UPDATE_LEARNING_RATE = [50, 100, 150]


IMAGE_PATH = r'D:\Documents\KITTI\images\training\image'
CALIB_PATH = r'D:\Documents\KITTI\calib\training\calib'
LABEL_PATH = r'D:\Documents\KITTI\label\training\label'
BB3_FOLDER = r'D:\Documents\KITTI\bb3_files'
PGP_FOLDER = r'D:\Documents\KITTI\pgp_file'

#percent
TRAINING_SPLIT = 70
RADIUS = 2
CIRCLE_RATIO = 0.3
BOUNDARIES = 0.25

# amount of data from dataset which will be loaded
# -1 for all data

IMAGE_EXTENSION = 'png'


#GROUND PLANE EXTRACTOR

RANSAC_ITERATIONS = 10000


# NETWORK RELATED SETTINGS
IS_TRAINING = True
IMG_WIDTH = 256
IMG_HEIGHT = 128
IMG_CHANNELS = 3 # based on channel number, image will be loaded colored or grayscaled

DEVICE_NAME = "/gpu:0"
DATA_AMOUNT = 100
BATCH_SIZE = 1
LEARNING_RATE = 0.001
<<<<<<< HEAD
<<<<<<< HEAD
BATCH_SIZE = 8
MAX_ERROR = 0.0001
ITERATIONS = 500
=======
BATCH_SIZE = 4
MAX_ERROR = 0.001
ITERATIONS = 100
>>>>>>> 4c3411ae0b0e9084006a2522612b6d7783c6ce76
=======
MAX_ERROR = 0.001
UPDATE_EDGE = 0.01
ITERATIONS = 50
WEIGHT_FACTOR = 2.0

# NON-MAXIMA SUPPRESSION

NMS_TRESHOLD = 200
RESULT_TRESHOLD = 60

