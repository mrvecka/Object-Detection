import tensorflow as tf
import cv2
import numpy as np
import os
import glob
import random
import time
import datetime



# IMAGE_PATH = 'drive/My Drive/colab_data/images'
# CALIB_PATH = 'drive/My Drive/colab_data/calib'
# LABEL_PATH = 'drive/My Drive/colab_data/label'
# BB3_FOLDER = 'drive/My Drive/colab_data/bb3'

IMAGE_PATH = r'D:\Documents\KITTI\images\training\image'
CALIB_PATH = r'D:\Documents\KITTI\calib\training\calib'
LABEL_PATH = r'D:\Documents\KITTI\label\training\label'
BB3_FOLDER = r'D:\Documents\KITTI\bb3_files'

#percent
RADIUS = 2
CIRCLE_RATIO = 0.3
BOUNDARIES = 0.33

# amount of data from dataset which will be loaded
# -1 for all data
DATA_AMOUNT = 200

IMAGE_EXTENSION = 'png'

# NETWORK RELATED SETTINGS
IS_TRAINING = True
IMG_WIDTH = 256
IMG_HEIGHT = 128
IMG_CHANNELS = 3 # based on channel number, image will be loaded colored or grayscaled

LEARNING_RATE = 0.0001
BATCH_SIZE = 32
UPDATE_EDGE = 0.01
MAX_ERROR = 0.001
ITERATIONS = 100
WEIGHT_FACTOR = 2.0

def get_all_files(path, extension):
    files = [os.path.relpath(f,path) for f in glob.glob(path + "*/*."+extension)]
    return files

def check_file_exists(path):
    return os.path.exists(path)

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

class DataModel:
    
    def __init__(self):
        self.image = None
        self.labels = []
        self.calib_matrix = []
        self.image_path = ''
        self.image_name = ''
        
class BB3Txt():
    
    def __init__(self):
        self.file_name: ''
        self.label: ''
        self.confidence: 0
        self.fbl_x: 0
        self.fbl_y: 0
        self.fbr_x: 0
        self.fbr_y: 0
        self.rbl_x: 0
        self.rbl_y: 0
        self.ftl_y: 0
        
        self.bb_center_x = 0
        self.bb_center_y = 0
        
        self.largest_dim = 0

    def to_string(self):
        data = self.file_name + ' ' + self.label + ' ' + str(self.confidence) + ' ' + str(self.fbl_x) + ' ' + str(self.fbl_y) + ' ' + str(self.fbr_x) + ' ' + str(self.fbr_y) + ' ' + str(self.rbl_x) + ' ' + str(self.rbl_y) + ' ' + str(self.ftl_y) + ' ' + str(self.bb_center_x) + ' ' + str(self.bb_center_y) + ' ' + str(self.largest_dim)
        return data

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer():
    def __init__(self):
        self._start_time = None
        self.final_time = None
        
    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        #print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        self.final_time = elapsed_time
        return elapsed_time
    
    def get_formated_time(self):        
        return str(datetime.timedelta(seconds=self.final_time))

class NetworkLoss(tf.keras.losses.Loss):
    def __init__(self, loss_name, scale, reduction=tf.keras.losses.Reduction.AUTO):
        super().__init__(reduction=reduction, name=loss_name)
        self.weight_factor = WEIGHT_FACTOR
        self.scale = scale
    
    def get_config(self):
        base_config = super().get_config()
        config = {'scale':self.scale,'loss_name':self.loss_name,'reduction':tf.keras.losses.Reduction.AUTO}
        return dict(list(base_config.items()) + list(config.items()))
    
    @tf.function
    def call(self, labels, outputs):
        labels = tf.cast(labels,tf.float32)
        outputs = tf.cast(outputs,tf.float32)
        loss = self.run_for_scale(outputs,labels)
        
        return loss
        
    
    @tf.function
    def run_for_scale(self,images,labels):
        errors = []
        for i in range(images.shape[0]):          
            current_img = images[i]
            current_lbl = labels[i]
            img_error = self.object_loss(current_lbl, current_img)
            errors.append(img_error)

        loss = tf.reduce_sum(errors)
        return loss
     
    @tf.function   
    def object_loss(self,target, image):
        width = image.shape.dims[1].value
        height = image.shape.dims[0].value
        channels = image.shape.dims[2].value
        
        # number of neurons in output layer
        N = width * height

        N_p = tf.math.count_nonzero(target[:, :, 0])  
        second_error = 0.0
        error = 0.0
        
        initial = tf.constant(1,dtype=tf.float32, shape=(height,width))
        tmp_initial = initial
        condition = tf.greater(target[:,:, 0], tf.constant(0,dtype=tf.float32),name="greater")
        weight_factor_array = tf.add(initial, tf.where(condition, (tmp_initial + self.weight_factor - 1), tmp_initial, name="where_condition"), name="assign" )

        error = tf.reduce_sum(tf.multiply(weight_factor_array, tf.square(tf.subtract(target[:,:, 0], image[:, :, 0]))))
        for c in range(1, channels):
            second_error += tf.reduce_sum(
                tf.multiply(self.weight_factor,
                             tf.multiply(target[:, :, 0],
                                          tf.square(tf.subtract(target[:, :, c], image[:, :, c])))))
        
                    
        error = (1/(2*N))*error  
        sec_error = tf.cond(tf.equal(N_p,0),lambda: tf.constant(0,dtype=tf.float32, shape=()),lambda: tf.cast(1/ (2 * N_p * (channels -1)),dtype=tf.float32)*second_error )            
        error += sec_error
       
        return error

class ODM_Conv2D_Layer(tf.keras.layers.Layer):
    
    def __init__(self, kernel, output_size, stride_size, dilation, name, activation=True,trainable=True,dtype=tf.float32, **kwargs ):
        super(ODM_Conv2D_Layer, self).__init__(name=name,trainable=trainable,dtype=dtype, **kwargs)
        self.kernel = kernel
        self.output_size = output_size
        self.stride_size = stride_size
        self.dilation = dilation
        self.layer_name = name
        self.activation = activation

    def get_config(self):
        config = {'kernel':self.kernel,'output_size':self.output_size,'stride_size':self.stride_size,'dilation':self.dilation,'name':self.name,'activation':self.activation}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self,input_shape):
        self.kernel = self.add_weight(name=self.layer_name+"_weights", shape=(self.kernel[0],self.kernel[1],input_shape[3],self.output_size), initializer='uniform')
        super().build(input_shape)
        
    @tf.function
    def activation_condition(self):
        return self.activation == True
        
    @tf.function
    def call(self, inputs):
        out = tf.nn.conv2d( inputs , self.kernel , strides=[ 1 , self.stride_size , self.stride_size , 1 ] ,
                        dilations=[1, self.dilation, self.dilation, 1], padding="SAME",name=self.layer_name+"_convolution") 
        return tf.cond(self.activation_condition(),lambda: tf.nn.relu(out, name=self.layer_name+"_activation"),lambda: out)
     
        
class ODM_MaxPool_Layer(tf.keras.layers.Layer):
    def __init__(self, pool_size, stride_size, name):
        super(ODM_MaxPool_Layer, self).__init__(name=name)
        self.pool_size = pool_size
        self.stride_size = stride_size
        self.layer_name = name

    def get_config(self):
        base_config = super().get_config()
        config = {'pool_size':self.pool_size,'stride_size':self.stride_size,'name':self.name}
        return dict(list(base_config.items()) + list(config.items()))

    @tf.function
    def call(self, inputs):
        return tf.nn.max_pool2d( inputs , ksize=[ 1 , self.pool_size , self.pool_size , 1 ] ,
                                 padding='SAME' , strides=[ 1 , self.stride_size , self.stride_size , 1 ], name=self.layer_name+'pool' )

class ObjectDetectionModel(tf.keras.Model):
    
    def __init__(self, kernel_size,name, **kwargs):
        super(ObjectDetectionModel, self).__init__(name=name, **kwargs )
        self.kernel_size = kernel_size
        self.model_name =name

        # self.layer1 = tf.keras.layers.Conv2D(64, [3,3], [1,1], "Same", dilation_rate=[1,1], activation="relu", kernel_initializer=tf.keras.initializers.GlorotUnifirm(),use_bias=True,trainable=True,name="layer1" )

        self.layer1 = ODM_Conv2D_Layer(kernel_size,64,1,1,"layer1")
        self.layer1_normal = tf.keras.layers.BatchNormalization(name="layer1_normalization")
        self.layer2 = ODM_Conv2D_Layer(kernel_size,64,2,1,"layer2")
        self.layer2_normal = tf.keras.layers.BatchNormalization(name="layer2_normalization")
        self.layer3 = ODM_Conv2D_Layer(kernel_size,128,1,1,"layer3")
        self.layer3_normal = tf.keras.layers.BatchNormalization(name="layer3_normalization")
        self.layer4 = ODM_Conv2D_Layer(kernel_size,128,1,1,"layer4")
        self.layer4_normal = tf.keras.layers.BatchNormalization(name="layer4_normalization")
        self.layer5 = ODM_Conv2D_Layer(kernel_size,128,1,3,"layer5")
        self.layer5_normal = tf.keras.layers.BatchNormalization(name="layer5_normalization")
        self.layer6 = ODM_Conv2D_Layer(kernel_size,128,1,6,"layer6")  
        self.layer6_normal = tf.keras.layers.BatchNormalization(name="layer6_normalization")   
        self.output_2 = ODM_Conv2D_Layer([1,1],8,1,1,"output_2",activation=False) 
        self.max_pool_layer_2 = ODM_MaxPool_Layer(2, 2, "layer7_maxpool")
        
        #self.block_2 = self.layer1.trainable_variables + self.layer2.trainable_variables + self.layer3.trainable_variables + self.layer4.trainable_variables + self.layer5.trainable_variables + self.layer6.trainable_variables + self.output_2.trainable_variables
        
        
        self.layer8 = ODM_Conv2D_Layer(kernel_size,256, 1, 1, "layer8")
        self.layer8_normal = tf.keras.layers.BatchNormalization(name="layer8_normalization")
        self.layer9 = ODM_Conv2D_Layer(kernel_size,256, 1, 1, "layer9")
        self.layer9_normal = tf.keras.layers.BatchNormalization(name="layer9_normalization")
        self.layer10 = ODM_Conv2D_Layer(kernel_size,256, 1, 3, "layer10")
        self.layer10_normal = tf.keras.layers.BatchNormalization(name="layer10_normalization")
        self.output_4 = ODM_Conv2D_Layer([1,1],8,1,1,"output_4",activation=False) 
        self.max_pool_layer_4 = ODM_MaxPool_Layer(2, 2, "layer11_maxpool")
        
        #self.block_4 = self.layer8.trainable_variables + self.layer9.trainable_variables + self.layer10.trainable_variables + self.output_4.trainable_variables

        self.layer12 = ODM_Conv2D_Layer(kernel_size,512, 1, 1, "layer12")
        self.layer12_normal = tf.keras.layers.BatchNormalization(name="layer11_normalization")
        self.layer13 = ODM_Conv2D_Layer(kernel_size,512, 1, 1, "layer13")
        self.layer13_normal = tf.keras.layers.BatchNormalization(name="layer12_normalization")
        self.layer14 = ODM_Conv2D_Layer(kernel_size,512, 1, 3, "layer14")
        self.layer14_normal = tf.keras.layers.BatchNormalization(name="layer13_normalization")
        self.output_8 = ODM_Conv2D_Layer([1,1],8,1,1,"output_8",activation=False) 
        self.max_pool_layer_8 = ODM_MaxPool_Layer(2, 2, "layer15_maxpool")
        
        #self.block_8 = self.layer12.trainable_variables + self.layer13.trainable_variables + self.layer14.trainable_variables + self.output_8.trainable_variables
            
        self.layer16 = ODM_Conv2D_Layer(kernel_size,512, 1, 1, "layer16")
        self.layer16_normal = tf.keras.layers.BatchNormalization(name="layer16_normalization")
        self.layer17 = ODM_Conv2D_Layer(kernel_size,512, 1, 1, "layer17")
        self.layer17_normal = tf.keras.layers.BatchNormalization(name="layer17_normalization")
        self.layer18 = ODM_Conv2D_Layer(kernel_size,512, 1, 3, "layer18")
        self.layer18_normal = tf.keras.layers.BatchNormalization(name="layer18_normalization")
        self.output_16 = ODM_Conv2D_Layer([1,1],8,1,1,"output_16",activation=False) 
        
        #self.block_16 = self.layer16.trainable_variables + self.layer17.trainable_variables + self.layer18.trainable_variables + self.output_16.trainable_variables
       
    # def get_config(self):
    #     layer_configs = []
    #     for layer in self.layers:
    #         layer_configs.append({
    #             'class_name': layer.__class__.__name__,
    #             'config': layer.get_config()
    #         })
    #     config = {
    #         'name': self.model_name,
    #         'layers': copy.copy(layer_configs),
    #         "kernel_size": self.kernel_size
    #     }

    #     return config

    @tf.function
    def call(self, input, training):
        
        x = tf.cast( input , dtype=tf.float32 ) 
        x = self.layer1(x)
        #x = self.layer1_normal(x, training)
        x = self.layer2(x)
        #x = self.layer2_normal(x, training)
        x = self.layer3(x)
        #x = self.layer3_normal(x, training)
        x = self.layer4(x)
        #x = self.layer4_normal(x, training)
        x = self.layer5(x)
        #x = self.layer5_normal(x, training)
        x = self.layer6(x)
        #x = self.layer6_normal(x, training)
        self.data_2 = self.output_2(x)
        
        self.block_2 = self.layer1.trainable_variables 
        self.block_2 = self.block_2 + self.layer2.trainable_variables
        self.block_2 = self.block_2 + self.layer3.trainable_variables
        self.block_2 = self.block_2 + self.layer4.trainable_variables
        self.block_2 = self.block_2 + self.layer5.trainable_variables
        self.block_2 = self.block_2 + self.layer6.trainable_variables
        self.block_2 = self.block_2 + self.output_2.trainable_variables

        x = self.max_pool_layer_2(x)
        x = self.layer8(x)
        #x = self.layer8_normal(x, training)
        x = self.layer9(x)
        #x = self.layer9_normal(x, training)
        x = self.layer10(x)
        #x = self.layer10_normal(x, training)
        self.data_4 = self.output_4(x)
        
        self.block_4 = self.max_pool_layer_2.trainable_variables
        self.block_4 = self.block_4 + self.layer8.trainable_variables
        self.block_4 = self.block_4 + self.layer9.trainable_variables
        self.block_4 = self.block_4 + self.layer10.trainable_variables
        self.block_4 = self.block_4 + self.output_4.trainable_variables
        
        x = self.max_pool_layer_4(x)
        x = self.layer12(x)
        #x = self.layer12_normal(x, training)
        x = self.layer13(x)
        #x = self.layer13_normal(x, training)
        x = self.layer14(x)
        #x = self.layer14_normal(x, training)
        self.data_8 = self.output_8(x)
        
        self.block_8 = self.max_pool_layer_4.trainable_variables
        self.block_8 = self.block_8 + self.layer12.trainable_variables
        self.block_8 = self.block_8 + self.layer13.trainable_variables
        self.block_8 = self.block_8 + self.layer14.trainable_variables
        self.block_8 = self.block_8 + self.output_8.trainable_variables
        
        x = self.max_pool_layer_8(x)
        x = self.layer16(x)
        #x = self.layer16_normal(x, training)
        x = self.layer17(x)
        #x = self.layer17_normal(x, training)
        x = self.layer18(x)
        #x = self.layer18_normal(x, training)
        self.data_16 = self.output_16(x)
        
        self.block_16 = self.max_pool_layer_8.trainable_variables
        self.block_16 = self.block_16 + self.layer16.trainable_variables
        self.block_16 = self.block_16 + self.layer17.trainable_variables
        self.block_16 = self.block_16 + self.layer18.trainable_variables
        self.block_16 = self.block_16 + self.output_16.trainable_variables

        return [self.data_2, self.data_4, self.data_8, self.data_16]
    
    
class NetworkCreator():
    
    def __init__(self):
        self.BatchSize = BATCH_SIZE
        self.learning = LEARNING_RATE

        self.optimizer2 = tf.optimizers.Adam(name="adam_optimizer_2",learning_rate=self.get_learning_rate)
        self.optimizer4 = tf.optimizers.Adam(name="adam_optimizer_4",learning_rate=self.get_learning_rate)
        self.optimizer8 = tf.optimizers.Adam(name="adam_optimizer_8",learning_rate=self.get_learning_rate)
        self.optimizer16 = tf.optimizers.Adam(name="adam_optimizer_16",learning_rate=self.get_learning_rate)
        
        self.loss2 = NetworkLoss( "loss_function_2",2)
        self.loss4 = NetworkLoss( "loss_function_4",4)
        self.loss8 = NetworkLoss( "loss_function_8",8)
        self.loss16 = NetworkLoss( "loss_function_16",16)
    
    @tf.function
    def compute_loss(self, label, output):
        loss_2 = self.loss2(label[0], output[0])
        loss_4 = self.loss4(label[1], output[1])
        loss_8 = self.loss8(label[2], output[2])
        loss_16 = self.loss16(label[3], output[3])
        return [loss_2, loss_4, loss_8, loss_16]
    
    @tf.function
    def loss(self,model,x,y,training):
        y_ = model(x,training)
        loss = self.compute_loss(y, y_)

        return loss
    
    def compute_gradient(self, model, inputs, targets):
        #print(model.block_2)
        with tf.GradientTape(persistent= True) as tape:

            loss_value = self.loss(model, inputs, targets, training=True)
         
        grad2 = tape.gradient(loss_value[0], model.block_2)
        grad4 = tape.gradient(loss_value[1], model.block_4)
        grad8 = tape.gradient(loss_value[2], model.block_8)
        grad16 = tape.gradient(loss_value[3], model.block_16)
        del tape  
        
        self.optimizer2.apply_gradients(zip(grad2, model.block_2)) 
        self.optimizer4.apply_gradients(zip(grad4, model.block_4)) 
        self.optimizer8.apply_gradients(zip(grad8, model.block_8)) 
        self.optimizer16.apply_gradients(zip(grad16, model.block_16)) 
       
        return loss_value
    
    def get_learning_rate(self):
        return self.learning
    
    def continue_training(self, acc):           
        if acc[0] > MAX_ERROR or acc[1] > MAX_ERROR or acc[2] > MAX_ERROR or acc[3] > MAX_ERROR:
            return True
        else:
            return False 
    
    def train(self, loader, model):
        
        epoch = 1
        acc = [9,9,9,9]
        errors = []
        t_global = Timer()
        t_global.start()

        while self.continue_training(acc):
            epoch_loss_avg = []
            t = Timer()
            t.start()
            for i in range(ITERATIONS):                
                image_batch, labels_batch, names = loader.get_train_data(BATCH_SIZE) 
                loss_value = self.compute_gradient(model,image_batch, labels_batch)              
                epoch_loss_avg.append(loss_value)

            _ = t.stop()
            acc = np.mean(epoch_loss_avg, axis=0)
            print("Epoch {:d}: Loss 2: {:.6f}, Loss 4: {:.6f}, Loss 8: {:.6f}, Loss 16: {:.6f}  Epoch duration: ".format(epoch,acc[0],acc[1],acc[2],acc[3]) + t.get_formated_time())
            # model.save_weights(cfg.MODEL_WEIGHTS)

            if epoch == 30:
                self.learning = self.learning / 10
            if epoch == 80:
                self.learning = self.learning / 10
                
            epoch += 1
                         
        _ = t_global.stop()
        print(errors) 
     
     
    def start_train(self, loader):
        

        model = ObjectDetectionModel([3,3],'ObjectDetectionModel')
        self.train(loader,model)
        model.summary()

                
class Loader():
    def __init__(self):
        self.Data = []
        self.image_path = ''
        self.label_path = ''
        self.calib_path = ''
        self.amount = 0
        self.start_from = 0
        self.image_extension = ''
        self.colored = False

        self.init()

    def init(self):
        
        if check_file_exists(IMAGE_PATH):
            self.image_path = IMAGE_PATH
        else:
            print("Image path '"+ IMAGE_PATH +"' not found!!!")
            
        if check_file_exists(LABEL_PATH):
            self.label_path = LABEL_PATH
        else:
            print("Label path '"+ LABEL_PATH +"' not found!!!")  
            
        if check_file_exists(CALIB_PATH):
            self.calib_path = CALIB_PATH
        else:
            print("Calibration path '"+ CALIB_PATH +"' not found!!!")
            
        self.amount = DATA_AMOUNT
        self.image_extension = IMAGE_EXTENSION
        if IMG_CHANNELS == 1:
            self.colored = False
        else:
            self.colored = True           
          
    def load_data(self):
        """
        Load data from files on specified path.
        Image file name is formated to be "000000" with specified extension
        
        Returns:
            Data are stored to properties
        """
        assert self.image_path != '', 'Image path not set. Nothing to work with. Check config file.'
        assert self.label_path != '', 'Label path not set. Nothing to work with. Check config file.'
        assert self.calib_path != '', 'Calibration path not set. Nothing to work with. Check config file.'
        print('Loading training files')
        
        image_pathss = []
        label_paths = []
        calib_paths = []
        
        # in img_files are absolut paths
        img_files = get_all_files(self.image_path, self.image_extension)
        if self.amount == -1:
            amount_to_load = len(img_files)
        else:
            amount_to_load = self.amount
        
        for i in range(len(img_files)):
            file_ = img_files[i]
            dot_index = file_.find('.')
            file_name = file_[:dot_index]
            
            image_path = self.image_path + '/' + file_
            label_path = self.label_path + '/' + file_name + '.txt'
            if not check_file_exists(label_path):
                continue
                    
            calib_path = self.calib_path + '/' + file_name + '.txt'
            if not check_file_exists(calib_path):
                continue
            
            image, width, height = self._load_image(image_path)
            if image is None:
                continue
                
            # calibration
            calib_matrix = self.load_calibration(calib_path)
            if calib_matrix is None:
                continue
            
            # label
            labels = self._load_label(label_path, file_name, calib_matrix, width, height)
            if labels is None:
                continue
          
            data = DataModel()
            data.image = image
            data.image_path = image_path
            data.image_name = file_
            data.labels = labels
            data.calib_matrix = calib_matrix

            self.Data.append(data)
            if len(self.Data) == amount_to_load:
                break
            
    def _load_image(self, image_path):
        """
        Reads a image with number x and based on parameter colored result will have 1 or 3 chanels

        Input:
        image_path: Path to folder with images
        x: number of image
        Returns:
        image as matrix
        """
        if self.colored:
            im = cv2.imread(image_path, cv2.IMREAD_COLOR)
        else:
            im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if im.any() == None:
            return None

        resized = cv2.resize(im, (IMG_WIDTH,IMG_HEIGHT), interpolation=cv2.INTER_AREA) # opencv resize function takes as desired shape (width,height) !!!
        normalized = normalize(resized)
        return normalized, im.shape[1], im.shape[0]

    def load_calibration(self, calib_path):
        """
        Reads a camera matrix P (3x4) stored in the row-major scheme.

        Input:
            calib_path: Row-major stored matrix separated by spaces, first element is the matrix name
            x: number of image
        Returns:
            camera matrix P 4x4
        """

        with open(calib_path, 'r') as infile_calib:
            for line in infile_calib:
                if line[:2] == 'P2':
                    line_data = line.rstrip('\n')
                    data = line_data.split(' ')

                    if data[0] != 'P2:':
                        print('ERROR: We need left camera matrix (P2)!')
                        exit(1)

                    P = np.asmatrix([[float(data[1]), float(data[2]),  float(data[3]),  float(data[4])],
                                    [float(data[5]), float(data[6]),  float(data[7]),  float(data[8])],
                                    [float(data[9]), float(data[10]), float(data[11]), float(data[12])]])
                    return P

        return None

    def _load_label(self, label_path, file_name, calib_matrix, width, height):
        """
        Reads a label file to specific image.
        Read only Car labels

        Input:
            label_path: Row-major stored label separated by spaces, first element is the label name
            x: number of image
        Returns:
            LabelModel object
        """

        # check if bb3_files folder exists
        # if exists then load from this file they are pre processed to 33btxt format and ready to use
        # if not load kitti label and create bb3txt file, next time this file will be used
        bb3_path = BB3_FOLDER       
        bb3_file_path = bb3_path + '/' +file_name+'.txt'
        result = self._load_from_bb3_folder(bb3_file_path)
        return result
    
    def _load_from_bb3_folder(self, path):
        labels = []
        with open(path, 'r') as infile_label:

            for line in infile_label:
                line = line.rstrip(r'\n')
                data = line.split(' ')

                label = BB3Txt()
                label.file_name = data[0]
                label.label = data[1]
                label.fbl_x = float(data[2])
                label.fbl_y = float(data[3])
                label.fbr_x = float(data[4])
                label.fbr_y = float(data[5])
                label.rbl_x = float(data[6])
                label.rbl_y = float(data[7])
                label.ftl_y = float(data[8])
                label.bb_center_x = float(data[9])
                label.bb_center_y = float(data[10])
                
                label.largest_dim = float(data[11])
                
                labels.append(label)

        return labels
    
    def get_train_data(self, batch_size):
        
        data = random.sample(self.Data, batch_size)
        result_image = []
        result_image_names = []
        gt_2 = []
        gt_4 = []
        gt_8 = []
        gt_16 = []
        for x in range(batch_size):
            if x >= batch_size:
                break
            result_image.append(data[x].image)
            scale2, scale4, scale8, scale16 = self.create_ground_truth(self.labels_array_for_training(data[x].labels)) 
            gt_2.append(scale2)
            gt_4.append(scale4)
            gt_8.append(scale8)
            gt_16.append(scale16)
            result_image_names.append(data[x].image_name)

        return np.asarray(result_image), [np.asarray(gt_2), np.asarray(gt_4), np.asarray(gt_8), np.asarray(gt_16)], np.asarray(result_image_names)
    
    def create_ground_truth(self, label):
        scale2 = self.create_target_response_map(label,128,64,RADIUS,CIRCLE_RATIO,BOUNDARIES,2)        
        scale4 = self.create_target_response_map(label,64,32,RADIUS,CIRCLE_RATIO,BOUNDARIES,4)        
        scale8 = self.create_target_response_map(label,32,16,RADIUS,CIRCLE_RATIO,BOUNDARIES,8)        
        scale16 = self.create_target_response_map(label,16,8,RADIUS,CIRCLE_RATIO,BOUNDARIES,16)
        np.save('scale2.npy', scale2)
        return scale2, scale4, scale8, scale16

    def GetObjectBounds(self,radius,circle_ratio,boundaries,scale):
        ideal_size = (2.0 * radius + 1.0) / circle_ratio * scale
        # bound above
        ext_above = ((1.0 - boundaries) * ideal_size) / 2.0 + boundaries * ideal_size
        bound_above = ideal_size + ext_above
        
        # bound below
        diff = ideal_size / 2.0
        ext_below = ((1 - boundaries)* diff) / 2.0 + boundaries * diff
        bound_below = ideal_size - ext_below
        
        return bound_above, bound_below, ideal_size
    
    def create_target_response_map(self, labels, width, height, radius,circle_ratio,boundaries,scale):
        
        result = np.zeros((height,width,8))        
        maps = cv2.split(result)
  
        bound_above, bound_below, ideal = self.GetObjectBounds(radius,circle_ratio,boundaries,scale)
        for i in range(len(labels)):            
            label = labels[i]
            if label[0] == -1:
                # 2. dimension of array have to be same acros 1. dimension, we complete the missing elements with -1, now they will be ignored
                continue
            # 0       1       2       3       4       5       6     7           8           9
            # fblx    fbly    fbrx    fbry    rblx    rbly    ftly  center_x    center_y    largest_dim
            
            if label[9] >= bound_below and label[9] <= bound_above:
                x = int(label[7] / scale)
                y = int(label[8] / scale)
                
                scaling_ratio = 1.0 / scale                
                cv2.circle(maps[0], ( x, y ), int(radius), 1, -1)
                cv2.GaussianBlur(maps[0], (3, 3), 100)

                for k in range(1,8):
                    for l in range(-radius,radius,1):
                        for j in range(-radius,radius,1):
                            xp = x + j
                            yp = y + l
                            if xp >= 0 and xp < width and yp >= 0 and yp < height:
                                if maps[0][yp][xp] > 0.0:
                                    if k ==1 or k == 3 or k == 5:
                                        maps[k][yp][xp] = 0.5 + (label[k-1] - x - j * scale) / ideal
                                    elif k == 2 or k == 4 or k == 6 or k == 7:
                                        maps[k][yp][xp] = 0.5 + (label[k-1] - y - l * scale) / ideal
        
        
        # stack created maps together
        result[:,:,0] = maps[0]
        result[:,:,1] = maps[1]
        result[:,:,2] = maps[2]
        result[:,:,3] = maps[3]
        result[:,:,4] = maps[4]
        result[:,:,5] = maps[5]
        result[:,:,6] = maps[6]
        result[:,:,7] = maps[7]
        
        return result
    
    def labels_array_for_training(self,labels):
        label_array = []
        for i in range(len(labels)):
            label = labels[i]
            label_array.append([float(label.fbl_x), float(label.fbl_y), float(label.fbr_x), float(label.fbr_y), float(label.rbl_x), float(label.rbl_y), float(label.ftl_y), float(label.bb_center_x), float(label.bb_center_y), float(label.largest_dim)])
        
        return label_array
    def complete_uneven_arrays(self, array, insert_val = -1.0):
        lens = np.array([len(item) for item in array])
        mask = lens[:,None] > np.arange(lens.max())
        out = np.full((mask.shape[0],mask.shape[1],10),insert_val,dtype=np.float32)
        out[mask] = np.concatenate(array)
        return out


loader = Loader()
loader.load_data()
print("loader done")
nc = NetworkCreator()
with tf.device('/device:GPU:0'):
  nc.start_train(loader)


