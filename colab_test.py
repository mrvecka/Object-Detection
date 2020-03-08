import tensorflow as tf
import cv2
import numpy as np
import os
import glob
import random
import time
import datetime



IMAGE_PATH = 'drive/My Drive/colab_data/images'
CALIB_PATH = 'drive/My Drive/colab_data/calib'
LABEL_PATH = 'drive/My Drive/colab_data/label'
BB3_FOLDER = 'drive/My Drive/colab_data/bb3'

#percent
RADIUS = 2
CIRCLE_RATIO = 0.3
BOUNDARIES = 0.33

# amount of data from dataset which will be loaded
# -1 for all data
DATA_AMOUNT = 32

IMAGE_EXTENSION = 'png'

# NETWORK RELATED SETTINGS
IS_TRAINING = True
IMG_WIDTH = 256
IMG_HEIGHT = 128
IMG_CHANNELS = 3 # based on channel number, image will be loaded colored or grayscaled

DEVICE_NAME = "/gpu:0"
LEARNING_RATE = 0.0001
BATCH_SIZE = 8
UPDATE_EDGE = 0.001
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

class Timer:
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
    def __init__(self,batch, scale, loss_name, reduction=tf.losses.Reduction.AUTO):
        super().__init__(reduction=reduction, name=loss_name)
        self.radius = RADIUS
        self.circle_ratio = CIRCLE_RATIO
        self.boundaries = BOUNDARIES
        self.weight_factor = WEIGHT_FACTOR
        self.scale = scale
        self.batch_size = batch
    
    def get_config(self):
        base_config = super().get_config()
        config = {'batch':self.batch,'scale':self.scale,'loss_name':self.loss_name,'reduction':tf.keras.losses.Reduction.AUTO}
        return dict(list(base_config.items()) + list(config.items()))
    
    def call(self,images,labels):
        errors = []
        for i in range(self.batch_size):          
            current_img = images[i]
            current_lbl = labels[i]
            img_error = self.scan_image_function(current_img, current_lbl)
            errors.append(img_error)

        errors_as_tensor = tf.convert_to_tensor(errors,dtype=tf.float32)
        loss = tf.reduce_sum(errors_as_tensor)
        return loss
    
    def scan_image_function(self, image, label):

        width = image.shape.dims[1].value
        height = image.shape.dims[0].value
        channels = image.shape.dims[2].value
        
        target = self.create_target_response_map(label, width, height)
        
        target = tf.reshape(target,(channels,height,width))        
        
        # number of neurons in each output layer
        N = width * height

        N_p = tf.math.count_nonzero(image[:, :, 0])      
        second_error = 0
        error = 0.0
                   
        # get array of weight factor with the same shape as target 
        initial = tf.constant(1,dtype=tf.float32, shape=(height,width))
        tmp_initial = initial
        condition = tf.greater(target[0,:, :], tf.constant(0,dtype=tf.float32),name="greater")
        weight_factor_array = tf.add(initial, tf.where(condition, (tmp_initial + self.weight_factor - 1), tmp_initial, name="where_condition"), name="assign" )

        error = tf.reduce_sum(tf.multiply(weight_factor_array, tf.square(tf.subtract(target[0,:, :], image[:, :, 0]))))
        for c in range(1, channels):
            second_error += tf.reduce_sum(
                tf.multiply(self.weight_factor,
                             tf.multiply(target[0, :, :],
                                          tf.square(tf.subtract(target[c,:, :], image[:, :, c])))))
        
                    
        error = (1/(2*N))*error     
        tmp = 1/ (3 * N_p * (channels -1))    
        error += tf.cast(tmp, tf.float32) * second_error
       
        return error
           
    def GetObjectBounds(self):
        ideal_size = (2.0 * self.radius + 1.0) / self.circle_ratio * self.scale
        # bound above
        ext_above = ((1.0 - self.boundaries) * ideal_size) / 2.0 + self.boundaries * ideal_size
        bound_above = ideal_size + ext_above
        
        # bound below
        diff = ideal_size / 2.0
        ext_below = ((1 - self.boundaries)* diff) / 2.0 + self.boundaries * diff
        bound_below = ideal_size - ext_below
        
        return bound_above, bound_below, ideal_size
    
    def create_target_response_map(self, labels, width, height):
                
        maps = cv2.split(np.zeros((height,width,8)))
        bound_above, bound_below, ideal = self.GetObjectBounds()
        for i in range(len(labels)):            
            label = labels[i]
            if label[0] == -1:
                continue
            # 0       1       2       3       4       5       6     7           8           9
            # fblx    fbly    fbrx    fbry    rblx    rbly    ftly  center_x    center_y    largest_dim
            
            if label[9] >= bound_below and label[9] <= bound_above:
                x = int(label[7] / self.scale)
                y = int(label[8] / self.scale)
                
                scaling_ratio = 1.0 / self.scale                
                cv2.circle(maps[0], ( x, y ), int(self.radius), 1, -1)
                cv2.GaussianBlur(maps[0], (3, 3), 100)

                for c in range(1,8):
                    
                    for l in range(-self.radius,self.radius,1):
                        for j in range(-self.radius,self.radius,1):
                            xp = x + j
                            yp = y + l
                            
                            if xp >= 0 and xp < width and yp >= 0 and yp < height:
                                if maps[0][yp][xp] > 0.0:
                                    if c ==1 or c == 3 or c == 5:
                                        maps[c][yp][xp] = 0.5 + (label[c-1] - x - j * self.scale) / ideal
                                    elif c == 2 or c == 4 or c == 6 or c == 7:
                                        maps[c][yp][xp] = 0.5 + (label[c-1] - y - l * self.scale) / ideal
        
        return np.asarray(maps,dtype=np.float32)

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
     
    @tf.function
    def call(self, inputs):
        out = tf.nn.conv2d( inputs , self.kernel , strides=[ 1 , self.stride_size , self.stride_size , 1 ] ,
                        dilations=[1, self.dilation, self.dilation, 1], padding="SAME", name=self.layer_name+'_convolution' ) 
        if self.activation:
            return tf.nn.relu( out, name=self.layer_name+'_relu_activation') 
        else:
            return out
        
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
        self.output_2 = ODM_Conv2D_Layer([1,1],8,1,1,"output_2", False)
        self.max_pool_layer_2 = ODM_MaxPool_Layer(2, 2, "layer7_maxpool")
        
        self.out_2_trainable_variables = self.layer1.trainable_variables + self.layer2.trainable_variables + self.layer3.trainable_variables + self.layer4.trainable_variables + self.layer5.trainable_variables + self.layer6.trainable_variables + self.output_2.trainable_variables
        
        
        self.layer8 = ODM_Conv2D_Layer(kernel_size,256, 1, 1, "layer8")
        self.layer8_normal = tf.keras.layers.BatchNormalization(name="layer8_normalization")
        self.layer9 = ODM_Conv2D_Layer(kernel_size,256, 1, 1, "layer9")
        self.layer9_normal = tf.keras.layers.BatchNormalization(name="layer9_normalization")
        self.layer10 = ODM_Conv2D_Layer(kernel_size,256, 1, 3, "layer10")
        self.layer10_normal = tf.keras.layers.BatchNormalization(name="layer10_normalization")
        self.output_4 = ODM_Conv2D_Layer([1,1],8, 1, 1, "output_4", False)
        self.max_pool_layer_4 = ODM_MaxPool_Layer(2, 2, "layer11_maxpool")
        
        self.out_4_trainable_variables = self.layer8.trainable_variables + self.layer9.trainable_variables + self.layer10.trainable_variables + self.output_4.trainable_variables

        self.layer12 = ODM_Conv2D_Layer(kernel_size,512, 1, 1, "layer12")
        self.layer12_normal = tf.keras.layers.BatchNormalization(name="layer11_normalization")
        self.layer13 = ODM_Conv2D_Layer(kernel_size,512, 1, 1, "layer13")
        self.layer13_normal = tf.keras.layers.BatchNormalization(name="layer12_normalization")
        self.layer14 = ODM_Conv2D_Layer(kernel_size,512, 1, 3, "layer14")
        self.layer14_normal = tf.keras.layers.BatchNormalization(name="layer13_normalization")
        self.output_8 = ODM_Conv2D_Layer([1,1], 8, 1, 1, "output_8", False)
        self.max_pool_layer_8 = ODM_MaxPool_Layer(2, 2, "layer15_maxpool")
        
        self.out_8_trainable_variables = self.layer12.trainable_variables + self.layer13.trainable_variables + self.layer14.trainable_variables + self.output_8.trainable_variables
            
        self.layer16 = ODM_Conv2D_Layer(kernel_size,512, 1, 1, "layer16")
        self.layer16_normal = tf.keras.layers.BatchNormalization(name="layer16_normalization")
        self.layer17 = ODM_Conv2D_Layer(kernel_size,512, 1, 1, "layer17")
        self.layer17_normal = tf.keras.layers.BatchNormalization(name="layer17_normalization")
        self.layer18 = ODM_Conv2D_Layer(kernel_size,512, 1, 3, "layer18")
        self.layer18_normal = tf.keras.layers.BatchNormalization(name="layer18_normalization")
        self.output_16 = ODM_Conv2D_Layer([1,1],8, 1, 1, "output_16", False)
        
        self.out_16_trainable_variables = self.layer16.trainable_variables + self.layer17.trainable_variables + self.layer18.trainable_variables + self.output_16.trainable_variables

    @tf.function
    def call(self, input, training):
        
        x = tf.cast( input , dtype=tf.float32 ) 
        x = self.layer1(x)
        x = self.layer1_normal(x, training)
        x = self.layer2(x)
        x = self.layer2_normal(x, training)
        x = self.layer3(x)
        x = self.layer3_normal(x, training)
        x = self.layer4(x)
        x = self.layer4_normal(x, training)
        x = self.layer5(x)
        x = self.layer5_normal(x, training)
        x = self.layer6(x)
        x = self.layer6_normal(x, training)
        self.data_2 = self.output_2(x)
        x = self.max_pool_layer_2(x)
        
        self.out_2_trainable_variables = self.layer1.trainable_variables + self.layer2.trainable_variables + self.layer3.trainable_variables + self.layer4.trainable_variables + self.layer5.trainable_variables + self.layer6.trainable_variables + self.output_2.trainable_variables
        
        x = self.layer8(x)
        x = self.layer8_normal(x, training)
        x = self.layer9(x)
        x = self.layer9_normal(x, training)
        x = self.layer10(x)
        x = self.layer10_normal(x, training)
        self.data_4 = self.output_4(x)
        x = self.max_pool_layer_4(x)
        
        self.out_4_trainable_variables = self.layer8.trainable_variables + self.layer9.trainable_variables + self.layer10.trainable_variables + self.output_4.trainable_variables
        
        x = self.layer12(x)
        x = self.layer12_normal(x, training)
        x = self.layer13(x)
        x = self.layer13_normal(x, training)
        x = self.layer14(x)
        x = self.layer14_normal(x, training)
        self.data_8 = self.output_8(x)
        x = self.max_pool_layer_8(x)
        
        self.out_8_trainable_variables = self.layer12.trainable_variables + self.layer13.trainable_variables + self.layer14.trainable_variables + self.output_8.trainable_variables
        
        x = self.layer16(x)
        x = self.layer16_normal(x, training)
        x = self.layer17(x)
        x = self.layer17_normal(x, training)
        x = self.layer18(x)
        x = self.layer18_normal(x, training)
        self.data_16 = self.output_16(x)
        
        self.out_16_trainable_variables = self.layer16.trainable_variables + self.layer17.trainable_variables + self.layer18.trainable_variables + self.output_16.trainable_variables
        
        return self.data_2, self.data_4, self.data_8, self.data_16
    
    
class NetworkCreator():
    
    def __init__(self):
        self.device = DEVICE_NAME
        self.BatchSize = BATCH_SIZE
        
        self.model = None
        self.optimizer = None
    
    def network_loss_function(self,out_2, out_4, out_8, out_16, labels):
                        
        # tf.config.experimental_run_functions_eagerly(True)
        with tf.name_scope('loss_2'):
            loss_2_model = NetworkLoss(self.BatchSize, 2.0, "loss_scale_2")
            loss_2_result = loss_2_model(out_2,labels)
                  
        del loss_2_model    
        with tf.name_scope('loss_4'):
            loss_4_model = NetworkLoss(self.BatchSize, 4.0, "loss_scale_4")
            loss_4_result = loss_4_model(out_4,labels)
        
        del loss_4_model  
        with tf.name_scope('loss_8'):
            loss_8_model = NetworkLoss(self.BatchSize, 8.0, "loss_scale_8")
            loss_8_result = loss_8_model(out_8,labels)
            
        del loss_8_model        
        with tf.name_scope('loss_16'):
            loss_16_model = NetworkLoss(self.BatchSize, 16.0, "loss_scale_16")
            loss_16_result = loss_16_model(out_16,labels)           
        # tf.config.experimental_run_functions_eagerly(False)
   
        del loss_16_model
        
        return tf.reduce_sum([loss_2_result,loss_4_result,loss_8_result, loss_16_result],name="global_loss")
    
    def train_step(self, inputs, label):
            
        # GradientTape need to be persistent because we want to compute multiple gradients and it is no allowed by default
        # persistent=True
        with tf.GradientTape() as tape:
            out_2, out_4, out_8, out_16 = self.model(inputs,True)
            loss = self.network_loss_function(out_2, out_4, out_8, out_16, label)
            
        grads_2 = tape.gradient( loss , self.model.trainable_variables)
        
        # after gradient computation, we delete GradientTape object so it could be garbage collected
        del tape
        
        self.optimizer.apply_gradients(zip(grads_2, self.model.trainable_variables))

        
        return loss
    
    def get_learning_rate(self):
        return self.learning
    
    def train(self, loader,test_acc,epoch,update_edge,max_error):
        

        iteration = ITERATIONS
        errors = []
        t_global = Timer()
        t_global.start()
        while test_acc > max_error:
            t = Timer()
            t.start()
            for i in range(iteration):                
                # train_fn = self.train_step_fn()  
                image_batch, labels_batch, = loader.get_train_data(self.BatchSize)    
                _ = self.train_step(image_batch, labels_batch)
                #print("Iteration: ",i)
            
            image_batch, labels_batch, = loader.get_train_data(self.BatchSize)
            out_2, out_4, out_8, out_16 = self.model(image_batch,False)
            test_acc = self.network_loss_function(out_2, out_4, out_8, out_16, labels_batch)
            acc = test_acc.numpy()
            errors.append(acc)
            _ = t.stop()
            print(f"Epoch: {epoch:4d} test error: {acc:0.5f} Epoch duration: " + t.get_formated_time()) # make time hh:mm:ss
                
            epoch += 1
            
            # if test_acc < update_edge:
            #     self.learning = self.learning / 10
            #     update_edge = update_edge / 10
            #     print("Learning rate updated to", self.learning) 
                
        acc = test_acc.numpy()
        _ = t_global.stop()
        print(f"Final test error: {acc:0.5f} Training duration: " + t_global.get_formated_time())
        print(errors) 
     
     
    def start_train(self, loader):
        
        self.learning = LEARNING_RATE

        self.optimizer = tf.optimizers.Adam(name="adam_optimizer_2",learning_rate=self.get_learning_rate)
        
          
        self.model = ObjectDetectionModel([3,3],'ObjectDetectionModel')        
        self.model.compile()
        self.train(loader,1,1,UPDATE_EDGE,MAX_ERROR)
        # self.model.save_weights(MODEL_WEIGHTS)
                
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
                label.coinfidence = float(data[2])
                label.fbl_x = float(data[3])
                label.fbl_y = float(data[4])
                label.fbr_x = float(data[5])
                label.fbr_y = float(data[6])
                label.rbl_x = float(data[7])
                label.rbl_y = float(data[8])
                label.ftl_y = float(data[9])
                label.bb_center_x = float(data[10])
                label.bb_center_y = float(data[11])
                
                label.largest_dim = float(data[12])
                
                labels.append(label)

        return labels
    
    def get_train_data(self, batch_size):

        # print(len(self.Data))
        data = random.sample(self.Data, batch_size)
        result_image = []
        result_label = []
        for x in range(batch_size):
            if x >= batch_size:
                break
            result_image.append(data[x].image)
            labels = self.labels_array_for_training(data[x].labels)
            result_label.append(labels)
            
        result_label = self.complete_uneven_arrays(result_label)
        return np.asarray(result_image), np.asarray(result_label)
    
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


