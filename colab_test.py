__date__   = '14/05/2020'
__author__ = 'Lukas Mrvecka'
__email__  = 'lukas.mrvecka.st@vsb.cz'

import tensorflow as tf
import cv2
import sys
import numpy as np
import os
import glob
import random
import time
import datetime
import copy

IMAGE_PATH = r'D:\Documents\KITTI\images\training\image'
BB3_PATH = r'D:\Documents\KITTI\bb3_files'

# IMAGE_PATH = 'drive/My Drive/object detection/colab_data/images'
# CALIB_PATH = 'drive/My Drive/object detection/colab_data/calib'__date__   = '14/05/2020'
__author__ = 'Lukas Mrvecka'
__email__  = 'lukas.mrvecka.st@vsb.cz'

import tensorflow as tf
import cv2
import sys
import numpy as np
import os
import glob
import random
import time
import datetime
import copy


IMAGE_PATH = 'drive/My Drive/object detection/colab_data/images'
# CALIB_PATH = 'drive/My Drive/object detection/colab_data/calib'
# LABEL_PATH = 'drive/My Drive/object detection/colab_data/label'
BB3_PATH = 'drive/My Drive/object detection/colab_data/bb3'
MODEL_WEIGHTS = 'drive/My Drive/object detection/colab_data/model_weights.h5'
# MODEL_JSON = 'drive/My Drive/object detection/colab_data/model_spec.json'

SAVE_MODEL_EVERY = 10
UPDATE_LEARNING_RATE = []
CONTINUE_TRAINING = True

#percent
RADIUS = 2
CIRCLE_RATIO = 0.3
BOUNDARIES = 0.33

# amount of data from dataset which will be loaded
# -1 for all data
DATA_AMOUNT = 2000

IMAGE_EXTENSION = 'png'

# NETWORK RELATED SETTINGS
IS_TRAINING = True
IMG_WIDTH = 256
IMG_HEIGHT = 128
IMG_CHANNELS = 3 # based on channel number, image will be loaded colored or grayscaled

LEARNING_RATE = 0.00000001
BATCH_SIZE = 32
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
    def object_loss(self, target, image):
        width = image.shape.dims[1].value
        height = image.shape.dims[0].value
        channels = image.shape.dims[2].value
        
        # number of neurons in output layer
        N_p = tf.math.count_nonzero(target[:, :, 0])  
     
        initial = tf.constant(1,dtype=tf.float32, shape=(height,width))
        tmp_initial = initial
        condition = tf.greater(target[:,:, 0], tf.constant(0,dtype=tf.float32),name="greater")
        weight_factor_array = tf.add(initial, tf.where(condition, (tmp_initial + self.weight_factor - 1), tmp_initial, name="where_condition"), name="assign" )

        error = tf.reduce_sum(tf.multiply(weight_factor_array, tf.square(tf.subtract(target[:,:, 0], image[:, :, 0]))))
        second_error = 0.0
        for c in range(1, channels):
            second_error += tf.reduce_sum(
                tf.multiply(self.weight_factor,
                             tf.multiply(target[:, :, 0],
                                          tf.square(tf.subtract(target[:, :, c], image[:, :, c])))))
        
                    
        N = width * height
        error = (1/(2*N))*error 
        
        # there are no objects in some scales and N_p will be 0 so to prevent division by 0
        # sec_error = tf.cast(1/ (2 * N_p * (channels -1)),dtype=tf.float32)*second_error
        sec_error = tf.cond(tf.equal(N_p,0),lambda: tf.constant(0,dtype=tf.float32, shape=()),lambda: tf.cast(1/ (2 * N_p * (channels -1)),dtype=tf.float32)*second_error )            
        error += sec_error
        return tf.cast(error, dtype=tf.float32)

class ODM_Input_Layer(tf.keras.layers.Layer):
    
    def __init__(self, name, dtype=tf.float32, **kwargs ):
        super(ODM_Input_Layer, self).__init__(name=name,trainable=False,dtype=dtype,autocast=False, **kwargs)
        self.layer_name = name

    def get_config(self):
        config = {'name':self.layer_name}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self,input_shape):
        super(ODM_Input_Layer, self).build(input_shape)
     
    @tf.function   
    def call(self, inputs):    
        _min = tf.reduce_min(inputs)
        _max = tf.reduce_max(inputs)
        
        result = tf.divide(tf.subtract(inputs,_min),tf.subtract(_max,_min))
        result = tf.cast( result , dtype=tf.float32 ) 
        return result

class ODM_Conv2D_Layer(tf.keras.layers.Layer):
    
    def __init__(self, kernel, output_size, stride_size, dilation, name, activation=True,trainable=True,dtype=tf.float32, **kwargs ):
        super(ODM_Conv2D_Layer, self).__init__(name=name,trainable=trainable,dtype=dtype,autocast=False, **kwargs)
        self.kernel = kernel
        self.output_size = output_size
        self.stride_size = stride_size
        self.dilation = dilation
        self.layer_name = name

    def get_config(self):
        config = {'name':self.layer_name,'kernel':self.kernel,'output_size':self.output_size,'stride_size':self.stride_size,'dilation':self.dilation,'activation':"ReLU"}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self,input_shape):
        self._weights = self.add_weight(name=self.layer_name+"_weights", shape=(self.kernel[0],self.kernel[1],input_shape[3],int(self.output_size)), trainable= True, initializer=tf.keras.initializers.GlorotUniform())
        self.bias = self.add_weight(name=self.layer_name+'_bias',shape=(self.output_size),initializer="zeros",trainable=True)
        super(ODM_Conv2D_Layer, self).build(input_shape)
     
    @tf.function   
    def call(self, inputs):            
        out = tf.nn.conv2d( inputs , self._weights , strides=[ 1 , int(self.stride_size) , int(self.stride_size) , 1 ] ,
                        dilations=[1, int(self.dilation), int(self.dilation), 1], padding="SAME",name=self.layer_name+"_convolution") 
        
        out = tf.nn.bias_add(out, self.bias)
        return tf.keras.activations.relu(out)
   
class ODM_Conv2D_OutputLayer(tf.keras.layers.Layer):
    
    def __init__(self, name, trainable=True, dtype=tf.float32, **kwargs ):
        super(ODM_Conv2D_OutputLayer, self).__init__(name=name,trainable=trainable,dtype=dtype, autocast=False, **kwargs)
        self.layer_name = name

    def get_config(self):
        config = {'name':self.layer_name,'kernel':[1,1],'output_size':8,'stride_size':1,'dilation':1,'activation':"None",'use_bias':False}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self,input_shape):
        self._weights = self.add_weight(name=self.layer_name+"_weights", shape=(1,1,input_shape[3],8), trainable= True, initializer='uniform')
        super(ODM_Conv2D_OutputLayer, self).build(input_shape)
     
    @tf.function   
    def call(self, inputs):            
        return tf.nn.conv2d( inputs , self._weights , strides=[ 1, 1, 1, 1 ] , padding="SAME",name=self.layer_name+"_convolution") 
      
class ODM_MaxPool_Layer(tf.keras.layers.Layer):
    def __init__(self, pool_size, stride_size, name, dtype=tf.float32, **kwargs ):
        super(ODM_MaxPool_Layer, self).__init__(name=name, dtype=tf.float32, autocast=False, **kwargs )
        self.pool_size = pool_size
        self.stride_size = stride_size
        self.layer_name = name

    def get_config(self):
        base_config = super().get_config()
        config = {'name':self.layer_name,'pool_size':self.pool_size,'stride_size':self.stride_size,'padding':'SAME','type':'MaxPool2D'}
        return dict(list(base_config.items()) + list(config.items()))

    def build(self,input_shape):
        super(ODM_MaxPool_Layer, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        return tf.nn.max_pool( inputs , ksize=[ 1 , self.pool_size , self.pool_size , 1 ] ,
                                 padding='SAME' , strides=[ 1 , self.stride_size , self.stride_size , 1 ], name=self.layer_name+'_pool' )

class ObjectDetectionModel(tf.keras.Model):
    
    def __init__(self, kernel_size,name, **kwargs):
        super(ObjectDetectionModel, self).__init__(name=name, dtype=tf.float32, **kwargs )
        self.kernel_size = kernel_size
        self.model_name =name

        # self.layer1 = tf.keras.layers.Conv2D(64, [3,3], [1,1], "Same", dilation_rate=[1,1], activation="relu", kernel_initializer=tf.keras.initializers.GlorotUnifirm(),use_bias=True,trainable=True,name="layer1" )
        self.input_layer = ODM_Input_Layer("input_layer")
        self.layer1 = ODM_Conv2D_Layer(kernel_size,64,1,1,"layer1")
        self.layer2 = ODM_Conv2D_Layer(kernel_size,64,2,2,"layer2")
        self.layer3 = ODM_Conv2D_Layer(kernel_size,128,1,1,"layer3")
        self.layer4 = ODM_Conv2D_Layer(kernel_size,128,1,1,"layer4")
        self.layer5 = ODM_Conv2D_Layer(kernel_size,128,1,3,"layer5")
        self.layer6 = ODM_Conv2D_Layer(kernel_size,128,1,6,"layer6")  
        self.output_2 = ODM_Conv2D_OutputLayer("output_2")
        
        self.layer7 = ODM_Conv2D_Layer(kernel_size,256,2, 1, "layer7")        
        self.layer8 = ODM_Conv2D_Layer(kernel_size,256, 1, 1, "layer8")
        self.layer9 = ODM_Conv2D_Layer(kernel_size,256, 1, 1, "layer9")
        self.layer10 = ODM_Conv2D_Layer(kernel_size,256, 1, 3, "layer10")
        self.output_4 = ODM_Conv2D_OutputLayer("output_4")
        
        self.layer11 = ODM_Conv2D_Layer(kernel_size,512,2, 1, "layer11")
        self.layer12 = ODM_Conv2D_Layer(kernel_size,512, 1, 1, "layer12")
        self.layer13 = ODM_Conv2D_Layer(kernel_size,512, 1, 1, "layer13")
        self.layer14 = ODM_Conv2D_Layer(kernel_size,512, 1, 3, "layer14")
        self.output_8 = ODM_Conv2D_OutputLayer("output_8")
            
        self.layer15 = ODM_Conv2D_Layer(kernel_size,512,2, 1, "layer15")      
        self.layer16 = ODM_Conv2D_Layer(kernel_size,512, 1, 1, "layer16")
        self.layer17 = ODM_Conv2D_Layer(kernel_size,512, 1, 1, "layer17")
        self.layer18 = ODM_Conv2D_Layer(kernel_size,512, 1, 3, "layer18")
        self.output_16 = ODM_Conv2D_OutputLayer( "output_16")
        
       
    def get_config(self):
        layer_configs = []
        for layer in self.layers:
            layer_configs.append({
                'class_name': layer.__class__.__name__,
                'config': layer.get_config()
            })
        config = {
            'name': self.model_name,
            'layers': copy.copy(layer_configs)
        }

        return config

    @tf.function
    def call(self, input, training):
        
        # !!! don't use normalization, not even on the start, it will ruin everything
        x = self.input_layer(input) 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        self.data_2 = self.output_2(x)
        
        # stack trainable variables for optimizers, one optimizer for one network block (scale)
        self.block_2 = self.layer1.trainable_variables 
        self.block_2 = self.block_2 + self.layer2.trainable_variables
        self.block_2 = self.block_2 + self.layer3.trainable_variables
        self.block_2 = self.block_2 + self.layer4.trainable_variables
        self.block_2 = self.block_2 + self.layer5.trainable_variables
        self.block_2 = self.block_2 + self.layer6.trainable_variables
        self.block_2 = self.block_2 + self.output_2.trainable_variables

        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        self.data_4 = self.output_4(x)
        
        self.block_4 = self.layer7.trainable_variables
        self.block_4 = self.block_4 + self.layer8.trainable_variables
        self.block_4 = self.block_4 + self.layer9.trainable_variables
        self.block_4 = self.block_4 + self.layer10.trainable_variables
        self.block_4 = self.block_4 + self.output_4.trainable_variables
        
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        self.data_8 = self.output_8(x)
        
        self.block_8 = self.layer11.trainable_variables
        self.block_8 = self.block_8 + self.layer12.trainable_variables
        self.block_8 = self.block_8 + self.layer13.trainable_variables
        self.block_8 = self.block_8 + self.layer14.trainable_variables
        self.block_8 = self.block_8 + self.output_8.trainable_variables
        
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = self.layer18(x)
        self.data_16 = self.output_16(x)
        
        self.block_16 = self.layer15.trainable_variables
        self.block_16 = self.block_16 + self.layer16.trainable_variables
        self.block_16 = self.block_16 + self.layer17.trainable_variables
        self.block_16 = self.block_16 + self.layer18.trainable_variables
        self.block_16 = self.block_16 + self.output_16.trainable_variables

        return [self.data_2, self.data_4, self.data_8, self.data_16]
    
class NetworkCreator():
    
    def __init__(self):
        self.batch_size = BATCH_SIZE
        
        self.learning = LEARNING_RATE
        
        self.optimizer2 = tf.optimizers.SGD(name="adam_optimizer_2",learning_rate=self.get_learning_rate, momentum=0.9)
        self.optimizer4 = tf.optimizers.SGD(name="adam_optimizer_4",learning_rate=self.get_learning_rate, momentum=0.9)
        self.optimizer8 = tf.optimizers.SGD(name="adam_optimizer_8",learning_rate=self.get_learning_rate, momentum=0.9)
        self.optimizer16 = tf.optimizers.SGD(name="adam_optimizer_16",learning_rate=self.get_learning_rate, momentum=0.9)
        
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
    
    @tf.function
    def compute_gradient(self, model, inputs, targets):
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
        
    def loss_is_nan(self, acc):
        if np.isnan(acc[0]) or np.isnan(acc[1]) or np.isnan(acc[2]) or np.isnan(acc[3]):
            return True
        else:
            return False 
    
    def train(self, loader, model):
        
        tf.keras.backend.set_floatx('float32')
        
        epoch = 1
        acc = [9,9,9,9]
        t_global = Timer()
        t_global.start()

        while self.continue_training(acc):
            epoch_loss_avg = []
            t = Timer()
            t.start()
            for i in range(ITERATIONS):                
                image_batch, labels_batch, _ = loader.get_train_data(self.batch_size) 
                loss_value = self.compute_gradient(model,image_batch, labels_batch)
                self.check_computed_loss(loss_value)
                
                epoch_loss_avg.append(loss_value)

            _ = t.stop()
            acc = np.mean(epoch_loss_avg, axis=0)
            print("Epoch {:d}: Loss 2: {:.6f}, Loss 4: {:.6f}, Loss 8: {:.6f}, Loss 16: {:.6f}  Epoch duration: ".format(epoch,acc[0],acc[1],acc[2],acc[3]) + t.get_formated_time())
            
            self.save_model(model, epoch)
            self.update_learning_rate(epoch)               
            epoch += 1
                           
        _ = t_global.stop()
        self.save_model(model, 0)
    
    def check_computed_loss(self, loss_value):
        if self.loss_is_nan(loss_value):
            print("One of loss values is NaN, program will be terminated!")
            print(loss_value)
            print(names)
            sys.exit()
     
    def update_learning_rate(self, epoch):
        if epoch in UPDATE_LEARNING_RATE:
            self.learning = self.learning / 10

    
    def save_model(self, model, epoch):
        if epoch % SAVE_MODEL_EVERY == 0:
            model.save_weights(MODEL_WEIGHTS)
            
    
     
    def start_train(self, loader):
        
        model = ObjectDetectionModel([3,3],'ObjectDetectionModel')
        
        if CONTINUE_TRAINING:
            if check_file_exists(MODEL_WEIGHTS):
                model.build((self.batch_size,IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS))
                model.load_weights(MODEL_WEIGHTS)
                
        self.train(loader,model)
        model.summary()

                
class Loader():
    def __init__(self):
        self.Data = []
        self.image_path = ''
        self.bb3_path = ''
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
            
        if check_file_exists(BB3_PATH):
            self.bb3_path = BB3_PATH
        else:
            print("Label path '"+ BB3_PATH +"' not found!!!")  
            
            
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
        assert self.bb3_path != '', 'Label path not set. Nothing to work with. Check config file.'
        print('Loading training files')
        
        image_pathss = []
        bb3_paths = []
        
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
            bb3_path = self.bb3_path + '/' + file_name + '.txt'
            if not check_file_exists(bb3_path):
                continue
                    
            
            image, width, height = self._load_image(image_path)
            if image is None:
                continue
            
            # label
            labels = self._load_label(bb3_path)
            if labels is None or len(labels) == 0:
                continue
          
            data = DataModel()
            data.image = image
            data.image_path = image_path
            data.image_name = file_
            data.labels = labels

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
        # normalized = normalize(resized)
        return resized, im.shape[1], im.shape[0]

    def _load_label(self, file_path):
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

        result = self._load_from_bb3_folder(file_path)
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
    
    def load_specific_label(self, file_name):
            
        image_path = IMAGE_PATH + '/' + file_name + '.png'
        bb3_path = BB3_PATH + '/' + file_name + '.txt'
        
        if not check_file_exists(image_path):
            assert "Path for image not found"
        
        if not check_file_exists(bb3_path):
            assert "Path for label not found"
                    
        image, width, height = self._load_image(image_path)
        labels = self._load_label( bb3_path)
        
        
        data = DataModel()
        data.image = image
        data.image_path = image_path
        data.image_name = file_name
        data.labels = labels

        self.Data.append(data) 
    
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
                re = int(radius/2) 

                for k in range(1,8):
                    for l in range(-radius - re,radius + re,1):
                        for j in range(-radius - re,radius + re,1):
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


loader = Loader()

loader.load_data()
# loader.load_specific_label('000008')
print("loader done")
nc = NetworkCreator()
with tf.device('/device:GPU:0'):
  nc.start_train(loader)



# LABEL_PATH = 'drive/My Drive/object detection/colab_data/label'
# BB3_FOLDER = 'drive/My Drive/object detection/colab_data/bb3'
MODEL_WEIGHTS = 'drive/My Drive/object detection/colab_data/model_weights.h5'
# MODEL_JSON = 'drive/My Drive/object detection/colab_data/model_spec.json'

SAVE_MODEL_EVERY = 10
UPDATE_LEARNING_RATE = [100, 500, 1000]
CONTINUE_TRAINING = True

#percent
RADIUS = 2
CIRCLE_RATIO = 0.3
BOUNDARIES = 0.33

# amount of data from dataset which will be loaded
# -1 for all data
DATA_AMOUNT = 100

IMAGE_EXTENSION = 'png'

# NETWORK RELATED SETTINGS
IS_TRAINING = True
IMG_WIDTH = 256
IMG_HEIGHT = 128
IMG_CHANNELS = 3 # based on channel number, image will be loaded colored or grayscaled

LEARNING_RATE = 0.00001
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
    def object_loss(self, target, image):
        width = image.shape.dims[1].value
        height = image.shape.dims[0].value
        channels = image.shape.dims[2].value
        
        # number of neurons in output layer
        N_p = tf.math.count_nonzero(target[:, :, 0])  
     
        initial = tf.constant(1,dtype=tf.float32, shape=(height,width))
        tmp_initial = initial
        condition = tf.greater(target[:,:, 0], tf.constant(0,dtype=tf.float32),name="greater")
        weight_factor_array = tf.add(initial, tf.where(condition, (tmp_initial + self.weight_factor - 1), tmp_initial, name="where_condition"), name="assign" )

        error = tf.reduce_sum(tf.multiply(weight_factor_array, tf.square(tf.subtract(target[:,:, 0], image[:, :, 0]))))
        second_error = 0.0
        for c in range(1, channels):
            second_error += tf.reduce_sum(
                tf.multiply(self.weight_factor,
                             tf.multiply(target[:, :, 0],
                                          tf.square(tf.subtract(target[:, :, c], image[:, :, c])))))
        
                    
        N = width * height
        error = (1/(2*N))*error 
        
        # there are no objects in some scales and N_p will be 0 so to prevent division by 0
        # sec_error = tf.cast(1/ (2 * N_p * (channels -1)),dtype=tf.float32)*second_error
        sec_error = tf.cond(tf.equal(N_p,0),lambda: tf.constant(0,dtype=tf.float32, shape=()),lambda: tf.cast(1/ (2 * N_p * (channels -1)),dtype=tf.float32)*second_error )            
        error += sec_error
        return tf.cast(error, dtype=tf.float32)

class ODM_Input_Layer(tf.keras.layers.Layer):
    
    def __init__(self, name, dtype=tf.float32, **kwargs ):
        super(ODM_Input_Layer, self).__init__(name=name,trainable=False,dtype=dtype,autocast=False, **kwargs)
        self.layer_name = name

    def get_config(self):
        config = {'name':self.layer_name}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self,input_shape):
        super(ODM_Input_Layer, self).build(input_shape)
     
    @tf.function   
    def call(self, inputs):    
        _min = tf.reduce_min(inputs)
        _max = tf.reduce_max(inputs)
        
        result = tf.divide(tf.subtract(inputs,_min),tf.subtract(_max,_min))
        result = tf.cast( result , dtype=tf.float32 ) 
        return result

class ODM_Conv2D_Layer(tf.keras.layers.Layer):
    
    def __init__(self, kernel, output_size, stride_size, dilation, name, activation=True,trainable=True,dtype=tf.float32, **kwargs ):
        super(ODM_Conv2D_Layer, self).__init__(name=name,trainable=trainable,dtype=dtype,autocast=False, **kwargs)
        self.kernel = kernel
        self.output_size = output_size
        self.stride_size = stride_size
        self.dilation = dilation
        self.layer_name = name

    def get_config(self):
        config = {'name':self.layer_name,'kernel':self.kernel,'output_size':self.output_size,'stride_size':self.stride_size,'dilation':self.dilation,'activation':"ReLU"}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self,input_shape):
        self._weights = self.add_weight(name=self.layer_name+"_weights", shape=(self.kernel[0],self.kernel[1],input_shape[3],int(self.output_size)), trainable= True, initializer=tf.keras.initializers.GlorotUniform())
        self.bias = self.add_weight(name=self.layer_name+'_bias',shape=(self.output_size),initializer="zeros",trainable=True)
        super(ODM_Conv2D_Layer, self).build(input_shape)
     
    @tf.function   
    def call(self, inputs):            
        out = tf.nn.conv2d( inputs , self._weights , strides=[ 1 , int(self.stride_size) , int(self.stride_size) , 1 ] ,
                        dilations=[1, int(self.dilation), int(self.dilation), 1], padding="SAME",name=self.layer_name+"_convolution") 
        
        out = tf.nn.bias_add(out, self.bias)
        return tf.keras.activations.relu(out)
   
class ODM_Conv2D_OutputLayer(tf.keras.layers.Layer):
    
    def __init__(self, name, trainable=True, dtype=tf.float32, **kwargs ):
        super(ODM_Conv2D_OutputLayer, self).__init__(name=name,trainable=trainable,dtype=dtype, autocast=False, **kwargs)
        self.layer_name = name

    def get_config(self):
        config = {'name':self.layer_name,'kernel':[1,1],'output_size':8,'stride_size':1,'dilation':1,'activation':"None",'use_bias':False}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self,input_shape):
        self._weights = self.add_weight(name=self.layer_name+"_weights", shape=(1,1,input_shape[3],8), trainable= True, initializer='uniform')
        super(ODM_Conv2D_OutputLayer, self).build(input_shape)
     
    @tf.function   
    def call(self, inputs):            
        return tf.nn.conv2d( inputs , self._weights , strides=[ 1, 1, 1, 1 ] , padding="SAME",name=self.layer_name+"_convolution") 
      
class ODM_MaxPool_Layer(tf.keras.layers.Layer):
    def __init__(self, pool_size, stride_size, name, dtype=tf.float32, **kwargs ):
        super(ODM_MaxPool_Layer, self).__init__(name=name, dtype=tf.float32, autocast=False, **kwargs )
        self.pool_size = pool_size
        self.stride_size = stride_size
        self.layer_name = name

    def get_config(self):
        base_config = super().get_config()
        config = {'name':self.layer_name,'pool_size':self.pool_size,'stride_size':self.stride_size,'padding':'SAME','type':'MaxPool2D'}
        return dict(list(base_config.items()) + list(config.items()))

    def build(self,input_shape):
        super(ODM_MaxPool_Layer, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        return tf.nn.max_pool( inputs , ksize=[ 1 , self.pool_size , self.pool_size , 1 ] ,
                                 padding='SAME' , strides=[ 1 , self.stride_size , self.stride_size , 1 ], name=self.layer_name+'_pool' )

class ObjectDetectionModel(tf.keras.Model):
    
    def __init__(self, kernel_size,name, **kwargs):
        super(ObjectDetectionModel, self).__init__(name=name, dtype=tf.float32, **kwargs )
        self.kernel_size = kernel_size
        self.model_name =name

        # self.layer1 = tf.keras.layers.Conv2D(64, [3,3], [1,1], "Same", dilation_rate=[1,1], activation="relu", kernel_initializer=tf.keras.initializers.GlorotUnifirm(),use_bias=True,trainable=True,name="layer1" )
        self.input_layer = ODM_Input_Layer("input_layer")
        self.layer1 = ODM_Conv2D_Layer(kernel_size,64,1,1,"layer1")
        self.layer2 = ODM_Conv2D_Layer(kernel_size,64,2,2,"layer2")
        self.layer3 = ODM_Conv2D_Layer(kernel_size,128,1,1,"layer3")
        self.layer4 = ODM_Conv2D_Layer(kernel_size,128,1,1,"layer4")
        self.layer5 = ODM_Conv2D_Layer(kernel_size,128,1,3,"layer5")
        self.layer6 = ODM_Conv2D_Layer(kernel_size,128,1,6,"layer6")  
        self.output_2 = ODM_Conv2D_OutputLayer("output_2")
        
        self.layer7 = ODM_Conv2D_Layer(kernel_size,256,2, 1, "layer7")        
        self.layer8 = ODM_Conv2D_Layer(kernel_size,256, 1, 1, "layer8")
        self.layer9 = ODM_Conv2D_Layer(kernel_size,256, 1, 1, "layer9")
        self.layer10 = ODM_Conv2D_Layer(kernel_size,256, 1, 3, "layer10")
        self.output_4 = ODM_Conv2D_OutputLayer("output_4")
        
        self.layer11 = ODM_Conv2D_Layer(kernel_size,512,2, 1, "layer11")
        self.layer12 = ODM_Conv2D_Layer(kernel_size,512, 1, 1, "layer12")
        self.layer13 = ODM_Conv2D_Layer(kernel_size,512, 1, 1, "layer13")
        self.layer14 = ODM_Conv2D_Layer(kernel_size,512, 1, 3, "layer14")
        self.output_8 = ODM_Conv2D_OutputLayer("output_8")
            
        self.layer15 = ODM_Conv2D_Layer(kernel_size,512,2, 1, "layer15")      
        self.layer16 = ODM_Conv2D_Layer(kernel_size,512, 1, 1, "layer16")
        self.layer17 = ODM_Conv2D_Layer(kernel_size,512, 1, 1, "layer17")
        self.layer18 = ODM_Conv2D_Layer(kernel_size,512, 1, 3, "layer18")
        self.output_16 = ODM_Conv2D_OutputLayer( "output_16")
        
       
    def get_config(self):
        layer_configs = []
        for layer in self.layers:
            layer_configs.append({
                'class_name': layer.__class__.__name__,
                'config': layer.get_config()
            })
        config = {
            'name': self.model_name,
            'layers': copy.copy(layer_configs)
        }

        return config

    @tf.function
    def call(self, input, training):
        
        # !!! don't use normalization, not even on the start, it will ruin everything
        x = self.input_layer(input) 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        self.data_2 = self.output_2(x)
        
        # stack trainable variables for optimizers, one optimizer for one network block (scale)
        self.block_2 = self.layer1.trainable_variables 
        self.block_2 = self.block_2 + self.layer2.trainable_variables
        self.block_2 = self.block_2 + self.layer3.trainable_variables
        self.block_2 = self.block_2 + self.layer4.trainable_variables
        self.block_2 = self.block_2 + self.layer5.trainable_variables
        self.block_2 = self.block_2 + self.layer6.trainable_variables
        self.block_2 = self.block_2 + self.output_2.trainable_variables

        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        self.data_4 = self.output_4(x)
        
        self.block_4 = self.layer7.trainable_variables
        self.block_4 = self.block_4 + self.layer8.trainable_variables
        self.block_4 = self.block_4 + self.layer9.trainable_variables
        self.block_4 = self.block_4 + self.layer10.trainable_variables
        self.block_4 = self.block_4 + self.output_4.trainable_variables
        
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        self.data_8 = self.output_8(x)
        
        self.block_8 = self.layer11.trainable_variables
        self.block_8 = self.block_8 + self.layer12.trainable_variables
        self.block_8 = self.block_8 + self.layer13.trainable_variables
        self.block_8 = self.block_8 + self.layer14.trainable_variables
        self.block_8 = self.block_8 + self.output_8.trainable_variables
        
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = self.layer18(x)
        self.data_16 = self.output_16(x)
        
        self.block_16 = self.layer15.trainable_variables
        self.block_16 = self.block_16 + self.layer16.trainable_variables
        self.block_16 = self.block_16 + self.layer17.trainable_variables
        self.block_16 = self.block_16 + self.layer18.trainable_variables
        self.block_16 = self.block_16 + self.output_16.trainable_variables

        return [self.data_2, self.data_4, self.data_8, self.data_16]
    
class NetworkCreator():
    
    def __init__(self):
        self.batch_size = BATCH_SIZE
        
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
    
    @tf.function
    def compute_gradient(self, model, inputs, targets):
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
        
    def loss_is_nan(self, acc):
        if np.isnan(acc[0]) or np.isnan(acc[1]) or np.isnan(acc[2]) or np.isnan(acc[3]):
            return True
        else:
            return False 
    
    def train(self, loader, model):
        
        tf.keras.backend.set_floatx('float32')
        
        epoch = 1
        acc = [9,9,9,9]
        t_global = Timer()
        t_global.start()

        while self.continue_training(acc):
            epoch_loss_avg = []
            t = Timer()
            t.start()
            for i in range(ITERATIONS):                
                image_batch, labels_batch, _ = loader.get_train_data(self.batch_size) 
                loss_value = self.compute_gradient(model,image_batch, labels_batch)
                self.check_computed_loss(loss_value)
                
                epoch_loss_avg.append(loss_value)

            _ = t.stop()
            acc = np.mean(epoch_loss_avg, axis=0)
            print("Epoch {:d}: Loss 2: {:.6f}, Loss 4: {:.6f}, Loss 8: {:.6f}, Loss 16: {:.6f}  Epoch duration: ".format(epoch,acc[0],acc[1],acc[2],acc[3]) + t.get_formated_time())
            
            self.save_model(model, epoch)
            self.update_learning_rate(epoch)               
            epoch += 1
                           
        _ = t_global.stop()
        self.save_model(model, 0)
    
    def check_computed_loss(self, loss_value):
        if self.loss_is_nan(loss_value):
            print("One of loss values is NaN, program will be terminated!")
            print(loss_value)
            print(names)
            sys.exit()
     
    def update_learning_rate(self, epoch):
        if epoch in UPDATE_LEARNING_RATE:
            self.learning = self.learning / 10

    
    def save_model(self, model, epoch):
        if epoch % SAVE_MODEL_EVERY == 0:
            model.save_weights(MODEL_WEIGHTS)
            
    
     
    def start_train(self, loader):
        
        model = ObjectDetectionModel([3,3],'ObjectDetectionModel')
        
        if CONTINUE_TRAINING:
            if check_file_exists(MODEL_WEIGHTS):
                model.build((self.batch_size,IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS))
                model.load_weights(MODEL_WEIGHTS)
                
        self.train(loader,model)
        model.summary()

                
class Loader():
    def __init__(self):
        self.Data = []
        self.image_path = ''
        self.bb3_path = ''
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
            
        if check_file_exists(BB3_PATH):
            self.bb3_path = BB3_PATH
        else:
            print("Label path '"+ BB3_PATH +"' not found!!!")  
            
            
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
        assert self.bb3_path != '', 'Label path not set. Nothing to work with. Check config file.'
        print('Loading training files')
        
        image_pathss = []
        bb3_paths = []
        
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
            bb3_path = self.bb3_path + '/' + file_name + '.txt'
            if not check_file_exists(bb3_path):
                continue
                    
            
            image, width, height = self._load_image(image_path)
            if image is None:
                continue
            
            # label
            labels = self._load_label(bb3_path)
            if labels is None or len(labels) == 0:
                continue
          
            data = DataModel()
            data.image = image
            data.image_path = image_path
            data.image_name = file_
            data.labels = labels

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
        # normalized = normalize(resized)
        return resized, im.shape[1], im.shape[0]

    def _load_label(self, file_path):
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

        result = self._load_from_bb3_folder(file_path)
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
    
    def load_specific_label(self, file_name):
            
        image_path = IMAGE_PATH + '/' + file_name + '.png'
        bb3_path = BB3_PATH + '/' + file_name + '.txt'
        
        if not check_file_exists(image_path):
            assert "Path for image not found"
        
        if not check_file_exists(bb3_path):
            assert "Path for label not found"
                    
        image, width, height = self._load_image(image_path)
        labels = self._load_label( bb3_path)
        
        
        data = DataModel()
        data.image = image
        data.image_path = image_path
        data.image_name = file_name
        data.labels = labels

        self.Data.append(data) 
    
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


loader = Loader()

loader.load_data()
# loader.load_specific_label('000008')
print("loader done")
nc = NetworkCreator()
with tf.device('/device:GPU:0'):
  nc.start_train(loader)


