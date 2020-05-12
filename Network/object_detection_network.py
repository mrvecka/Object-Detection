import config as cfg
import cv2
import numpy as np
import copy

import tensorflow as tf

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
    
    def __init__(self, kernel_size, name, batch_size,  **kwargs):
        super(ObjectDetectionModel, self).__init__(name=name, dtype=tf.float32, **kwargs )
        self.kernel_size = kernel_size
        self.model_name =name

        # self.layer1 = tf.keras.layers.Conv2D(64, [3,3], [1,1], "Same", dilation_rate=[1,1], activation="relu", kernel_initializer=tf.keras.initializers.GlorotUnifirm(),use_bias=True,trainable=True,name="layer1" )
        self.input_layer = ODM_Input_Layer("input_layer", input_shape=(batch_size,cfg.IMG_HEIGHT,cfg.IMG_WIDTH,cfg.IMG_CHANNELS))       
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

        return self.data_2, self.data_4, self.data_8, self.data_16
