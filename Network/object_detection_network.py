import config as cfg
import cv2
import numpy as np
import copy

import tensorflow as tf

class ActivityRegularizationLayer(tf.keras.layers.Layer):
    
    def __init__(self, rate=1e-2):
        super(ActivityRegularizationLayer, self).__init__()
        self.rate = rate

    # @tf.function
    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_sum(inputs))
        return inputs

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
        self.kernel = self.add_weight(name=self.layer_name+"_weights", shape=(self.kernel[0],self.kernel[1],input_shape[3],int(self.output_size)), trainable= True, initializer=tf.keras.initializers.GlorotUniform())
        self.bias = self.add_weight(name='bias',shape=(self.output_size),initializer="zeros",trainable=True)
        super(ODM_Conv2D_Layer, self).build(input_shape)
        
    @tf.function    
    def call(self, inputs):            
        out = tf.nn.conv2d( inputs , self.kernel , strides=[ 1 , int(self.stride_size) , int(self.stride_size) , 1 ] ,
                        dilations=[1, int(self.dilation), int(self.dilation), 1], padding="SAME",name=self.layer_name+"_convolution") 
        
        out = tf.nn.bias_add(out, self.bias)
        return tf.keras.activations.relu(out)
   
class ODM_Conv2D_OutputLayer(tf.keras.layers.Layer):
    
    def __init__(self, name, trainable=True, dtype=tf.float32, **kwargs ):
        super(ODM_Conv2D_OutputLayer, self).__init__(name=name,trainable=trainable,dtype=dtype, **kwargs)
        self.layer_name = name

    def get_config(self):
        config = {'name':self.name}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self,input_shape):
        self.kernel = self.add_weight(name=self.layer_name+"_weights", shape=(1,1,input_shape[3],8), trainable= True, initializer='uniform')
        super(ODM_Conv2D_OutputLayer, self).build(input_shape)
        
    @tf.function    
    def call(self, inputs):            
        out = tf.nn.conv2d( inputs , self.kernel , strides=[ 1, 1, 1, 1 ] , padding="SAME",name=self.layer_name+"_convolution") 
        return out
      
class ODM_MaxPool_Layer(tf.keras.layers.Layer):
    def __init__(self, pool_size, stride_size, name, **kwargs ):
        super(ODM_MaxPool_Layer, self).__init__(name=name, **kwargs )
        self.pool_size = pool_size
        self.stride_size = stride_size
        self.layer_name = name

    def get_config(self):
        base_config = super().get_config()
        config = {'pool_size':self.pool_size,'stride_size':self.stride_size,'name':self.name}
        return dict(list(base_config.items()) + list(config.items()))

    def build(self,input_shape):
        super(ODM_MaxPool_Layer, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        return tf.nn.max_pool( inputs , ksize=[ 1 , self.pool_size , self.pool_size , 1 ] ,
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
        self.output_2 = ODM_Conv2D_OutputLayer("output_2")
        self.max_pool_layer_2 = ODM_MaxPool_Layer(2, 2, "layer7_maxpool")
        
        #self.block_2 = self.layer1.trainable_variables + self.layer2.trainable_variables + self.layer3.trainable_variables + self.layer4.trainable_variables + self.layer5.trainable_variables + self.layer6.trainable_variables + self.output_2.trainable_variables
        
        
        self.layer8 = ODM_Conv2D_Layer(kernel_size,256, 1, 1, "layer8")
        self.layer8_normal = tf.keras.layers.BatchNormalization(name="layer8_normalization")
        self.layer9 = ODM_Conv2D_Layer(kernel_size,256, 1, 1, "layer9")
        self.layer9_normal = tf.keras.layers.BatchNormalization(name="layer9_normalization")
        self.layer10 = ODM_Conv2D_Layer(kernel_size,256, 1, 3, "layer10")
        self.layer10_normal = tf.keras.layers.BatchNormalization(name="layer10_normalization")
        self.output_4 = ODM_Conv2D_OutputLayer("output_4")
        self.max_pool_layer_4 = ODM_MaxPool_Layer(2, 2, "layer11_maxpool")
        
        #self.block_4 = self.layer8.trainable_variables + self.layer9.trainable_variables + self.layer10.trainable_variables + self.output_4.trainable_variables

        self.layer12 = ODM_Conv2D_Layer(kernel_size,512, 1, 1, "layer12")
        self.layer12_normal = tf.keras.layers.BatchNormalization(name="layer11_normalization")
        self.layer13 = ODM_Conv2D_Layer(kernel_size,512, 1, 1, "layer13")
        self.layer13_normal = tf.keras.layers.BatchNormalization(name="layer12_normalization")
        self.layer14 = ODM_Conv2D_Layer(kernel_size,512, 1, 3, "layer14")
        self.layer14_normal = tf.keras.layers.BatchNormalization(name="layer13_normalization")
        self.output_8 = ODM_Conv2D_OutputLayer("output_8")
        self.max_pool_layer_8 = ODM_MaxPool_Layer(2, 2, "layer15_maxpool")
        
        #self.block_8 = self.layer12.trainable_variables + self.layer13.trainable_variables + self.layer14.trainable_variables + self.output_8.trainable_variables
            
        self.layer16 = ODM_Conv2D_Layer(kernel_size,512, 1, 1, "layer16")
        self.layer16_normal = tf.keras.layers.BatchNormalization(name="layer16_normalization")
        self.layer17 = ODM_Conv2D_Layer(kernel_size,512, 1, 1, "layer17")
        self.layer17_normal = tf.keras.layers.BatchNormalization(name="layer17_normalization")
        self.layer18 = ODM_Conv2D_Layer(kernel_size,512, 1, 3, "layer18")
        self.layer18_normal = tf.keras.layers.BatchNormalization(name="layer18_normalization")
        self.output_16 = ODM_Conv2D_OutputLayer( "output_16")
        
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
        
        self.block_2 = self.layer1.trainable_variables + self.layer1_normal.trainable_variables
        self.block_2 = self.block_2 + self.layer2.trainable_variables + self.layer2_normal.trainable_variables
        self.block_2 = self.block_2 + self.layer3.trainable_variables + self.layer3_normal.trainable_variables
        self.block_2 = self.block_2 + self.layer4.trainable_variables + self.layer4_normal.trainable_variables
        self.block_2 = self.block_2 + self.layer5.trainable_variables + self.layer5_normal.trainable_variables
        self.block_2 = self.block_2 + self.layer6.trainable_variables + self.layer6_normal.trainable_variables
        self.block_2 = self.block_2 + self.output_2.trainable_variables

        x = self.max_pool_layer_2(x)
        x = self.layer8(x)
        x = self.layer8_normal(x, training)
        x = self.layer9(x)
        x = self.layer9_normal(x, training)
        x = self.layer10(x)
        x = self.layer10_normal(x, training)
        self.data_4 = self.output_4(x)
        
        self.block_4 = self.max_pool_layer_2.trainable_variables
        self.block_4 = self.block_4 + self.layer8.trainable_variables + self.layer8_normal.trainable_variables 
        self.block_4 = self.block_4 + self.layer9.trainable_variables + self.layer9_normal.trainable_variables 
        self.block_4 = self.block_4 + self.layer10.trainable_variables + self.layer10_normal.trainable_variables
        self.block_4 = self.block_4 + self.output_4.trainable_variables
        
        x = self.max_pool_layer_4(x)
        x = self.layer12(x)
        x = self.layer12_normal(x, training)
        x = self.layer13(x)
        x = self.layer13_normal(x, training)
        x = self.layer14(x)
        x = self.layer14_normal(x, training)
        self.data_8 = self.output_8(x)
        
        self.block_8 = self.max_pool_layer_4.trainable_variables
        self.block_8 = self.block_8 + self.layer12.trainable_variables + self.layer12_normal.trainable_variables
        self.block_8 = self.block_8 + self.layer13.trainable_variables + self.layer13_normal.trainable_variables
        self.block_8 = self.block_8 + self.layer14.trainable_variables + self.layer14_normal.trainable_variables
        self.block_8 = self.block_8 + self.output_8.trainable_variables
        
        x = self.max_pool_layer_8(x)
        x = self.layer16(x)
        x = self.layer16_normal(x, training)
        x = self.layer17(x)
        x = self.layer17_normal(x, training)
        x = self.layer18(x)
        x = self.layer18_normal(x, training)
        self.data_16 = self.output_16(x)
        
        self.block_16 = self.max_pool_layer_8.trainable_variables
        self.block_16 = self.block_16 + self.layer16.trainable_variables + self.layer16_normal.trainable_variables
        self.block_16 = self.block_16 + self.layer17.trainable_variables + self.layer17_normal.trainable_variables
        self.block_16 = self.block_16 + self.layer18.trainable_variables + self.layer18_normal.trainable_variables
        self.block_16 = self.block_16 + self.output_16.trainable_variables

        return [self.data_2, self.data_4, self.data_8, self.data_16]

    
class ObjectDetectionModel2(tf.keras.Model):
    
    def __init__(self, kernel_size,name, **kwargs):
        super(ObjectDetectionModel2, self).__init__(name=name, **kwargs )
        self.kernel_size = kernel_size
        self.model_name =name

        self.input_normal = tf.keras.layers.BatchNormalization(name="input_normalization")

        self.layer1 = tf.keras.layers.Conv2D(64, [3,3], [1,1], "Same", dilation_rate=[1,1], activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(),use_bias=True,trainable=True,name="layer1" )
        self.layer1_normal = tf.keras.layers.BatchNormalization(name="layer1_normalization")
        self.layer2 = tf.keras.layers.Conv2D(64, [3,3], [2,2], "Same", dilation_rate=[1,1], activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(),use_bias=True,trainable=True,name="layer2" )
        self.layer2_normal = tf.keras.layers.BatchNormalization(name="layer2_normalization")
        self.layer3 = tf.keras.layers.Conv2D(128, [3,3], [1,1], "Same", dilation_rate=[1,1], activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(),use_bias=True,trainable=True,name="layer3" )
        self.layer3_normal = tf.keras.layers.BatchNormalization(name="layer3_normalization")
        self.layer4 = tf.keras.layers.Conv2D(128, [3,3], [1,1], "Same", dilation_rate=[1,1], activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(),use_bias=True,trainable=True,name="layer4" )
        self.layer4_normal = tf.keras.layers.BatchNormalization(name="layer4_normalization")
        self.layer5 = tf.keras.layers.Conv2D(128, [3,3], [1,1], "Same", dilation_rate=[3,3], activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(),use_bias=True,trainable=True,name="layer5" )
        self.layer5_normal = tf.keras.layers.BatchNormalization(name="layer5_normalization")
        self.layer6 = tf.keras.layers.Conv2D(128, [3,3], [1,1], "Same", dilation_rate=[6,6], activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(),use_bias=True,trainable=True,name="layer6" )
        self.layer6_normal = tf.keras.layers.BatchNormalization(name="layer6_normalization")   

        self.output_2 = tf.keras.layers.Conv2D(8, [1,1], [1,1], "Same", dilation_rate=[1,1], activation=None, kernel_initializer=tf.keras.initializers.GlorotUniform(),use_bias=False,trainable=True,name="output_2" )
        self.max_pool_layer_2 = tf.keras.layers.MaxPool2D([2,2], [2,2],"Same", "channels_last")


        self.layer8 = tf.keras.layers.Conv2D(256, [3,3], [1,1], "Same", dilation_rate=[1,1], activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(),use_bias=True,trainable=True,name="layer8" )
        self.layer8_normal = tf.keras.layers.BatchNormalization(name="layer8_normalization")
        self.layer9 = tf.keras.layers.Conv2D(256, [3,3], [1,1], "Same", dilation_rate=[1,1], activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(),use_bias=True,trainable=True,name="layer9" )
        self.layer9_normal = tf.keras.layers.BatchNormalization(name="layer9_normalization")
        self.layer10 = tf.keras.layers.Conv2D(256, [3,3], [1,1], "Same", dilation_rate=[3,3], activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(),use_bias=True,trainable=True,name="layer10" )
        self.layer10_normal = tf.keras.layers.BatchNormalization(name="layer10_normalization")

        self.output_4 = tf.keras.layers.Conv2D(8, [1,1], [1,1], "Same", dilation_rate=[1,1], activation=None, kernel_initializer=tf.keras.initializers.GlorotUniform(),use_bias=False,trainable=True,name="output_4" )
        self.max_pool_layer_4 = tf.keras.layers.MaxPool2D([2,2], [2,2],"Same", "channels_last")
        
        self.layer12 = tf.keras.layers.Conv2D(512, [3,3], [1,1], "Same", dilation_rate=[1,1], activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(),use_bias=True,trainable=True,name="layer12" )
        self.layer12_normal = tf.keras.layers.BatchNormalization(name="layer11_normalization")
        self.layer13 = tf.keras.layers.Conv2D(512, [3,3], [1,1], "Same", dilation_rate=[1,1], activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(),use_bias=True,trainable=True,name="layer13" )
        self.layer13_normal = tf.keras.layers.BatchNormalization(name="layer12_normalization")
        self.layer14 = tf.keras.layers.Conv2D(512, [3,3], [1,1], "Same", dilation_rate=[3,3], activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(),use_bias=True,trainable=True,name="layer14" )
        self.layer14_normal = tf.keras.layers.BatchNormalization(name="layer13_normalization")
        
        self.output_8 = tf.keras.layers.Conv2D(8, [1,1], [1,1], "Same", dilation_rate=[1,1], activation=None, kernel_initializer=tf.keras.initializers.GlorotUniform(),use_bias=False,trainable=True,name="output_8" )
        self.max_pool_layer_8 = tf.keras.layers.MaxPool2D([2,2], [2,2],"Same", "channels_last")
        
            
        self.layer16 = tf.keras.layers.Conv2D(512, [3,3], [1,1], "Same", dilation_rate=[1,1], activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(),use_bias=True,trainable=True,name="layer16" )
        self.layer16_normal = tf.keras.layers.BatchNormalization(name="layer16_normalization")
        self.layer17 = tf.keras.layers.Conv2D(512, [3,3], [1,1], "Same", dilation_rate=[1,1], activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(),use_bias=True,trainable=True,name="layer17" )
        self.layer17_normal = tf.keras.layers.BatchNormalization(name="layer17_normalization")
        self.layer18 = tf.keras.layers.Conv2D(512, [3,3], [1,1], "Same", dilation_rate=[3,3], activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(),use_bias=True,trainable=True,name="layer18" )
        self.layer18_normal = tf.keras.layers.BatchNormalization(name="layer18_normalization")

        self.output_16 = tf.keras.layers.Conv2D(8, [1,1], [1,1], "Same", dilation_rate=[1,1], activation=None, kernel_initializer=tf.keras.initializers.GlorotUniform(),use_bias=False,trainable=True,name="output_16" )

       
    @tf.function
    def call(self, input, training):
        
        x = tf.cast( input , dtype=tf.float32 ) 
        #x = self.input_normal(x,training)
        x = self.layer1(x)
        #x = self.layer1_normal(x, training)
        x = self.layer2(x)
        #x = self.layer2_normal(x, training)
        x = self.layer3(x)
        #x = self.layer3_normal(x, training)
        x = self.layer4(x)
        # x = self.layer4_normal(x, training)
        x = self.layer5(x)
        #x = self.layer5_normal(x, training)
        x = self.layer6(x)
        # x = self.layer6_normal(x, training)
        self.data_2 = self.output_2(x)
        
        self.block_2 = self.layer1.trainable_variables# + self.layer1_normal.trainable_variables
        self.block_2 = self.block_2 + self.layer2.trainable_variables #+ self.layer2_normal.trainable_variables
        self.block_2 = self.block_2 + self.layer3.trainable_variables #+ self.layer3_normal.trainable_variables
        self.block_2 = self.block_2 + self.layer4.trainable_variables #+ self.layer4_normal.trainable_variables
        self.block_2 = self.block_2 + self.layer5.trainable_variables #+ self.layer5_normal.trainable_variables
        self.block_2 = self.block_2 + self.layer6.trainable_variables #+ self.layer6_normal.trainable_variables
        self.block_2 = self.block_2 + self.output_2.trainable_variables

        x = self.max_pool_layer_2(x)
        x = self.layer8(x)
        #x = self.layer8_normal(x, training)
        x = self.layer9(x)
        #x = self.layer9_normal(x, training)
        x = self.layer10(x)
        # x = self.layer10_normal(x, training)
        self.data_4 = self.output_4(x)
        
        self.block_4 = self.layer8.trainable_variables #+ self.layer8_normal.trainable_variables 
        self.block_4 = self.block_4 + self.layer9.trainable_variables #+ self.layer9_normal.trainable_variables 
        self.block_4 = self.block_4 + self.layer10.trainable_variables #+ self.layer10_normal.trainable_variables
        self.block_4 = self.block_4 + self.output_4.trainable_variables
        
        x = self.max_pool_layer_4(x)
        x = self.layer12(x)
        #x = self.layer12_normal(x, training)
        x = self.layer13(x)
        #x = self.layer13_normal(x, training)
        x = self.layer14(x)
        # x = self.layer14_normal(x, training)
        self.data_8 = self.output_8(x)
        
        self.block_8 = self.layer12.trainable_variables# + self.layer12_normal.trainable_variables
        self.block_8 = self.block_8 + self.layer13.trainable_variables #+ self.layer13_normal.trainable_variables
        self.block_8 = self.block_8 + self.layer14.trainable_variables #+ self.layer14_normal.trainable_variables
        self.block_8 = self.block_8 + self.output_8.trainable_variables
        
        x = self.max_pool_layer_8(x)
        x = self.layer16(x)
        #x = self.layer16_normal(x, training)
        x = self.layer17(x)
        #x = self.layer17_normal(x, training)
        x = self.layer18(x)
        # x = self.layer18_normal(x, training)
        self.data_16 = self.output_16(x)
        
        self.block_16 = self.layer16.trainable_variables #+ self.layer16_normal.trainable_variables
        self.block_16 = self.block_16 + self.layer17.trainable_variables# + self.layer17_normal.trainable_variables
        self.block_16 = self.block_16 + self.layer18.trainable_variables #+ self.layer18_normal.trainable_variables
        self.block_16 = self.block_16 + self.output_16.trainable_variables

        return [self.data_2, self.data_4, self.data_8, self.data_16]