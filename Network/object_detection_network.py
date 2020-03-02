import config as cfg
import cv2
import numpy as np
import copy

import tensorflow as tf


class ODM_Conv2D_Layer(tf.keras.layers.Layer):
    
    def __init__(self, kernel, output_size, stride_size, dilation, name, activation=True,trainable=True,dtype=tf.float32 ):
        super(ODM_Conv2D_Layer, self).__init__(name=name,trainable=trainable,dtype=dtype)
        self.kernel = kernel
        self.output_size = output_size
        self.stride_size = stride_size
        self.dilation = dilation
        self.layer_name = name
        self.activation = activation
        
        self.initializer = tf.initializers.glorot_uniform()

    def get_config(self):
        config = {'kernel':self.kernel,'output_size':self.output_size,'stride_size':self.stride_size,'dilation':self.dilation,'name':self.name,'activation':self.activation}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self,input_shape):
        self.kernel = tf.Variable( self.initializer( [self.kernel[0],self.kernel[1],input_shape[3],self.output_size] ),
                                   name=self.layer_name+'_weight' , trainable=True , dtype=tf.float32 )
        # self.add_variable(self.name,shape=[self.kernel[0],self.kernel[1],self.input.shape[3],self.output_size])

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

def get_object_detection_model(kernel_size):
    
    inputs = tf.keras.Input(shape=(cfg.IMG_HEIGHT,cfg.IMG_WIDTH,cfg.IMG_CHANNELS))
    x = ODM_Conv2D_Input_Layer(kernel_size,64,1,1,"layer1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="layer1_normalization")(x)
    x = ODM_Conv2D_Layer(kernel_size,64,2,1,"layer2")(x)
    x = tf.keras.layers.BatchNormalization(name="layer2_normalization")(x)
    x = ODM_Conv2D_Layer(kernel_size,128,1,1,"layer3")(x)
    x = tf.keras.layers.BatchNormalization(name="layer3_normalization")(x)
    x = ODM_Conv2D_Layer(kernel_size,128,1,1,"layer4")(x)
    x = tf.keras.layers.BatchNormalization(name="layer4_normalization")(x)
    x = ODM_Conv2D_Layer(kernel_size,128,1,3,"layer5")(x)
    x = tf.keras.layers.BatchNormalization(name="layer5_normalization")(x)
    x = ODM_Conv2D_Layer(kernel_size,128,1,6,"layer6")(x)
    x = tf.keras.layers.BatchNormalization(name="layer6_normalization")(x)
    output_2 = ODM_Conv2D_Layer([1,1],8,1,1,"output_2", False)(x)
    x = ODM_MaxPool_Layer(2, 2, "layer7")(x)
    
    #out_2_trainable_variables = layer1.trainable_variables + layer2.trainable_variables + layer3.trainable_variables + layer4.trainable_variables + layer5.trainable_variables + layer6.trainable_variables + output_2.trainable_variables
      
    x = ODM_Conv2D_Layer(kernel_size,256, 1, 1, "layer8")(x)
    x = tf.keras.layers.BatchNormalization(name="layer8_normalization")(x)
    x = ODM_Conv2D_Layer(kernel_size,256, 1, 1, "layer9")(x)
    x = tf.keras.layers.BatchNormalization(name="layer9_normalization")(x)
    x = ODM_Conv2D_Layer(kernel_size,256, 1, 3, "layer10")(x)
    x = tf.keras.layers.BatchNormalization(name="layer10_normalization")(x)
    output_4 = ODM_Conv2D_Layer([1,1],8, 1, 1, "output_4", False)(x)
    x = ODM_MaxPool_Layer(2, 2, "layer11")(x)
    
    #out_4_trainable_variables = layer8.trainable_variables + layer9.trainable_variables + layer10.trainable_variables + output_4.trainable_variables

    x = ODM_Conv2D_Layer(kernel_size,512, 1, 1, "layer12")(x)
    x = tf.keras.layers.BatchNormalization(name="layer11_normalization")(x)
    x = ODM_Conv2D_Layer(kernel_size,512, 1, 1, "layer13")(x)
    x = tf.keras.layers.BatchNormalization(name="layer12_normalization")(x)
    x = ODM_Conv2D_Layer(kernel_size,512, 1, 3, "layer14")(x)
    x = tf.keras.layers.BatchNormalization(name="layer13_normalization")(x)
    output_8 = ODM_Conv2D_Layer([1,1], 8, 1, 1, "output_8", False)(x)
    x = ODM_MaxPool_Layer(2, 2, "layer15")(x)
    
    #out_8_trainable_variables = layer12.trainable_variables + layer13.trainable_variables + layer14.trainable_variables + output_8.trainable_variables
    
    x = ODM_Conv2D_Layer(kernel_size,512, 1, 1, "layer16")(x)
    x = tf.keras.layers.BatchNormalization(name="layer16_normalization")(x)
    x = ODM_Conv2D_Layer(kernel_size,512, 1, 1, "layer17")(x)
    x = tf.keras.layers.BatchNormalization(name="layer17_normalization")(x)
    x = ODM_Conv2D_Layer(kernel_size,512, 1, 3, "layer18")(x)
    x = tf.keras.layers.BatchNormalization(name="layer18_normalization")(x)
    output_16 = ODM_Conv2D_Layer([1,1],8, 1, 1, "output_16", False)(x)
    
    #out_16_trainable_variables = layer16.trainable_variables + layer17.trainable_variables + layer18.trainable_variables + output_16.trainable_variables
    
    return tf.keras.Model(inputs = inputs, outputs = [output_2,output_4,output_8,output_16]) #, [out_2_trainable_variables,out_4_trainable_variables,out_8_trainable_variables,out_16_trainable_variables]

class ObjectDetectionModel(tf.keras.Model):
    
    def __init__(self, kernel_size,name):
        super(ObjectDetectionModel, self).__init__(name=name)
        self.kernel_size = kernel_size
        self.model_name =name
        self._build_input_shape = (None,cfg.IMG_HEIGHT,cfg.IMG_WIDTH, cfg.IMG_CHANNELS)
        # inputs = tf.keras.Input(shape=[cfg.IMG_HEIGHT,cfg.IMG_WIDTH,cfg.IMG_CHANNELS])
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
       
    def get_config(self):
        layer_configs = []
        for layer in self.layers:
            layer_configs.append({
                'class_name': layer.__class__.__name__,
                'config': layer.get_config()
            })
        config = {
            'name': self.model_name,
            'layers': copy.copy(layer_configs),
            "kernel_size": self.kernel_size
        }

        return config

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