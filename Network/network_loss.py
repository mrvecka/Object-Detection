import tensorflow as tf
import config as cfg
import cv2
import numpy as np

from Network.object_detection_network import ObjectDetectionModel

# class NetworkFakeLoss(tf.keras.losses.Loss):
#         def __init__(self, loss_name, scale, reduction=tf.keras.losses.Reduction.AUTO):
#         super().__init__(reduction=reduction, name=loss_name)
#         self.weight_factor = cfg.WEIGHT_FACTOR
#         self.scale = scale
    
#     def get_config(self):
#         base_config = super().get_config()
#         config = {'scale':self.scale,'loss_name':self.loss_name,'reduction':tf.keras.losses.Reduction.AUTO}
#         return dict(list(base_config.items()) + list(config.items()))
    
#     @tf.function
#     def call(self, labels, outputs):
#         tf.config.experimental_run_functions_eagerly(True)
#         labels = tf.cast(labels,tf.float32)
#         outputs = tf.cast(outputs,tf.float32)
#         loss = self.run_for_scale(outputs,labels,self.weight_factor)
        
#         tf.config.experimental_run_functions_eagerly(False)        
#         return loss

class NetworkLoss(tf.keras.losses.Loss):
    def __init__(self, loss_name, scale, reduction=tf.keras.losses.Reduction.AUTO):
        super().__init__(reduction=reduction, name=loss_name)
        self.weight_factor = cfg.WEIGHT_FACTOR
        self.scale = scale
    
    def get_config(self):
        base_config = super().get_config()
        config = {'scale':self.scale,'loss_name':self.loss_name,'reduction':tf.keras.losses.Reduction.AUTO}
        return dict(list(base_config.items()) + list(config.items()))
    
    @tf.function
    def call(self, labels, outputs):
        tf.config.experimental_run_functions_eagerly(True)
        labels = tf.cast(labels,tf.float32)
        outputs = tf.cast(outputs,tf.float32)
        loss = self.run_for_scale(outputs,labels,self.weight_factor)
        
        tf.config.experimental_run_functions_eagerly(False)        
        return loss
        
    
    @tf.function
    def run_for_scale(self,images,labels,weight_factor):
        errors = []
        for i in range(images.shape[0]):          
            current_img = images[i]
            current_lbl = labels[i]
            img_error = self.scan_image_function(current_img, current_lbl,weight_factor)
            errors.append(img_error)

        loss = tf.reduce_sum(errors)
        return loss
    
    @tf.function
    def none_object_loss(self, target, image):
        N_p = tf.math.count_nonzero(image)  

        error = tf.square(tf.subtract(target,image))
        error = tf.cast((1 / (4 * N_p)), tf.float32) * error
        return error
     
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
    
    @tf.function
    def scan_image_function(self, image, target,weight_factor):
                
        T_p = tf.math.count_nonzero(target[:,:,0])
        return self.object_loss(target,image)
        # return tf.cond(tf.equal(T_p,0), lambda: self.none_object_loss(target,image), lambda: self.object_loss(target,image))
           
