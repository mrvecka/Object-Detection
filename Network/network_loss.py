__date__   = '14/05/2020'
__author__ = 'Lukas Mrvecka'
__email__  = 'lukas.mrvecka.st@vsb.cz'
__source__ = 'http://libornovak.com/files/master_thesis.pdf'

import tensorflow as tf
import config as cfg
import cv2
import numpy as np
from tensorflow.python.framework import ops

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
    def call(self, y_true, y_pred):
        # y_pred = ops.convert_to_tensor(y_pred)
        # y_true = ops.convert_to_tensor(y_true)
        
        y_pred = tf.cast(y_pred,tf.float32)
        y_true = tf.cast(y_true,tf.float32)
        # tf.config.experimental_run_functions_eagerly(True)
        loss = self.run_for_scale(y_true,y_pred)
        # tf.config.experimental_run_functions_eagerly(False)
        return loss
        
    @tf.function
    def run_for_scale(self,y_true,y_pred):
        errors = []
        for i in range(y_pred.shape[0]):          
            current_img = y_pred[i]
            current_lbl = y_true[i]
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
     

           
