__date__   = '14/05/2020'
__author__ = 'Lukas Mrvecka'
__email__  = 'lukas.mrvecka.st@vsb.cz'

import os
import sys
import math

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import tensorflow as tf
import cv2
import numpy as np
import Services.loader as load
import config as cfg


class NetworkLoss(tf.keras.losses.Loss):
    def __init__(self, loss_name, scale, reduction=tf.keras.losses.Reduction.AUTO):
        super().__init__(reduction=reduction, name=loss_name)
        self.radius = cfg.RADIUS
        self.circle_ratio = cfg.CIRCLE_RATIO
        self.boundaries = cfg.BOUNDARIES
        self.weight_factor = cfg.WEIGHT_FACTOR
        self.batch_size = 1
        self.scale = scale
    
    def get_config(self):
        base_config = super().get_config()
        config = {'batch':self.batch,'scale':self.scale,'loss_name':self.loss_name,'reduction':tf.keras.losses.Reduction.AUTO}
        return dict(list(base_config.items()) + list(config.items()))
    
    @tf.function
    def call(self, labels, outputs):
        # tf.config.experimental_run_functions_eagerly(True)
        outputs = tf.cast(outputs,tf.float32)
        loss = self.run_for_scale(outputs,labels,self.radius, self.circle_ratio, self.boundaries,self.batch_size,self.weight_factor,self.scale)
        # loss_4 = self.run_for_scale(outputs[1],labels,self.radius, self.circle_ratio, self.boundaries,self.batch_size,self.weight_factor,4)
        # loss_8 = self.run_for_scale(outputs[2],labels,self.radius, self.circle_ratio, self.boundaries,self.batch_size,self.weight_factor,8)
        # loss_16 = self.run_for_scale(outputs[3],labels,self.radius, self.circle_ratio, self.boundaries,self.batch_size,self.weight_factor,16)
        # tf.config.experimental_run_functions_eagerly(False)        
        # tf.reduce_sum([loss_2,loss_4,loss_8, loss_16],name="global_loss")
        return loss
        
    
    @tf.function
    def run_for_scale(self,images,labels, radius,circle_ratio,boundaries,batch,weight_factor,scale):
        errors = []
        for i in range(batch):          
            current_img = images[i]
            current_lbl = labels[i]
            img_error = self.scan_image_function(current_img, current_lbl,radius,circle_ratio,boundaries,scale,weight_factor)
            errors.append(img_error)

        errors_as_tensor = tf.convert_to_tensor(errors,dtype=tf.float32)
        loss = tf.reduce_sum(errors_as_tensor)
        tf.config.experimental_run_functions_eagerly(False)
        return loss
    
    @tf.function
    def scan_image_function(self, image, label,radius,circle_ratio,boundaries,scale,weight_factor):

        width = image.shape.dims[1].value
        height = image.shape.dims[0].value
        channels = image.shape.dims[2].value
        weight_factor = tf.cast(weight_factor,tf.float32)
        

        target = tf.numpy_function(self.create_target_response_map, [label, width, height, radius, circle_ratio, boundaries, scale], tf.float32)
            
        # target = self.create_target_response_map(label, width, height,radius, circle_ratio, boundaries,scale)        
        target = tf.cast(tf.reshape(target,(channels,height,width)),tf.float32)  

        
        # number of neurons in each output layer
        N = width * height

        N_p = tf.math.count_nonzero(image[:, :, 0])      
        second_error = 0
        error = 0.0
                   
        # get array of weight factor with the same shape as target 
        initial = tf.constant(1,dtype=tf.float32, shape=(height,width))
        tmp_initial = initial
        condition = tf.greater(target[0,:, :], tf.constant(0,dtype=tf.float32),name="greater")
        weight_factor_array = tf.add(initial, tf.where(condition, (tmp_initial + weight_factor - 1), tmp_initial, name="where_condition"), name="assign" )

        error = tf.reduce_sum(tf.multiply(weight_factor_array, tf.square(tf.subtract(target[0,:, :], image[:, :, 0]))))
        for c in range(1, channels):
            second_error += tf.reduce_sum(
                tf.multiply(weight_factor,
                             tf.multiply(target[0, :, :],
                                          tf.square(tf.subtract(target[c,:, :], image[:, :, c])))))
        
                    
        error = (1/(2*N))*error     
        tmp = 1/ (3 * N_p * (channels -1))    
        error += tf.cast(tmp, tf.float32) * second_error
       
        return error
    
    def GetObjectBounds(self,radius,circle_ratio,boundaries,scale):
        ideal_size = (2.0 * radius + 1.0) / circle_ratio * scale
        # bound above
        ext_above = ((1.0 - boundaries) * ideal_size) / 2.0 + boundaries * ideal_size
        bound_above = ideal_size + ext_above
        
        # bound below
        diff = ideal_size / 2.0
        ext_below = ((1 - boundaries)* diff) / 2.0 + boundaries * diff
        bound_below = ideal_size - ext_below
        
        return tf.cast(bound_above,tf.float32), tf.cast(bound_below,tf.float32), tf.cast(ideal_size,tf.float32)
    
    def create_target_response_map(self, labels, width, height, radius,circle_ratio,boundaries,scale):
                
        maps = cv2.split(np.zeros((height,width,8)))
        

        
        bound_above, bound_below, ideal = self.GetObjectBounds(radius,circle_ratio,boundaries,scale)
        for i in range(labels.shape[0]):            
            label = labels[i]
            if label[0] == -1:
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
        
        return np.asarray(maps,dtype=np.float32)


def test_loss(loader):
    
    image_batch, label_batch, image_paths,image_names, calib_matrices = loader.get_test_data(1)

    dummy = np.zeros((1,128,64,8))
    loss = NetworkLoss("loss2", 2)
    loss_value = loss(label_batch, dummy)
    
    
    # target_2 = create_target_response_map(label_batch[0], 128, 64,8,cfg.RADIUS, cfg.CIRCLE_RATIO, cfg.BOUNDARIES,2)        
    # target_2 = tf.reshape(target_2,(8,64,128))  
     
    # target_4 = create_target_response_map(label_batch[0], 64, 32,8,cfg.RADIUS, cfg.CIRCLE_RATIO, cfg.BOUNDARIES,4)        
    # target_4 = tf.reshape(target_4,(8,32,64)) 
    
    # target_8 = create_target_response_map(label_batch[0], 32, 16,8,cfg.RADIUS, cfg.CIRCLE_RATIO, cfg.BOUNDARIES,8)        
    # target_8 = tf.reshape(target_8,(8,16,32)) 
    
    # target_16 = create_target_response_map(label_batch[0], 16, 8,8,cfg.RADIUS, cfg.CIRCLE_RATIO, cfg.BOUNDARIES,16)        
    # target_16 = tf.reshape(target_16,(8,8,16)) 
    
    # cv2.imshow("target 2", target_2.numpy()[0,:,:])
    # cv2.imshow("target 4", target_4.numpy()[0,:,:])
    # cv2.imshow("target 8", target_8.numpy()[0,:,:])
    # cv2.imshow("target 16", target_16.numpy()[0,:,:])

    # cv2.waitKey(0)

if __name__ == "__main__":
    loader = load.Loader()
    loader.load_specific_label("000013")
    test_loss(loader)