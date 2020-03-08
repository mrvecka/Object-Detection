from __future__ import print_function
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
import tensorflow as tf
# from Network.network_creator import NetworkCreator
# import Models.bb3txt as bb
import config as cfg
import cv2
import Services.loader as load
import Services.helper as h

class NetworkLoss():
    def __init__(self,batch, scale, loss_name, reduction=tf.keras.losses.Reduction.AUTO):
        self.radius = cfg.RADIUS
        self.circle_ratio = cfg.CIRCLE_RATIO
        self.boundaries = cfg.BOUNDARIES
        self.weight_factor = cfg.WEIGHT_FACTOR
        self.scale = scale
        self.batch_size = batch
      
    def GetObjectBounds(self):
        
        ideal_size = tf.divide(tf.add(tf.multiply(2.0, self.radius),1.0), tf.multiply(self.circle_ratio, self.scale))
        ext_above = tf.divide(tf.multiply(tf.subtract(1.0, self.boundaries),ideal_size), tf.add(2.0,tf.multiply(self.boundaries, ideal_size)))
        bound_above = tf.add(ideal_size,ext_above)
        
        diff = tf.divide(ideal_size, 2.0)
        ext_below = tf.divide(tf.multiply(tf.subtract(1.0, self.boundaries),diff), tf.add(2.0,tf.multiply(self.boundaries, diff)))
        bound_below = tf.subtract(ideal_size, ext_below)
        
        # ideal_size = (2.0 * self.radius + 1.0) / self.circle_ratio * self.scale
        # # bound above
        # ext_above = ((1.0 - self.boundaries) * ideal_size) / 2.0 + self.boundaries * ideal_size
        # bound_above = ideal_size + ext_above
        
        # # bound below
        # diff = ideal_size / 2.0
        # ext_below = ((1 - self.boundaries)* diff) / 2.0 + self.boundaries * diff
        # bound_below = ideal_size - ext_below
        
        return bound_above, bound_below, ideal_size
    
    def scan_image_function(self, image, label):
    
        width = image.shape.dims[1].value
        height = image.shape.dims[0].value
        channels = image.shape.dims[2].value
        
        print(tf.autograph.to_code(self.create_target_response_map.python_function))
        target = self.create_target_response_map(label, width, height)
        
        target = tf.reshape(target,(channels,height,width))        
        
        # number of neurons in each output layer
        N = width * height

        N_p = tf.math.count_nonzero(image[:, :, 0])      
                   
        # get array of weight factor with the same shape as target 
        initial = tf.constant(1,dtype=tf.float32, shape=(height,width))
        tmp_initial = initial
        condition = tf.greater(target[0,:, :], tf.constant(0,dtype=tf.float32),name="greater")
        weight_factor_array = tf.add(initial, tf.where(condition, (tmp_initial + self.weight_factor - 1), tmp_initial, name="where_condition"), name="assign" )

        error = tf.reduce_sum(tf.multiply(weight_factor_array, tf.square(tf.subtract(target[0,:, :], image[:, :, 0]))))
        
        # second_error = tf.constant(0,dtype=tf.float32)
        c = tf.constant(1,dtype=tf.int16)
        def error_condition(c,second_e):
            return tf.less(c,channels)
        
        def error_body(c,second_e):
            tmp_error = tf.reduce_sum(
                tf.multiply(self.weight_factor,
                             tf.multiply(target[0, :, :],
                                          tf.square(tf.subtract(target[c,:, :], image[:, :, c])))))
            second_e = tf.add(second_e,tmp_error)
            c = tf.add(c,1)
            return [c,second_e]
            
        c,second_error = tf.while_loop(error_condition,error_body,[c,0])
        
        # for c in range(1, channels):
        #     second_error += tf.reduce_sum(
        #         tf.multiply(self.weight_factor,
        #                      tf.multiply(target[0, :, :],
        #                                   tf.square(tf.subtract(target[c,:, :], image[:, :, c])))))
               
        del target  
        del initial    

        error = tf.multiply(error,tf.divide(1,tf.multiply(2,N)))
        tmp = tf.divide(1,tf.multiply(3,tf.multiply(N_p,tf.subtract(channels,1))))
        error = tf.add( error, tf.multiply( tf.cast(tmp, tf.float32), second_error))
               
        return error
    
        
    @tf.function
    def create_target_response_map(self, labels, width, height):
                
        maps = cv2.split(np.zeros((height,width,8)))
        bound_above, bound_below, ideal = self.GetObjectBounds()
        for i in range(labels.shape[0]):            
            label = labels[i]
            
            if label[0] == -1:
                continue
            # 0       1       2       3       4       5       6     7           8           9
            # fblx    fbly    fbrx    fbry    rblx    rbly    ftly  center_x    center_y    largest_dim
            
            def label_bound_condition(label):
                return tf.logical_and(tf.greater_equal(tf.gather(label,9),bound_below),tf.less_equal(tf.gather(label,9),bound_above))
            
            def label_bound_body(label,maps):
                x = int(label[7] / self.scale)
                y = int(label[8] / self.scale)
                
                scaling_ratio = 1.0 / self.scale                
                cv2.circle(maps[0], ( x, y ), int(self.radius), 1, -1)
                cv2.GaussianBlur(maps[0], (3, 3), 100)

                c_ = tf.constant(1,dtype=int32)
                def channels_condition(c_,maps):
                    return tf.less(c_,8)
                
                def channels_body(c_,maps):
                    for l in range(-self.radius,self.radius,1):
                        for j in range(-self.radius,self.radius,1):
                            xp = x + j
                            yp = y + l
                            
                            if xp >= 0 and xp < width and yp >= 0 and yp < height:
                                if maps[0][yp][xp] > 0.0:
                                    if c_ ==1 or c_ == 3 or c_ == 5:
                                        maps[c_][yp][xp] = 0.5 + (label[c_-1] - x - j * self.scale) / ideal
                                    elif c_ == 2 or c_ == 4 or c_ == 6 or c_ == 7:
                                        maps[c_][yp][xp] = 0.5 + (label[c_-1] - y - l * self.scale) / ideal

                c_, maps = tf.while_loop(channels_condition,channels_body,[c_,maps])
                return maps
            
            maps = tf.cond(label_bound_condition(label), lambda: label_bound_body(label,maps), lambda: maps)
            
            # if label[9] >= bound_below and label[9] <= bound_above:
            #     x = int(label[7] / self.scale)
            #     y = int(label[8] / self.scale)
                
            #     scaling_ratio = 1.0 / self.scale                
            #     cv2.circle(maps[0], ( x, y ), int(self.radius), 1, -1)
            #     cv2.GaussianBlur(maps[0], (3, 3), 100)

            #     c_ = tf.constant(1,dtype=int32)
            #     def channels_condition(c_,maps):
            #         return tf.less(c_,8)
                
            #     def channels_body(c_,maps):
            #         for l in range(-self.radius,self.radius,1):
            #             for j in range(-self.radius,self.radius,1):
            #                 xp = x + j
            #                 yp = y + l
                            
            #                 if xp >= 0 and xp < width and yp >= 0 and yp < height:
            #                     if maps[0][yp][xp] > 0.0:
            #                         if c_ ==1 or c_ == 3 or c_ == 5:
            #                             maps[c_][yp][xp] = 0.5 + (label[c_-1] - x - j * self.scale) / ideal
            #                         elif c_ == 2 or c_ == 4 or c_ == 6 or c_ == 7:
            #                             maps[c_][yp][xp] = 0.5 + (label[c_-1] - y - l * self.scale) / ideal

            #     c_, maps = tf.while_loop(channels_condition,channels_body,[c_,maps])
                # for c_ in range(1,8):
                    
                #     for l in range(-self.radius,self.radius,1):
                #         for j in range(-self.radius,self.radius,1):
                #             xp = x + j
                #             yp = y + l
                            
                #             if xp >= 0 and xp < width and yp >= 0 and yp < height:
                #                 if maps[0][yp][xp] > 0.0:
                #                     if c_ ==1 or c_ == 3 or c_ == 5:
                #                         maps[c_][yp][xp] = 0.5 + (label[c_-1] - x - j * self.scale) / ideal
                #                     elif c_ == 2 or c_ == 4 or c_ == 6 or c_ == 7:
                #                         maps[c_][yp][xp] = 0.5 + (label[c_-1] - y - l * self.scale) / ideal
        
        return np.asarray(maps,dtype=np.float32)


            
def test_target_map_creation(name):
    loader = load.Loader()
    loader.load_specific_label(name)

    image_batch, label, image_paths,image_names, calib_matrices = loader.get_test_data(1)


    loss_2_model = NetworkLoss(1, 2.0, "loss_scale_2")
    loss_2_result = loss_2_model.create_target_response_map(label[0],128,64)
                
    loss_4_model = NetworkLoss(1, 4.0, "loss_scale_4")
    loss_4_result = loss_4_model.create_target_response_map(label[0],64,32)

    loss_8_model = NetworkLoss(1, 8.0, "loss_scale_8")
    loss_8_result = loss_8_model.create_target_response_map(label[0],32,16)

    loss_16_model = NetworkLoss(1, 16.0, "loss_scale_16")
    loss_16_result = loss_16_model.create_target_response_map(label[0],16,8) 
    
    
    # target = tf.py_func(create_target_response_map, [label[0], 128, 64, 8, 2,0.3, 0.33, 2], [tf.float32])
    # target2 = tf.reshape(target,(8,64,128))
    
    # tmp = tf.py_func(create_target_response_map, [label[0], 64, 32, 8, 2,0.3, 0.33, 4], [tf.float32])
    # target4 = tf.reshape(tmp,(8,32,64)) 
    
    # tmp = tf.py_func(create_target_response_map, [label[0], 32, 16, 8, 2,0.3, 0.33, 8], [tf.float32])
    # target8 = tf.reshape(tmp,(8,16,32)) 
    
    # tmp = tf.py_func(create_target_response_map, [label[0], 16, 8, 8, 2,0.3, 0.33, 16], [tf.float32])
    # target16 = tf.reshape(tmp,(8,8,16)) 
    

    

    
    # maps = create_target_response_map(labels_batch[0], 128, 64, 8, 2,0.3, 0.33, 4)
    # print(maps.shape)
    # tmp = h.change_first_x_last_dim(maps)
    
    # cv2.imshow("target 1", maps[0,:,:])
    # cv2.imshow("target 2", maps[1,:,:])
    # cv2.imshow("target 3", maps[2,:,:])
    # cv2.imshow("target 4", maps[3,:,:])
    # cv2.imshow("target 5", maps[4,:,:])
    # cv2.imshow("target 6", maps[5,:,:])
    # cv2.imshow("target 7", maps[6,:,:])
    # cv2.imshow("target 8", maps[7,:,:])
    
    # cv2.imshow("target 11", tmp[:,:,0])
    # cv2.imshow("target 22", tmp[:,:,1])
    # cv2.imshow("target 33", tmp[:,:,2])
    # cv2.imshow("target 44", tmp[:,:,3])
    # cv2.imshow("target 55", tmp[:,:,4])
    # cv2.imshow("target 66", tmp[:,:,5])
    # cv2.imshow("target 77", tmp[:,:,6])
    # cv2.imshow("target 88", tmp[:,:,7])
    
    # cv2.waitKey(0)
                
    #         cv2.waitKey(0)
    # tmp = np.reshape(maps,(8,16,8))
    # print(tmp)
    # with tf.Session() as session:
    #     # initialise the variables

    #     session.run(init)

    #     tmp,weight = session.run([target2,weight_factor_array], feed_dict={label: labels_batch})
    #     maps = h.change_first_x_last_dim(tmp)
    #     cv2.imshow("target 2", maps[:,:,0])
    #     cv2.imshow("weights", weight*127)
    #     cv2.waitKey(0)

    #     cv2.imwrite(r'C:\Users\Lukas\Documents\Object detection\test\target2.jpg', 255*maps[:,:,0])
    #     resized = cv2.resize(maps[:,:,0], (1242,375), interpolation = cv2.INTER_AREA)
    #     cv2.imshow("resized", resized)
    #     cv2.imwrite(r'C:\Users\Lukas\Documents\Object detection\test\resized2.jpg', 255*resized)
    #     # cv2.imshow("target 2", maps[:,:,1])
    #     # cv2.imshow("target 3", maps[:,:,2])
    #     # cv2.imshow("target 4", maps[:,:,3])
    #     # cv2.imshow("target 5", maps[:,:,4])
    #     # cv2.imshow("target 6", maps[:,:,5])
    #     # cv2.imshow("target 7", maps[:,:,6])
    #     # cv2.imshow("target 8", maps[:,:,7])
               
    #     tmp2 = session.run(target4, feed_dict={label: labels_batch})
    #     maps = h.change_first_x_last_dim(tmp2)
    #     cv2.imshow("target 4", maps[:,:,0])
    #     cv2.imwrite(r'C:\Users\Lukas\Documents\Object detection\test\target4.jpg', 255*maps[:,:,0])
    #     resized4 = cv2.resize(maps[:,:,0], (1242,375), interpolation = cv2.INTER_AREA)
    #     cv2.imshow("resized4", resized4)
    #     cv2.imwrite(r'C:\Users\Lukas\Documents\Object detection\test\resized4.jpg', 255*resized4)

        
    #     tmp4 = session.run(target8, feed_dict={label: labels_batch})
    #     maps = h.change_first_x_last_dim(tmp4)
    #     cv2.imshow("target 8", maps[:,:,0])
    #     cv2.imwrite(r'C:\Users\Lukas\Documents\Object detection\test\target8.jpg', 255*maps[:,:,0])
    #     resized8 = cv2.resize(maps[:,:,0], (1242,375), interpolation = cv2.INTER_AREA)
    #     cv2.imshow("resized8", resized8)
    #     cv2.imwrite(r'C:\Users\Lukas\Documents\Object detection\test\resized8.jpg', 255*resized8)

                
    #     tmp8 = session.run(target16, feed_dict={label: labels_batch})
    #     maps = h.change_first_x_last_dim(tmp8)
    #     cv2.imshow("target 16", maps[:,:,0])
    #     cv2.imwrite(r'C:\Users\Lukas\Documents\Object detection\test\target16.jpg', 255*maps[:,:,0])
    #     resized16 = cv2.resize(maps[:,:,0], (1242,375), interpolation = cv2.INTER_AREA)
    #     cv2.imshow("resized6", resized16)
    #     cv2.imwrite(r'C:\Users\Lukas\Documents\Object detection\test\resized16.jpg', 255*resized16)

            
    #     cv2.waitKey(0)
            
            # print("Epoch:", (epoch + 1), "test error: {:.5f}".format(test_acc), " learning rate: ",learning)

            # saver.save(session, cfg.MODEL_PATH)
    # maps2 = create_target_response_map(labels_batch[0], 2, 0.3, 0.33, 2,64,128)
    # maps4 = create_target_response_map(labels_batch[0], 2, 0.3, 0.33, 4, 32,64)
    # maps8 = create_target_response_map(labels_batch[0], 2, 0.3, 0.33, 8, 16,32)
    # maps16 = create_target_response_map(labels_batch[0], 2, 0.3, 0.33, 16, 8,16)

    # first2 = maps2[:,:,0]
    # second2 = maps2[:,:,1]
    # first4 = maps4[:,:,0]
    # first8 = maps8[:,:,0]
    # first16 = maps16[:,:,0]
    # cv2.imshow("response_2", first2)
    # cv2.imshow("response_2 second",second2)
    # cv2.imshow("response_4", first4)
    # cv2.imshow("response_8", first8)
    # cv2.imshow("response_16", first16)
    # cv2.waitKey(0)


if __name__ == "__main__":    
    test_target_map_creation("000046") 
    
