from __future__ import print_function
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
import tensorflow as tf
import tensorboard as tb
# from Network.network_creator import NetworkCreator
# import Models.bb3txt as bb
import cv2
import Services.loader as load
import Services.helper as h

def GetObjectBounds(r, cr, bo, scale):
    ideal_size = (2 * r + 1) / cr * scale
    # bound above
    ext_above = ((1 - bo) * ideal_size) / 2 + bo * ideal_size
    bound_above = ideal_size + ext_above
    
    # bound below
    diff = ideal_size / 2
    ext_below = ((1 - bo)* diff) /2 + bo * diff
    bound_below = ideal_size - ext_below
    
    return bound_above, bound_below, ideal_size

def create_target_response_map( labels, width, height, channels, r, circle_ratio, boundaries, scale):
                
    # maps => array of shape (channels, orig_height, orig_width) 
    maps = cv2.split(np.zeros((height,width,8)))
    # self.index = 0
    # result = tf.scan(self.scan_label_function, labels, initializer=0) 
    bound_above, bound_below, ideal = GetObjectBounds(r,circle_ratio,boundaries,scale)
    for i in range(len(labels)):            
        label = labels[i]
        if label[0] == -1:
            break
        # 0       1       2       3       4       5       6     7           8           9
        # fblx    fbly    fbrx    fbry    rblx    rbly    ftly  center_x    center_y    largest_dim
        
        #size = self.get_size_of_bounding_box(labels)
        if label[9] >= bound_below and label[9] <= bound_above:
            x = int(label[7] / scale)
            y = int(label[8] / scale)
            
            scaling_ratio = 1.0 / scale
            # print((self.orig_height,self.orig_width))
            #radius = ((circle_ration / scale) * szie ) - 1
            
            cv2.circle(maps[0], (x, y), int(r), 1, -1)
            cv2.GaussianBlur(maps[0], (3, 3), 100)

            # x_acc = x * scaling_ratio
            # y_acc = y * scaling_ratio
            
            # apply densebox method
            dense_label = label[0:7]
            dense_label[0] = x - dense_label[0]
            dense_label[2] = x - dense_label[2]
            dense_label[4] = x - dense_label[4]
            dense_label[1] = y - dense_label[1]
            dense_label[3] = y - dense_label[3]
            dense_label[5] = y - dense_label[5]
            dense_label[6] = y - dense_label[6]
            
            #normalize to range <0,1>
            dense_label = h.normalize(dense_label)
            
            for c in range(1,8):
                
                for l in range(-r,r,1):
                    for j in range(-r,r,1):
                        xp = x + j
                        yp = y + l
                        
                        if xp >= 0 and xp < width and yp >= 0 and yp < height:
                            if maps[0][yp][xp] > 0.0:
                                if c == 1 or c == 3 or c == 5:
                                    maps[c][yp][xp] = 100
                                    #maps[c][yp][xp] = dense_label[c-1]
                                elif c == 2 or c == 4 or c == 6 or c == 7:
                                    maps[c][yp][xp] = 100
                                    #maps[c][yp][xp] = dense_label[c-1]

    result = cv2.merge(maps)
    
    return np.asarray(maps,dtype=np.float32)


            
def test_target_map_creation(name):
    loader = load.Loader()
    loader.load_specific_label(name)

    label = tf.placeholder(tf.float32,(None,None,10), name="label")

    target = tf.py_func(create_target_response_map, [label[0], 128, 64, 8, 2,0.3, 0.33, 4], [tf.float32])
    target = tf.reshape(target,(8,64,128))    
    
    init = tf.global_variables_initializer()
    errors = [2.0,2.0]
    
    loss_output = tf.placeholder(dtype=tf.float32,shape=(2),name='loss_output_placeholder')
    loss_constant = tf.constant(1.0,shape=[2],dtype=tf.float32,name='loss_constant')
    loss_output = tf.multiply(errors,loss_constant, name='loss_output_multiply')
    

    image_batch, labels_batch, image_paths, calib_matrices = loader.get_test_data(1)
    
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
    with tf.Session() as session:
        # initialise the variables

        session.run(init)

        tmp = session.run(target, feed_dict={label: labels_batch})
        maps = h.change_first_x_last_dim(tmp)
        cv2.imshow("target 1", maps[:,:,0])
        cv2.imshow("target 2", maps[:,:,1])
        cv2.imshow("target 3", maps[:,:,2])
        cv2.imshow("target 4", maps[:,:,3])
        cv2.imshow("target 5", maps[:,:,4])
        cv2.imshow("target 6", maps[:,:,5])
        cv2.imshow("target 7", maps[:,:,6])
        cv2.imshow("target 8", maps[:,:,7])
            
        cv2.waitKey(0)
            
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
    
