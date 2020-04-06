
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
            
            for c in range(1,8):
                
                for l in range(-r,r,1):
                    for j in range(-r,r,1):
                        xp = x + j
                        yp = y + l
                        
                        if xp >= 0 and xp < width and yp >= 0 and yp < height:
                            if maps[0][yp][xp] > 0.0:
                                if c ==1 or c == 3 or c == 5:
                                    maps[c][yp][xp] = 0.5 + (label[c-1] - x - j * scale) / ideal
                                elif c == 2 or c == 4 or c == 6 or c == 7:
                                    maps[c][yp][xp] = 0.5 + (label[c-1] - y - l * scale) / ideal

    
    return np.asarray(maps,dtype=np.float32)


            
def test_target_map_creation(name):
    loader = load.Loader()
    loader.load_specific_label(name)

    # target = create_target_response_map(label[0], 128, 64, 8, 2,0.3, 0.33, 2)
    # target2 = tf.reshape(target,(8,64,128))
    
    # tmp = tf.py_func(create_target_response_map, [label[0], 64, 32, 8, 2,0.3, 0.33, 4], [tf.float32])
    # target4 = tf.reshape(tmp,(8,32,64)) 
    
    # tmp = tf.py_func(create_target_response_map, [label[0], 32, 16, 8, 2,0.3, 0.33, 8], [tf.float32])
    # target8 = tf.reshape(tmp,(8,16,32)) 
    
    # tmp = tf.py_func(create_target_response_map, [label[0], 16, 8, 8, 2,0.3, 0.33, 16], [tf.float32])
    # target16 = tf.reshape(tmp,(8,8,16)) 
    
    # initial = tf.Variable(tf.zeros_like(target2[0,:, :]),dtype=tf.float32, name="initial")
    # tmp_initial = initial
    # condition = tf.greater(target2[0,:, :], tf.constant(0,dtype=tf.float32),name="greater")
    # weight_factor_array = initial.assign( tf.where(condition, (tmp_initial + 2.0), tmp_initial, name="where_condition"), name="assign" )
    
    
    # init = tf.global_variables_initializer()
    # errors = [2.0,2.0]
    
    # loss_output = tf.placeholder(dtype=tf.float32,shape=(2),name='loss_output_placeholder')
    # loss_constant = tf.constant(1.0,shape=[2],dtype=tf.float32,name='loss_constant')
    # loss_output = tf.multiply(errors,loss_constant, name='loss_output_multiply')
    

    image_batch, labels_batch, image_paths,image_names, calib_matrices = loader.get_test_data(1)
    
    maps = create_target_response_map(labels_batch[0], 128, 64, 8, 2,0.3, 0.25, 2)
    target2 = tf.reshape(maps,(8,64,128)).numpy()
    maps = create_target_response_map(labels_batch[0], 64, 32, 8, 2,0.3, 0.25, 4)
    target4 = tf.reshape(maps,(8,32,64)).numpy()
    maps = create_target_response_map(labels_batch[0], 32, 16, 8, 2,0.3, 0.25, 8)
    target8 = tf.reshape(maps,(8,16,32)).numpy()
    maps = create_target_response_map(labels_batch[0], 16, 8, 8, 2,0.3, 0.25, 16)
    target16 = tf.reshape(maps,(8,8,16)).numpy()
    
    tmp = cv2.resize(image_batch[0],(128,64),interpolation = cv2.INTER_AREA)
    cv2.imshow("original",tmp)
    cv2.imshow("target2 1", target2[0,:,:])
    cv2.imshow("target2 2", target2[1,:,:])
    cv2.imshow("target2 3", target2[2,:,:])
    cv2.imshow("target2 4", target2[3,:,:])
    cv2.imshow("target2 5", target2[4,:,:])
    cv2.imshow("target2 6", target2[5,:,:])
    cv2.imshow("target2 7", target2[6,:,:])
    cv2.imshow("target2 8", target2[7,:,:])
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow("target4 1", target4[0,:,:])
    cv2.imshow("target4 2", target4[1,:,:])
    cv2.imshow("target4 3", target4[2,:,:])
    cv2.imshow("target4 4", target4[3,:,:])
    cv2.imshow("target4 5", target4[4,:,:])
    cv2.imshow("target4 6", target4[5,:,:])
    cv2.imshow("target4 7", target4[6,:,:])
    cv2.imshow("target4 8", target4[7,:,:])
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow("target8 1", target8[0,:,:])
    cv2.imshow("target8 2", target8[1,:,:])
    cv2.imshow("target8 3", target8[2,:,:])
    cv2.imshow("target8 4", target8[3,:,:])
    cv2.imshow("target8 5", target8[4,:,:])
    cv2.imshow("target8 6", target8[5,:,:])
    cv2.imshow("target8 7", target8[6,:,:])
    cv2.imshow("target8 8", target8[7,:,:])
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow("target16 1", target16[0,:,:])
    cv2.imshow("target16 2", target16[1,:,:])
    cv2.imshow("target16 3", target16[2,:,:])
    cv2.imshow("target16 4", target16[3,:,:])
    cv2.imshow("target16 5", target16[4,:,:])
    cv2.imshow("target16 6", target16[5,:,:])
    cv2.imshow("target16 7", target16[6,:,:])
    cv2.imshow("target16 8", target16[7,:,:])
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    test_target_map_creation("000001") 

