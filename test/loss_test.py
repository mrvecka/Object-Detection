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

batchSize = 1

def scan_image_function( image, label, radius,circle_ratio, boundaries, scale):
        
    # ground_truth = self.create_target_response_map(label, label_size, 2)
    
    width = image.shape.dims[1].value
    height = image.shape.dims[0].value
    channels = image.shape.dims[2].value
    
    target = tf.py_func(create_target_response_map, [label, width, height, channels, radius,circle_ratio, boundaries, scale], [tf.float32])
    # assert target.shape != shaped_output.shape, "While computing loss of NN shape of ground truth must be same as shape of network result"
    
    target = tf.reshape(target,(channels,height,width),name="response_map_reshape")        
    
    # number of neurons in each output layer
    N = width * height
    
    # count of positive pixels
    
    N_p = tf.math.count_nonzero(image[:, :, 0])

    
    second_error = 0
    error = 0.0
                
    # get array of weight factor with the same shape as target 
    initial = tf.Variable(tf.ones_like(target[0,:, :]),dtype=tf.float32, name="initial")
    tmp_initial = initial
    condition = tf.greater(target[0,:, :], tf.constant(0,dtype=tf.float32),name="greater")
    weight_factor_array = initial.assign( tf.where(condition, (tmp_initial + 2.0), tmp_initial, name="where_condition"), name="assign" )
    
                
    # mask = tf.greater(target[:,:, 0], 0)
    # non_zero_indices = tf.boolean_mask(image[:, :, 0], mask)
    # weight_factor_array = tf.dtypes.cast(non_zero_indices, tf.float32)
    # weight_factor_array = tf.add(weight_factor_array, tf.add(tf.dtypes.cast(1,tf.float32), tf.multiply(weight_factor_array, tf.dtypes.cast(self.weight_factor - 1,tf.float32))))
    
    error = tf.reduce_sum(tf.multiply(weight_factor_array, tf.square(tf.subtract(target[0,:, :], image[:, :, 0]))))
    for c in range(1, channels):
        second_error += tf.reduce_sum(
            tf.multiply(2.0,
                            tf.multiply(target[0, :, :],
                                        tf.square(tf.subtract(target[c,:, :], image[:, :, c])))))
    
                
    error = (1/(2*N))*error
    
    tmp = 1/ (3 * N_p * (channels -1))
    
    error += tf.cast(tmp, tf.float32) * second_error
    
    
    return error
    
def GetObjectBounds( r, cr, bo, scale):
    ideal_size = (2 * r + 1) / cr * scale
    # bound above
    ext_above = ((1 - bo) * ideal_size) / 2 + bo * ideal_size
    bound_above = ideal_size + ext_above
    
    # bound below
    diff = ideal_size / 2
    ext_below = ((1 - bo)* diff) /2 + bo * diff
    bound_below = ideal_size - ext_below
    
    return bound_above, bound_below, ideal_size
    
def create_target_response_map(labels, width, height, channels, r, circle_ratio, boundaries, scale):
            
    # maps => array of shape (channels, orig_height, orig_width) 
    maps = cv2.split(np.zeros((height,width,8)))
    # self.index = 0
    # result = tf.scan(self.scan_label_function, labels, initializer=0) 
    bound_above, bound_below, ideal = GetObjectBounds(r,circle_ratio,boundaries,scale)
    for i in range(len(labels)):            
        label = labels[i]
        if label[0] == -1:
            continue
        # 0       1       2       3       4       5       6     7           8           9
        # fblx    fbly    fbrx    fbry    rblx    rbly    ftly  center_x    center_y    largest_dim
        
        #size = self.get_size_of_bounding_box(labels)
        if label[9] >= bound_below and label[9] <= bound_above:
            x = int(label[7] / scale)
            y = int(label[8] / scale)
            
            scaling_ratio = 1.0 / scale
            # print((self.orig_height,self.orig_width))
            #radius = ((circle_ration / scale) * szie ) - 1
            
            cv2.circle(maps[0], ( x, y ), int(r), 1, -1)
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

def network_loss_function(_input, labels, radius, circle_ration, boundaries, scale):
    
    # print(self.NET.shape)
    
    errors = []
    # self.NET.shape.dims[0].value this is batch size

    for i in range(batchSize):          
        current_img = _input[i]
        current_lbl = labels[i]
        img_error = scan_image_function(current_img, current_lbl, radius, circle_ration, boundaries, scale)
        errors.append(img_error)
        
    # LOSS FUNCTION
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_predicted))
    

    # optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    # train = optimizer.minimize(loss)
    # print(len(errors))
    loss_output = tf.placeholder(dtype=tf.float32,shape=(batchSize),name='loss_output_placeholder')
    loss_constant = tf.constant(1.0,shape=[batchSize],dtype=tf.float32,name='loss_constant')
    loss_output = tf.multiply(errors,loss_constant, name='loss_output_multiply')
    return tf.reduce_sum(loss_output)


def test_loss(loader):
    labels_placeholder = tf.placeholder(tf.float32, [None, None, 10], name="input_label_placeholder")


    _in_2 = tf.Variable(np.load(r"C:\Users\Lukas\Documents\Object detection\test\000006.pngs_2.txt.npy"),dtype=tf.float32) 
    _in_4 = tf.Variable(np.load(r"C:\Users\Lukas\Documents\Object detection\test\000006.pngs_4.txt.npy"),dtype=tf.float32) 
    _in_8 = tf.Variable(np.load(r"C:\Users\Lukas\Documents\Object detection\test\000006.pngs_8.txt.npy"),dtype=tf.float32) 
    _in_16 = tf.Variable(np.load(r"C:\Users\Lukas\Documents\Object detection\test\000006.pngs_16.txt.npy"),dtype=tf.float32) 

    with tf.variable_scope('loss_2'):
        loss_s_2 = network_loss_function(_in_2, labels_placeholder, 2, 0.3, 0.33, 2)
        
    with tf.variable_scope('loss_4'):
        loss_s_4 = network_loss_function(_in_4, labels_placeholder, 2, 0.3, 0.33, 4)
    
    with tf.variable_scope('loss_8'):
        loss_s_8 = network_loss_function(_in_8, labels_placeholder, 2, 0.3, 0.33, 8)
    
    with tf.variable_scope('loss_16'):
        loss_s_16 = network_loss_function(_in_16, labels_placeholder, 2, 0.3, 0.33, 16)
        
    errors = tf.convert_to_tensor([loss_s_2, loss_s_4, loss_s_8, loss_s_16], dtype=tf.float32)
    error_value = tf.reduce_mean(errors)
    
    
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        graph = session.graph

        trm_reshape = graph.get_tensor_by_name("response_map_reshape:0")
        assign = graph.get_tensor_by_name("assign:0")
        
        for i in range(len(loader.Data)):
            image_batch, labels_batch, paths, _ = loader.get_test_data_sequentialy(i)

            acc, trm_resh, ass = session.run([error_value,trm_reshape,assign], feed_dict={labels_placeholder: labels_batch})
            print("Test loss accuracy",acc)
            if math.isnan(acc):
                print(paths[0])


if __name__ == "__main__":
    loader = load.Loader()
    loader.load_data()
    test_loss(loader)