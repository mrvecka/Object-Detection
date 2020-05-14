__date__   = '14/05/2020'
__author__ = 'Lukas Mrvecka'
__email__  = 'lukas.mrvecka.st@vsb.cz'
__source__ = 'http://libornovak.com/files/master_thesis.pdf'

# network architecture is inspired by Libor Novak work and upgraded

from __future__ import print_function
import tensorflow as tf
import config as cfg
import os
import cv2
import numpy as np
import Services.freeze_graph as freeze
import Services.helper as help
import math
from Services.timer import Timer


tf.logging.set_verbosity(tf.logging.ERROR)

class NetworkCreator():
    
    def __init__(self, batch):
        self.BatchSize = batch
        self.weight_factor = cfg.WEIGHT_FACTOR
        
        self.learning_rate = cfg.LEARNING_RATE
        self.update_edge = cfg.UPDATE_EDGE
        
        self.net_s_2 = None
        self.net_s_4 = None
        self.net_s_8 = None
        self.net_s_16 = None

    def create_detection_network_2(self,input):
        
        with tf.variable_scope("scale_2"):
            net = tf.layers.conv2d(input,64,[3,3],(1,1),"SAME",dilation_rate=(1,1),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer1")
            net = tf.layers.conv2d(net,64,[3,3],(2,2),"SAME",dilation_rate=(1,1),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer2")
            net = tf.layers.conv2d(net,128,[3,3],(1,1),"SAME",dilation_rate=(1,1),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer3")
            net = tf.layers.conv2d(net,128,[3,3],(1,1),"SAME",dilation_rate=(1,1),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer4")
            net = tf.layers.conv2d(net,128,[3,3],(1,1),"SAME",dilation_rate=(3,3),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer5")
            net = tf.layers.conv2d(net,128,[3,3],(1,1),"SAME",dilation_rate=(6,6),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer6")

            self.net_s_2 = tf.layers.conv2d(net,8,[1,1],(1,1),"SAME",dilation_rate=(1,1),activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=False,trainable=True,name="output2")
        
        with tf.variable_scope("scale_4"):
            net = tf.layers.conv2d(net,256,[3,3],(1,1),"SAME",dilation_rate=(1,1),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer7")
            net = tf.layers.conv2d(net,256,[3,3],(1,1),"SAME",dilation_rate=(1,1),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer8")
            net = tf.layers.conv2d(net,256,[3,3],(1,1),"SAME",dilation_rate=(1,1),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer9")
            net = tf.layers.conv2d(net,256,[3,3],(1,1),"SAME",dilation_rate=(3,3),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer10")
            
            self.net_s_4 = tf.layers.conv2d(net,8,[1,1],(1,1),"SAME",dilation_rate=(1,1),activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=False,trainable=True,name="output4")

        with tf.variable_scope("scale_8"):
            net = tf.layers.conv2d(net,512,[3,3],(1,1),"SAME",dilation_rate=(1,1),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer11")
            net = tf.layers.conv2d(net,512,[3,3],(1,1),"SAME",dilation_rate=(1,1),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer12")
            net = tf.layers.conv2d(net,512,[3,3],(1,1),"SAME",dilation_rate=(1,1),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer13")
            net = tf.layers.conv2d(net,512,[3,3],(1,1),"SAME",dilation_rate=(3,3),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer14")
                  
            self.net_s_8 = tf.layers.conv2d(net,8,[1,1],(1,1),"SAME",dilation_rate=(1,1),activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=False,trainable=True,name="output8")
       
        with tf.variable_scope("scale_16"):
            net = tf.layers.conv2d(net,512,[3,3],(1,1),"SAME",dilation_rate=(1,1),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer15")
            net = tf.layers.conv2d(net,512,[3,3],(1,1),"SAME",dilation_rate=(1,1),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer16",)
            net = tf.layers.conv2d(net,512,[3,3],(1,1),"SAME",dilation_rate=(1,1),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer17")
            net = tf.layers.conv2d(net,512,[3,3],(1,1),"SAME",dilation_rate=(3,3),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer18")

            self.net_s_16 = tf.layers.conv2d(net,8,[1,1],(1,1),"SAME",dilation_rate=(1,1),activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=False,trainable=True,name="output16")

    def scan_image_function(self, image, label, radius,circle_ratio, boundaries, scale):
                
        width = image.shape.dims[1].value
        height = image.shape.dims[0].value
        channels = image.shape.dims[2].value
        
        target = tf.py_func(self.create_target_response_map, [label, width, height, channels, radius,circle_ratio, boundaries, scale], [tf.float32])        
        target = tf.reshape(target,(channels,height,width))        
        
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
        weight_factor_array = initial.assign( tf.where(condition, (tmp_initial + self.weight_factor - 1), tmp_initial, name="where_condition"), name="assign" )

        error = tf.reduce_sum(tf.multiply(weight_factor_array, tf.square(tf.subtract(target[0,:, :], image[:, :, 0]))))
        for c in range(1, channels):
            second_error += tf.reduce_sum(
                tf.multiply(self.weight_factor,
                             tf.multiply(target[0, :, :],
                                          tf.square(tf.subtract(target[c,:, :], image[:, :, c])))))
        
                    
        error = (1/(2*N))*error
        
        sec_error = tf.cond(tf.equal(N_p,0),lambda: tf.constant(0,dtype=tf.float32, shape=()),lambda: tf.cast(1/ (2 * N_p * (channels -1)),dtype=tf.float32)*second_error )            
        
        error += sec_error
        return tf.cast(error, dtype=tf.float32)
    
    def GetObjectBounds(self, r, cr, bo, scale):
        ideal_size = (2 * r + 1) / cr * scale
        # bound above
        ext_above = ((1 - bo) * ideal_size) / 2 + bo * ideal_size
        bound_above = ideal_size + ext_above
        
        # bound below
        diff = ideal_size / 2
        ext_below = ((1 - bo)* diff) /2 + bo * diff
        bound_below = ideal_size - ext_below
        
        return bound_above, bound_below, ideal_size
    
    def create_target_response_map(self, labels, width, height, channels, r, circle_ratio, boundaries, scale):
                
        maps = cv2.split(np.zeros((height,width,8)))
        bound_above, bound_below, ideal = self.GetObjectBounds(r,circle_ratio,boundaries,scale)
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
                
                cv2.circle(maps[0], ( x, y ), int(r), 1, -1)
                cv2.GaussianBlur(maps[0], (3, 3), 100)

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

    def network_loss_function(self, labels):
                        
        with tf.variable_scope('loss_2'):
            errors = []
            for i in range(self.BatchSize):          
                current_img = self.net_s_2[i]
                current_lbl = labels[i]
                img_error = self.scan_image_function(current_img, current_lbl, 2, 0.3, 0.33, 2)
                errors.append(img_error)

            loss_output = tf.placeholder(dtype=tf.float32,shape=(self.BatchSize))
            loss_constant = tf.constant(1.0,shape=[self.BatchSize],dtype=tf.float32)
            loss_output = tf.multiply(errors,loss_constant)
            loss_2 = tf.reduce_sum(loss_output)
            
        with tf.variable_scope('loss_4'):  
            errors = []  
            for i in range(self.BatchSize):          
                current_img = self.net_s_4[i]
                current_lbl = labels[i]
                img_error = self.scan_image_function(current_img, current_lbl, 2, 0.3, 0.33, 4)
                errors.append(img_error)

            loss_output = tf.placeholder(dtype=tf.float32,shape=(self.BatchSize))
            loss_constant = tf.constant(1.0,shape=[self.BatchSize],dtype=tf.float32)
            loss_output = tf.multiply(errors,loss_constant)
            loss_4 = tf.reduce_sum(loss_output)
            
        with tf.variable_scope('loss_8'):
            errors = []
            for i in range(self.BatchSize):          
                current_img = self.net_s_8[i]
                current_lbl = labels[i]
                img_error = self.scan_image_function(current_img, current_lbl, 2, 0.3, 0.33, 8)
                errors.append(img_error)

            loss_output = tf.placeholder(dtype=tf.float32,shape=(self.BatchSize))
            loss_constant = tf.constant(1.0,shape=[self.BatchSize],dtype=tf.float32)
            loss_output = tf.multiply(errors,loss_constant)
            loss_8 = tf.reduce_sum(loss_output)
                    
        with tf.variable_scope('loss_16'):
            errors = []
            for i in range(self.BatchSize):          
                current_img = self.net_s_16[i]
                current_lbl = labels[i]
                img_error = self.scan_image_function(current_img, current_lbl, 2, 0.3, 0.33, 16)
                errors.append(img_error)

            loss_output = tf.placeholder(dtype=tf.float32,shape=(self.BatchSize))
            loss_constant = tf.constant(1.0,shape=[self.BatchSize],dtype=tf.float32)
            loss_output = tf.multiply(errors,loss_constant)
            loss_16 = tf.reduce_sum(loss_output)            
   
        return loss_2, loss_4, loss_8, loss_16

    def network_otimizer(self, loss):
        
        var_scale_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='scale_2')
        var_scale_4 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='scale_4')
        var_scale_8 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='scale_8')
        var_scale_16 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='scale_16')
        
        optimizer_2 = tf.train.AdamOptimizer(name="adam_optimalizer_2",learning_rate=self.learning)
        optimizer_4 = tf.train.AdamOptimizer(name="adam_optimalizer_4",learning_rate=self.learning)
        optimizer_8 = tf.train.AdamOptimizer(name="adam_optimalizer_8",learning_rate=self.learning)
        optimizer_16 = tf.train.AdamOptimizer(name="adam_optimalizer_16",learning_rate=self.learning)
        grads = tf.gradients(loss,var_scale_2 + var_scale_4 + var_scale_8 + var_scale_16)        

        grads_2 = grads[:len(var_scale_2)]
        grads_4 = grads[len(var_scale_2):len(var_scale_2) + len(var_scale_4)]
        grads_8 = grads[len(var_scale_2) + len(var_scale_4):len(var_scale_2) + len(var_scale_4) + len(var_scale_8)]
        grads_16 = grads[len(var_scale_2) + len(var_scale_4) + len(var_scale_8):]
        
        tran_opt_2 = optimizer_2.apply_gradients(zip(grads_2,var_scale_2))
        tran_opt_4 = optimizer_4.apply_gradients(zip(grads_4,var_scale_4))
        tran_opt_8 = optimizer_8.apply_gradients(zip(grads_8,var_scale_8))
        tran_opt_16 = optimizer_16.apply_gradients(zip(grads_16,var_scale_16))
        
        tran_opt = tf.group(tran_opt_2, tran_opt_4,tran_opt_8, tran_opt_16)
        return tran_opt

    def train(self, loader):
            
        iterations = cfg.ITERATIONS
        
        image_placeholder = tf.placeholder(tf.float32, [None, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_CHANNELS],name="input_image_placeholder")
        labels_placeholder = tf.placeholder(tf.float32, [None, None, 10], name="input_label_placeholder")
        self.learning = tf.placeholder(tf.float32,(),name="input_learning_rate")

        self.create_detection_network_2(image_placeholder)        
        loss_2, loss_4, loss_8, loss_16 = self.network_loss_function(labels_placeholder)
        loss = tf.reduce_mean([loss_2,loss_4,loss_8, loss_16],name="global_loss")    
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimize = self.network_otimizer(loss)
        
        saver = tf.train.Saver(name='model_saver')
        init = tf.global_variables_initializer()
        
        with tf.Session() as session:
            session.run(init)

            test_acc = 1
            epoch = 1
            tf.train.write_graph(session.graph_def,r".\model",'model.pbtxt')
            writer = tf.summary.FileWriter(r".\model", session.graph)
            print("Learning rate for AdamOtimizer", self.learning_rate)
            while test_acc > cfg.MAX_ERROR:
                t = Timer()
                t.start()
                for i in range(1):
                    image_batch, labels_batch, = loader.get_train_data(self.BatchSize)                    
                    session.run(optimize, 
                                    feed_dict={image_placeholder: image_batch, labels_placeholder: labels_batch, self.learning: self.learning_rate})

                image_batch, labels_batch = loader.get_train_data(self.BatchSize)
                test_acc = session.run(loss, 
                                feed_dict={image_placeholder: image_batch, labels_placeholder: labels_batch, self.learning: self.learning_rate})
                
                _ = t.stop()
                print("Epoch:", (epoch), "test error: {:.5f} time: ".format(test_acc)+t.get_formated_time())
                self.update_learning_rate(epoch, test_acc)
                epoch += 1
                
            saver.save(session, r".\model")            
            freeze.freeze_and_save()   
            
    def update_learning_rate(self, epoch, acc):
        if self.update_edge != -1 and acc < self.update_edge:
            self.learning_rate = self.learning_rate / 10
            self.update_edge = self.update_edge / 10
            return

        if epoch in cfg.UPDATE_LEARNING_RATE:
            self.learlearning_ratening = self.learning_rate / 10
            return           
 



    