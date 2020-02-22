from __future__ import print_function
import tensorflow as tf
import Network.object_detection_network as odn
import config as cfg
import os
import cv2
import numpy as np
import Services.freeze_graph as freeze
import Services.helper as help
from Services.lr_queue import LRQueue
import math

tf.logging.set_verbosity(tf.logging.ERROR)

class NetworkCreator():
    
    def __init__(self,):
        self.device = cfg.DEVICE_NAME
        self.is_training = tf.placeholder_with_default(cfg.IS_TRAINING, (), name='input_is_training_placeholder')  
        # self.is_training = True
        self.BatchSize = cfg.BATCH_SIZE
        self.weight_factor = cfg.WEIGHT_FACTOR
        
        self.net_s_2 = None
        self.net_s_4 = None
        self.net_s_8 = None
        self.net_s_16 = None
        
        
    def create_model(self, input):
        
        # data = odn.normalize_input_data(input_batch, self.is_training)
        
        #create actual network
        net = odn.create_detection_network_layer('layer1', input, [3, 3], 3, 64, 1, 1, self.is_training)
        net = odn.normalization_layer(net, self.is_training)

        net = odn.create_detection_network_layer('layer2', net, [3, 3], 64, 64, 2, 2, self.is_training)
        net = odn.normalization_layer(net, self.is_training)
        
        net = odn.create_detection_network_layer('layer3', net, [3,3], 64, 128, 1, 1, self.is_training)
        net = odn.normalization_layer(net, self.is_training)
        net = odn.create_detection_network_layer('layer4', net, [3,3], 128, 128, 1, 1, self.is_training)
        net = odn.normalization_layer(net, self.is_training)
        net = odn.create_detection_network_layer('layer5', net, [3,3], 128, 128, 3, 1, self.is_training)
        net = odn.normalization_layer(net, self.is_training)
        net = odn.create_detection_network_layer('layer6', net, [3,3], 128, 128, 6, 1, self.is_training)
        
        # first output
        net = odn.normalization_layer(net, self.is_training)
        self.net_s_2 = odn.create_detection_network_output_layer('output2', net, [1, 1], 128, 8, 1, 1, self.is_training)
                
        net = odn.create_detection_network_pool_layer(net, [2,2],'layer7')
        
        net = odn.create_detection_network_layer('layer8', net, [3,3], 128, 256, 1, 1, self.is_training)
        net = odn.normalization_layer(net, self.is_training)
        net = odn.create_detection_network_layer('layer9', net, [3,3], 256, 256, 1, 1, self.is_training)
        net = odn.normalization_layer(net, self.is_training)
        net = odn.create_detection_network_layer('layer10', net, [3,3], 256, 256, 3, 1, self.is_training)
        net = odn.normalization_layer(net, self.is_training)

        self.net_s_4 = odn.create_detection_network_output_layer('output4', net, [1, 1], 256, 8, 1, 1, self.is_training)

        
        net = odn.create_detection_network_pool_layer(net, [2,2], 'layer11')
        
        net = odn.create_detection_network_layer('layer12', net, [3,3], 256, 512, 1, 1, self.is_training)
        net = odn.normalization_layer(net, self.is_training)

        net = odn.create_detection_network_layer('layer13', net, [3,3], 512, 512, 1, 1, self.is_training)
        net = odn.normalization_layer(net, self.is_training)

        net = odn.create_detection_network_layer('layer14', net, [3,3], 512, 512, 3, 1, self.is_training)
        net = odn.normalization_layer(net, self.is_training)

        self.net_s_8 = odn.create_detection_network_output_layer('output8', net, [1, 1], 512, 8, 1, 1, self.is_training)


        net = odn.create_detection_network_pool_layer(net , [2,2], 'layer15')

        net = odn.create_detection_network_layer('layer16', net, [3,3],512, 512, 1, 1, self.is_training)
        net = odn.normalization_layer(net, self.is_training)

        net = odn.create_detection_network_layer('layer17', net, [3,3], 512, 512, 1, 1, self.is_training)
        net = odn.normalization_layer(net, self.is_training)

        net = odn.create_detection_network_layer('layer18', net, [3,3], 512, 512, 3, 1, self.is_training)
        net = odn.normalization_layer(net, self.is_training)
        
        self.net_s_16 = odn.create_detection_network_output_layer('output16', net, [1, 1], 512, 8, 1, 1, self.is_training)

    def create_detection_network_2(self,input):
        
        
        net = odn.normalization_layer(input, self.is_training)
        with tf.variable_scope("scale_2"):
            net = tf.layers.conv2d(net,64,[3,3],(1,1),"SAME",dilation_rate=(1,1),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer1")
            net = tf.layers.conv2d(net,64,[3,3],(2,2),"SAME",dilation_rate=(1,1),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer2")
            #net = odn.normalization_layer(net, self.is_training)
            net = tf.layers.conv2d(net,128,[3,3],(1,1),"SAME",dilation_rate=(1,1),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer3")
            #net = odn.normalization_layer(net, self.is_training)
            net = tf.layers.conv2d(net,128,[3,3],(1,1),"SAME",dilation_rate=(1,1),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer4")
            #net = odn.normalization_layer(net, self.is_training)
            net = tf.layers.conv2d(net,128,[3,3],(1,1),"SAME",dilation_rate=(3,3),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer5")
            #net = odn.normalization_layer(net, self.is_training)
            net = tf.layers.conv2d(net,128,[3,3],(1,1),"SAME",dilation_rate=(6,6),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer6")
            #net = odn.normalization_layer(net, self.is_training)
            # first output
            self.net_s_2 = tf.layers.conv2d(net,8,[1,1],(1,1),"SAME",dilation_rate=(1,1),activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=False,trainable=True,name="output2")
        #net = odn.normalization_layer(net, self.is_training)
        
        net = tf.layers.max_pooling2d(net,(2,2),(2,2),"SAME",name="layer7")
        # net = tf.layers.dropout(net, 0.5) 
        
        with tf.variable_scope("scale_4"):
            net = tf.layers.conv2d(net,256,[3,3],(1,1),"SAME",dilation_rate=(1,1),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer8")
            #net = odn.normalization_layer(net, self.is_training)
            net = tf.layers.conv2d(net,256,[3,3],(1,1),"SAME",dilation_rate=(1,1),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer9")
            #net = odn.normalization_layer(net, self.is_training)
            net = tf.layers.conv2d(net,256,[3,3],(1,1),"SAME",dilation_rate=(3,3),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer10")
            #net = odn.normalization_layer(net, self.is_training)

            self.net_s_4 = tf.layers.conv2d(net,8,[1,1],(1,1),"SAME",dilation_rate=(1,1),activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=False,trainable=True,name="output4")
        #net = odn.normalization_layer(net, self.is_training)

        net = tf.layers.max_pooling2d(net,(2,2),(2,2),"SAME",name="layer11")
        # net = tf.layers.dropout(net, 0.5) 

        with tf.variable_scope("scale_8"):
            net = tf.layers.conv2d(net,512,[3,3],(1,1),"SAME",dilation_rate=(1,1),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer12")
            #net = odn.normalization_layer(net, self.is_training)
            net = tf.layers.conv2d(net,512,[3,3],(1,1),"SAME",dilation_rate=(1,1),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer13")
            #net = odn.normalization_layer(net, self.is_training)
            net = tf.layers.conv2d(net,512,[3,3],(1,1),"SAME",dilation_rate=(3,3),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer14")
            #net = odn.normalization_layer(net, self.is_training)

                    
            self.net_s_8 = tf.layers.conv2d(net,8,[1,1],(1,1),"SAME",dilation_rate=(1,1),activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=False,trainable=True,name="output8")
        #net = odn.normalization_layer(net, self.is_training)

        net = tf.layers.max_pooling2d(net,(2,2),(2,2),"SAME",name="layer15")
        # net = tf.layers.dropout(net, 0.5) 
        
        
        with tf.variable_scope("scale_16"):
            net = tf.layers.conv2d(net,512,[3,3],(1,1),"SAME",dilation_rate=(1,1),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer16",)
            #net = odn.normalization_layer(net, self.is_training)
            net = tf.layers.conv2d(net,512,[3,3],(1,1),"SAME",dilation_rate=(1,1),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer17")
            #net = odn.normalization_layer(net, self.is_training)
            net = tf.layers.conv2d(net,512,[3,3],(1,1),"SAME",dilation_rate=(3,3),activation="relu",kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=True,trainable=True,name="layer18")
            #net = odn.normalization_layer(net, self.is_training)

            self.net_s_16 = tf.layers.conv2d(net,8,[1,1],(1,1),"SAME",dilation_rate=(1,1),activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=False,trainable=True,name="output16")

    def scan_image_function(self, image, label, radius,circle_ratio, boundaries, scale):
        
        # ground_truth = self.create_target_response_map(label, label_size, 2)
        
        width = image.shape.dims[1].value
        height = image.shape.dims[0].value
        channels = image.shape.dims[2].value
        
        target = tf.py_func(self.create_target_response_map, [label, width, height, channels, radius,circle_ratio, boundaries, scale], [tf.float32])
        # assert target.shape != shaped_output.shape, "While computing loss of NN shape of ground truth must be same as shape of network result"
        
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
        
        tmp = 1/ (3 * N_p * (channels -1))
        
        error += tf.cast(tmp, tf.float32) * second_error
        
        
        return error
    
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
                
        # maps => array of shape (channels, orig_height, orig_width) 
        maps = cv2.split(np.zeros((height,width,8)))
        # self.index = 0
        # result = tf.scan(self.scan_label_function, labels, initializer=0) 
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
            
        graph_path = os.path.dirname(os.path.abspath(__file__)) + r"\graphs\tensorboard"
        iterations = cfg.ITERATIONS
        
        image_placeholder = tf.placeholder(tf.float32, [None, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_CHANNELS],name="input_image_placeholder")
        labels_placeholder = tf.placeholder(tf.float32, [None, None, 10], name="input_label_placeholder")
        self.learning = tf.placeholder(tf.float32,(),name="input_learning_rate")
        #self.is_training = tf.placeholder(tf.bool, name='input_is_training_placeholder')
        #self.is_training = tf.placeholder_with_default(cfg.IS_TRAINING, (), name='input_is_training_placeholder')  

        self.create_detection_network_2(image_placeholder)        
        loss_2, loss_4, loss_8, loss_16 = self.network_loss_function(labels_placeholder)
        loss = tf.reduce_sum([loss_2,loss_4,loss_8, loss_16],name="global_loss")    
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimize = self.network_otimizer(loss)
        
        saver = tf.train.Saver(name='model_saver')
        init = tf.global_variables_initializer()
        
        with tf.Session() as session:
            session.run(init)

            test_acc = 1
            epoch = 1
            tf.train.write_graph(session.graph_def,r"C:\Users\Lukas\Documents\Object detection\model",'model.pbtxt')
            writer = tf.summary.FileWriter(r"C:\Users\Lukas\Documents\Object detection\model", session.graph)
            learning = cfg.LEARNING_RATE
            update_edge = 0.1
            print("Learning rate for AdamOtimizer", learning)
            while test_acc > cfg.MAX_ERROR:
                for i in range(iterations):
                    image_batch, labels_batch, = loader.get_train_data(self.BatchSize)                    
                    session.run(optimize, 
                                    feed_dict={image_placeholder: image_batch, labels_placeholder: labels_batch, self.is_training: True, self.learning: learning})

                image_batch, labels_batch = loader.get_train_data(self.BatchSize)
                test_acc = session.run(loss, 
                                feed_dict={image_placeholder: image_batch, labels_placeholder: labels_batch, self.is_training: False, self.learning: learning})
                
                print("Epoch:", (epoch), "test error: {:.5f}".format(test_acc))
                epoch += 1

                if math.isnan(test_acc):
                    s_2, s_4, s_8, s_16 = session.run([self.net_s_2,self.net_s_4,self.net_s_8,self.net_s_16], 
                                feed_dict={image_placeholder: image_batch[0], labels_placeholder: labels_batch[0], self.is_training: False, self.learning: learning})
                    np.save(r"C:\Users\Lukas\Documents\Object detection\test\train_s_2.txt",s_2)
                    np.save(r"C:\Users\Lukas\Documents\Object detection\test\train_s_4.txt",s_4)
                    np.save(r"C:\Users\Lukas\Documents\Object detection\test\train_s_8.txt",s_8)
                    np.save(r"C:\Users\Lukas\Documents\Object detection\test\train_s_16.txt",s_16)
                    break
                
                if test_acc < update_edge:
                    learning = learning / 10
                    update_edge = update_edge / 10
                    print("Learning rate updated to", learning)

                # if epoch % 20 == 0:
                #     learning = learning / 10
                #     print("Learning rate updated to", learning)
                
            saver.save(session, cfg.MODEL_PATH)            
            freeze.freeze_and_save()              
                        
    def test_loss(self, loader):
            
        graph_path = os.path.dirname(os.path.abspath(__file__)) + r"\graphs\tensorboard"
        iterations = cfg.ITERATIONS
        
        image_placeholder = tf.placeholder(tf.float32, [None, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_CHANNELS],name="input_image_placeholder")
        labels_placeholder = tf.placeholder(tf.float32, [None, None, 10], name="input_label_placeholder")
        is_training = tf.placeholder(tf.bool, name='input_is_training_placeholder')
        
        self.create_detection_network_2(image_placeholder)
        
        with tf.variable_scope('loss_2'):
            loss_s_2 = self.network_loss_function(self.net_s_2, labels_placeholder, 2, 0.3, 0.33, 2)
            
        with tf.variable_scope('loss_4'):    
            loss_s_4 = self.network_loss_function(self.net_s_4, labels_placeholder, 2, 0.3, 0.33, 4)
        
        with tf.variable_scope('loss_8'):
            loss_s_8 = self.network_loss_function(self.net_s_8, labels_placeholder, 2, 0.3, 0.33, 8)
        
        with tf.variable_scope('loss_16'):
            loss_s_16 = self.network_loss_function(self.net_s_16, labels_placeholder, 2, 0.3, 0.33, 16)
            
        errors = tf.convert_to_tensor([loss_s_2, loss_s_4, loss_s_8, loss_s_16], dtype=tf.float32)
        optimizer = tf.train.AdamOptimizer(name="adam_optimalizer",learning_rate=0.0001).minimize(errors)
        
        error_value = tf.reduce_mean(errors)
        
        saver = tf.train.Saver(name='model_saver')
        init = tf.global_variables_initializer()





        with tf.Session() as session:
        # initialise the variables

            session.run(init)

            test_acc = 1
            epoch = 1
            
            
            while test_acc > cfg.MAX_ERROR:
                for i in range(iterations):
                    image_batch, labels_batch, = loader.get_train_data(self.BatchSize)                    
                    session.run(optimizer, 
                                    feed_dict={image_placeholder: image_batch, labels_placeholder: labels_batch, is_training: True})

                image_batch, labels_batch, paths, _ = loader.get_test_data(self.BatchSize)
                test_acc, s_2 = session.run([error_value,self.net_s_2], 
                                feed_dict={image_placeholder: image_batch, labels_placeholder: labels_batch, is_training: False})
                
                print("Epoch:", (epoch), "test error: {:.5f}".format(test_acc))
                if math.isnan(test_acc):
                    for p in paths:
                        print(p)
                    
                    for i in range(len(image_batch)):
                        test_acc, s_2, s_4, s_8, s_16 = session.run([error_value,self.net_s_2,self.net_s_4,self.net_s_8,self.net_s_16], 
                                feed_dict={image_placeholder: image_batch[i], labels_placeholder: labels_batch[i], is_training: False})
                        
                        if math.isnan(test_acc):
                            np.save(r"C:\Users\Lukas\Documents\Object detection\test\\"+paths[i]+"s_2.txt",s_2)
                            np.save(r"C:\Users\Lukas\Documents\Object detection\test\\"+paths[i]+"s_4.txt",s_4)
                            np.save(r"C:\Users\Lukas\Documents\Object detection\test\\"+paths[i]+"s_8.txt",s_8)
                            np.save(r"C:\Users\Lukas\Documents\Object detection\test\\"+paths[i]+"s_16.txt",s_16)
                        
                    break              
            
                epoch +=1  
            
    def save_results(self, maps, scale):
        result = cv2.split(np.squeeze(maps,axis=0))
        path = r"C:\Users\Lukas\Documents\Object detection\result_s"+str(scale)+r"\response_map_0.jpg"
        cv2.imwrite(path, (maps[0,:,:,0] - maps[0,:,:,0].min()) * (255/(maps[0,:,:,0].max() - maps[0,:,:,0].min())))
        path = r"C:\Users\Lukas\Documents\Object detection\result_s"+str(scale)+r"\response_map_1.jpg"
        cv2.imwrite(path, 255* result[1])
        path = r"C:\Users\Lukas\Documents\Object detection\result_s"+str(scale)+r"\response_map_2.jpg"
        cv2.imwrite(path, 255*result[2])
        path = r"C:\Users\Lukas\Documents\Object detection\result_s"+str(scale)+r"\response_map_3.jpg"
        cv2.imwrite(path, 255*result[3])
        path = r"C:\Users\Lukas\Documents\Object detection\result_s"+str(scale)+r"\response_map_4.jpg"
        cv2.imwrite(path, 255*result[4])
        path = r"C:\Users\Lukas\Documents\Object detection\result_s"+str(scale)+r"\response_map_5.jpg"
        cv2.imwrite(path, 255*result[5])
        path = r"C:\Users\Lukas\Documents\Object detection\result_s"+str(scale)+r"\response_map_6.jpg"
        cv2.imwrite(path, 255*result[6])
        path = r"C:\Users\Lukas\Documents\Object detection\result_s"+str(scale)+r"\response_map_7.jpg"
        cv2.imwrite(path, 255*result[7])



    