from __future__ import print_function
import tensorflow as tf
import Network.object_detection_network as odn
import config as cfg
import os
import cv2
import numpy as np

class NetworkCreator():
    
    def __init__(self,):
        self.device = cfg.DEVICE_NAME
        self.is_training = cfg.IS_TRAINING
        self.NET = None
        self.BatchSize = cfg.BATCH_SIZE
        
        self.loss_prob_center = []
        self.loss_fblx_values = []
        self.loss_fbly_values = []
        self.loss_fbrx_values = []
        self.loss_fbry_values = []
        self.loss_rblx_values = []
        self.loss_rbly_values = []
        self.loss_ftly_values = []
        
        
    def create_detection_network(self, input_batch):
        
        # data = odn.normalize_input_data(input_batch, self.is_training)
        
        #create actual network
        net = odn.create_detection_network_layer('layer1', input_batch, [3, 3], 3, 64, 1, 1, self.is_training)
        # net = odn.normalize_input_data(net, self.is_training)

        net = odn.create_detection_network_layer('layer2', net, [3, 3], 64, 64, 2, 2, self.is_training)
        # net = odn.normalize_input_data(net, self.is_training)
        
        net = odn.create_detection_network_layer('layer3', net, [3,3], 64, 128, 1, 1, self.is_training)
        # net = odn.normalize_input_data(net, self.is_training)
        net = odn.create_detection_network_layer('layer4', net, [3,3], 128, 128, 1, 1, self.is_training)
        net = odn.normalize_input_data(net, self.is_training)
        net = odn.create_detection_network_layer('layer5', net, [3,3], 128, 128, 3, 1, self.is_training)
        net = odn.normalize_input_data(net, self.is_training)
        net = odn.create_detection_network_layer('layer6', net, [3,3], 128, 128, 6, 1, self.is_training)
        
        # first output
        net = odn.normalize_input_data(net, self.is_training)
        net = odn.create_detection_network_output_layer('output1', net, [1, 1], 128, 8, 1, 1, self.is_training)
        # net = odn.normalize_input_data(net, self.is_training)
        
        # net = odn.create_detection_network_pool_layer(net, [2,2],'layer7')
        
        # net = odn.create_detection_network_layer('layer8', net, [3,3], 128, 256, 1, 1, self.is_training)
        
        # net = odn.create_detection_network_layer('layer9', net, [3,3], 256, 256, 1, 1, self.is_training)
        
        # net = odn.create_detection_network_layer('layer10', net, [3,3], 256, 256, 3, 1, self.is_training)
        
        # net = odn.create_detection_network_pool_layer(net, [2,2], 'layer11')
        
        # net = odn.create_detection_network_layer('layer12', net, [3,3], 256, 512, 1, 1, self.is_training)
        
        # net = odn.create_detection_network_layer('layer13', net, [3,3], 512, 512, 1, 1, self.is_training)
        
        # net = odn.create_detection_network_layer('layer14', net, [3,3], 512, 512, 3, 1, self.is_training)
        
        # net = odn.create_detection_network_pool_layer(net, [2,2], '15')

        # net = odn.create_detection_network_layer('layer16', net, [3,3], 512, 512, 1, 1, self.is_training)
        
        # net = odn.create_detection_network_layer('layer17', net, [3,3], 512, 512, 1, 1, self.is_training)
        
        # net = odn.create_detection_network_layer('layer18', net, [3,3], 512, 512, 3, 1, self.is_training)
        
        self.NET = net
    

    def scan_image_function(self, image, label):
        
        # ground_truth = self.create_target_response_map(label, label_size, 2)
        
        target = tf.py_func(self.create_target_response_map, [label, 2], [tf.float32])
        # assert target.shape != shaped_output.shape, "While computing loss of NN shape of ground truth must be same as shape of network result"
        target = tf.squeeze(tf.stack(target))
        target.set_shape((self.height,self.width,self.channels))
        
        # number of neurons in each output layer
        N = self.width * self.height
        
        # count of positive pixels
        
        N_p = tf.math.count_nonzero(image[:,:, 0])
        # for i in range(self.height):
        #     for j in range(self.width):
        #         if shaped_output[0][i][j] != 0:
        #             N_p += 1
        
        second_error = 0
        error = tf.reduce_sum(tf.square(tf.subtract(target[:,:, 0], image[:, :, 0])))
        for c in range(1, self.channels):
            second_error += tf.reduce_sum(
                tf.multiply(self.weight_factor,
                             tf.multiply(target[:, :, 0],
                                          tf.square(tf.subtract(target[:,:, c], image[:, :, c])))))
        
        
        # error = 0
        # second_error = 0
        # for i in range(self.height):
        #     for j in range(self.width):
        #         if target[0][i][j] != 0:
        #             error += (1 + (self.weight_factor - 1)) * pow(target[0][i][j] - image[0][i][j],2)
        #         else:
        #             error += pow(target[0][i][j] - image[0][i][j],2)
                
        #         for c in range(1, self.channels):
        #             second_error += self.weight_factor * target[0][i][j] * pow(target[c][i][j] - image[c][i][j],2)
                    
        # tf.print(error)  
        # tf.print(second_error)
        # tf.print(N)
        # tf.print(N_p)  
        # print(error)  
        # print(second_error)
        # print(N)
        # print(N_p)                 
        error = (1/(2*N))*error
        
        tmp = 1/ (2 * N_p * (self.channels -1))
        
        error += tf.cast(tmp, tf.float32) * second_error
        
        
        return error
    
    def create_target_response_map(self, labels, r):
                
        # maps => array of shape (channels, orig_height, orig_width) 
        maps = cv2.split(np.zeros((self.orig_height,self.orig_width,self.channels)))
        # self.index = 0
        # result = tf.scan(self.scan_label_function, labels, initializer=0) 
        for i in range(len(labels)):            
            label = labels[i]
            if label[0] == -1:
                break
            # 0       1       2       3       4       5       6
            # fblx    fbly    fbrx    fbry    rblx    rbly    ftly
            
            #size = self.get_size_of_bounding_box(labels)
            x = (label[2] + label[4]) / 2            
            y = (label[3] + label[5]) / 2
            
            center_height = (label[1] - label[3])/2
            
            center_x = x
            center_y = y - center_height
            
            # check if out of bounds
            if center_x > self.orig_width:
                center_x = self.orig_width-1
            if center_y > self.orig_height:
                center_y = self.orig_height-1
            # print((self.orig_height,self.orig_width))
            #radius = ((circle_ration / scale) * szie ) - 1
            cv2.circle(maps[0], (int(center_y), int(center_x)), int(r), (255, 255, 255), -1)
            # print((int(center_x), int(center_y)))
            maps[1][int(center_y)][int(center_x)] = center_x - label[0]
            maps[2][int(center_y)][int(center_x)] = center_y - label[1]
            maps[3][int(center_y)][int(center_x)] = center_x - label[2]
            maps[4][int(center_y)][int(center_x)] = center_y - label[3]
            maps[5][int(center_y)][int(center_x)] = center_x - label[4]
            maps[6][int(center_y)][int(center_x)] = center_y - label[5]
            maps[7][int(center_y)][int(center_x)] = center_y - label[6]
               
        cv2.GaussianBlur(maps[0], (3, 3), 1)
        # opencv can resize max 4 channels at once so we will resize them separately
        # opencv use as parameter shape for resize function (width,height) not vice versa
        output = [cv2.resize(item,(self.width,self.height),interpolation=cv2.INTER_AREA) for item in maps]
        
        # output is still of shape (channels, orig_height, orig_width) 
        # we have to merge arrays to one with shape (orig_height, orig_width,channels) because it has to be same as network output 
        result = cv2.merge(output)
        return np.asarray(result,dtype=np.float32)
            
    def get_size_of_bounding_box(self, labels):
        
        values = []
        for i in range(len(labels)):
            label = labels[i]
            height = label.ftl_y - label.fbl_y
            values.append(height)
            width = sqrt(pow(label.rbl_x - label.fbr_x,2) + pow(label.rbl_y - label.fbr_y,2))
            values.append(width)
        
        return max(values)

    def network_loss_function(self, labels):
        
        print(self.NET.shape)
        
        errors = []
        self.scale = 4
        self.fov = 9
        # self.NET.shape.dims[0].value this is batch size
        self.width = self.NET.shape.dims[2].value
        self.height = self.NET.shape.dims[1].value
        self.channels = self.NET.shape.dims[3].value
        self.weight_factor = 1.5
        
        self.orig_width = cfg.IMG_ORIG_WIDTH
        self.orig_height = cfg.IMG_ORIG_HEIGHT
        

        
        for i in range(self.BatchSize):          
            current_img = self.NET[i]
            current_lbl = labels[i]
            img_error = self.scan_image_function(current_img, current_lbl)
            errors.append(img_error)
            
        # LOSS FUNCTION
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_predicted))
        

        # optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        # train = optimizer.minimize(loss)
        print(len(errors))
        loss_output = tf.placeholder(dtype=tf.float32,shape=(self.BatchSize),name='loss_output_placeholder')
        loss_constant = tf.constant(1.0,shape=[self.BatchSize],dtype=tf.float32,name='loss_constant')
        loss_output = tf.multiply(errors,loss_constant, name='loss_output_multiply')
        return loss_output

    def start_training(self, loader):
            
        graph_path = os.path.dirname(os.path.abspath(__file__)) + r"\graphs\tensorboard"
        epochs = cfg.EPOCHS
        iterations = cfg.ITERATIONS
        learn_rate = cfg.LEARNING_RATE
        
        image_placeholder = tf.placeholder(tf.float32, [None, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_CHANNELS],name="input_image_placeholder")
        labels_placeholder = tf.placeholder(tf.float32, [None, None, 7], name="input_label_placeholder")
        is_training = tf.placeholder(tf.bool, name='input_is_training_placeholder')
        
        self.create_detection_network(image_placeholder)
        
        # as y there should by last network layer
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.NET,logits=labels_placeholder))
        
        loss = self.network_loss_function(labels_placeholder)
        optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate,name="adam_optimalizer").minimize(loss)
        
        error_value = tf.reduce_mean(loss)
        
        saver = tf.train.Saver(name='model_saver')
        init = tf.global_variables_initializer()

        with tf.Session() as session:
        # initialise the variables

            session.run(init)
        
            # after model is fully trained
            # writer = tf.summary.FileWriter(graph_path, session.graph)

            for epoch in range(epochs):
                avg_cost = 0
                for i in range(iterations):
                    image_batch, labels_batch, object_count, _ = loader.get_train_data(self.BatchSize)
                    session.run(optimizer, 
                                    feed_dict={image_placeholder: image_batch, labels_placeholder: labels_batch, is_training: True})

                image_batch, labels_batch, object_count, _ = loader.get_train_data(self.BatchSize)
                test_acc = session.run(error_value, 
                                feed_dict={image_placeholder: image_batch, labels_placeholder: labels_batch, is_training: False})
                
                print("Epoch:", (epoch + 1), "test error: {:.5f}".format(test_acc*100))

            saver.save(session, cfg.MODEL_PATH)
            
            # call test 
            image_batch, labels_batch, object_count, image_paths = loader.get_train_data(1)
            
            response_maps = session.run(self.NET, feed_dict={image_placeholder: image_batch, labels_placeholder: labels_batch, is_training: False})
            result = cv2.split(np.squeeze(response_maps,axis=0))
            path1 = r"C:\Users\Lukas\Documents\Object detection\result\response_map0.jpg"
            cv2.imwrite(path1, 255 * result[0])
            path1 = r"C:\Users\Lukas\Documents\Object detection\result\response_map1.jpg"
            cv2.imwrite(path1, 255* result[1])
            path1 = r"C:\Users\Lukas\Documents\Object detection\result\response_map2.jpg"
            cv2.imwrite(path1, 255*result[2])
            path1 = r"C:\Users\Lukas\Documents\Object detection\result\response_map3.jpg"
            cv2.imwrite(path1, 255*result[3])
            path1 = r"C:\Users\Lukas\Documents\Object detection\result\response_map4.jpg"
            cv2.imwrite(path1, 255*result[4])
            path1 = r"C:\Users\Lukas\Documents\Object detection\result\response_map5.jpg"
            cv2.imwrite(path1, 255*result[5])
            path1 = r"C:\Users\Lukas\Documents\Object detection\result\response_map6.jpg"
            cv2.imwrite(path1, 255*result[6])
            path1 = r"C:\Users\Lukas\Documents\Object detection\result\response_map7.jpg"
            cv2.imwrite(path1, 255*result[7])
            
            print(image_paths[0])



    