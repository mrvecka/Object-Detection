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
        self.BatchSize = cfg.BATCH_SIZE
        self.weight_factor = cfg.WEIGHT_FACTOR
        
        self.net_s_2 = None
        self.net_s_4 = None
        self.net_s_8 = None
        self.net_s_16 = None
        
        
    def create_detection_network(self, input):
        
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


        net = odn.create_detection_network_pool_layer(net, [2,2], 'layer15')

        net = odn.create_detection_network_layer('layer16', net, [3,3],512, 512, 1, 1, self.is_training)
        net = odn.normalization_layer(net, self.is_training)

        net = odn.create_detection_network_layer('layer17', net, [3,3], 512, 512, 1, 1, self.is_training)
        net = odn.normalization_layer(net, self.is_training)

        net = odn.create_detection_network_layer('layer18', net, [3,3], 512, 512, 3, 1, self.is_training)
        net = odn.normalization_layer(net, self.is_training)
        
        self.net_s_16 = odn.create_detection_network_output_layer('output16', net, [1, 1], 512, 8, 1, 1, self.is_training)

    

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
        
        N_p = tf.math.count_nonzero(image[0, :, :])

        
        second_error = 0
        error = 0.0
                   
        # get array of weight factor with the same shape as target 
        initial = tf.Variable(tf.ones_like(target[0,:, :]),dtype=tf.float32, name="initial")
        tmp_initial = initial
        condition = tf.greater(target[0,:, :], tf.constant(0,dtype=tf.float32),name="greater")
        weight_factor_array = initial.assign( tf.where(condition, (tmp_initial + self.weight_factor - 1), tmp_initial, name="where_condition"), name="assign" )
        
                    
        # mask = tf.greater(target[:,:, 0], 0)
        # non_zero_indices = tf.boolean_mask(image[:, :, 0], mask)
        # weight_factor_array = tf.dtypes.cast(non_zero_indices, tf.float32)
        # weight_factor_array = tf.add(weight_factor_array, tf.add(tf.dtypes.cast(1,tf.float32), tf.multiply(weight_factor_array, tf.dtypes.cast(self.weight_factor - 1,tf.float32))))
        
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

        result = cv2.merge(maps)
        
        return np.asarray(result,dtype=np.float32)

    def network_loss_function(self,input, labels, radius, circle_ration, boundaries, scale):
        
        # print(self.NET.shape)
        
        errors = []
        # self.NET.shape.dims[0].value this is batch size
   
        for i in range(self.BatchSize):          
            current_img = input[i]
            current_lbl = labels[i]
            img_error = self.scan_image_function(current_img, current_lbl, radius, circle_ration, boundaries, scale)
            errors.append(img_error)
            
        # LOSS FUNCTION
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_predicted))
        

        # optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        # train = optimizer.minimize(loss)
        # print(len(errors))
        loss_output = tf.placeholder(dtype=tf.float32,shape=(self.BatchSize),name='loss_output_placeholder')
        loss_constant = tf.constant(1.0,shape=[self.BatchSize],dtype=tf.float32,name='loss_constant')
        loss_output = tf.multiply(errors,loss_constant, name='loss_output_multiply')
        return tf.reduce_sum(loss_output)

    def train(self, loader):
            
        graph_path = os.path.dirname(os.path.abspath(__file__)) + r"\graphs\tensorboard"
        iterations = cfg.ITERATIONS
        
        image_placeholder = tf.placeholder(tf.float32, [None, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_CHANNELS],name="input_image_placeholder")
        labels_placeholder = tf.placeholder(tf.float32, [None, None, 10], name="input_label_placeholder")
        is_training = tf.placeholder(tf.bool, name='input_is_training_placeholder')
        
        self.create_detection_network(image_placeholder)
        
        # as y there should by last network layer
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.NET,logits=labels_placeholder))
        
        loss_s_2 = self.network_loss_function(self.net_s_2, labels_placeholder, 2, 0.3, 0.33, 2)
        loss_s_4 = self.network_loss_function(self.net_s_4, labels_placeholder, 2, 0.3, 0.33, 4)
        loss_s_8 = self.network_loss_function(self.net_s_8, labels_placeholder, 2, 0.3, 0.33, 8)
        loss_s_16 = self.network_loss_function(self.net_s_16, labels_placeholder, 2, 0.3, 0.33, 16)
        errors = tf.convert_to_tensor([loss_s_2, loss_s_4, loss_s_8, loss_s_16], dtype=tf.float32)
        optimizer = tf.train.AdamOptimizer(name="adam_optimalizer").minimize(errors)
        
        error_value = tf.reduce_mean(errors)
        
        saver = tf.train.Saver(name='model_saver')
        init = tf.global_variables_initializer()

        with tf.Session() as session:
        # initialise the variables

            session.run(init)
        
            # after model is fully trained
            # writer = tf.summary.FileWriter(graph_path, session.graph)
            test_acc = 1
            epoch = 1
            while test_acc > cfg.MAX_ERROR:
                for i in range(iterations):
                    image_batch, labels_batch = loader.get_train_data(self.BatchSize)                    
                    session.run(optimizer, 
                                    feed_dict={image_placeholder: image_batch, labels_placeholder: labels_batch, is_training: True})

                image_batch, labels_batch = loader.get_train_data(self.BatchSize)
                test_acc = session.run(error_value, 
                                feed_dict={image_placeholder: image_batch, labels_placeholder: labels_batch, is_training: False})
                
                print("Epoch:", (epoch), "test error: {:.5f}".format(test_acc))
                epoch += 1
                
            saver.save(session, cfg.MODEL_PATH)
                        
            
    def save_results(self, maps, scale):
        result = cv2.split(np.squeeze(maps,axis=0))
        path = r"C:\Users\Lukas\Documents\Object detection\result_s"+str(scale)+r"\response_map_0.jpg"
        cv2.imwrite(path, (result[0] - result[0].min()) * (255/(result[0].max() - result[0].min())))
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



    