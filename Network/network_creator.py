import tensorflow as tf
import config as cfg
import os
import cv2
import numpy as np
import Services.helper as help
import math

class NetworkLoss():
    def __init__(self,batch,radius,circle_ratio, boundaries, scale):
        self.radius = radius
        self.circle_ratio = circle_ratio
        self.boundaries = boundaries
        self.scale = scale
        self.batch_size = batch
        self.weight_factor = cfg.WEIGHT_FACTOR
    
    #@tf.function
    def compute_loss(self,images,labels):
        errors = []
        for i in range(self.batch_size):          
            current_img = images[i]
            current_lbl = labels[i]
            img_error = self.scan_image_function(current_img, current_lbl)
            errors.append(img_error)

        errors_as_tensor = tf.convert_to_tensor(errors,dtype=tf.float32)
        loss = tf.reduce_sum(errors_as_tensor)
        return [loss]
    
    def scan_image_function(self, image, label):
        
        # ground_truth = self.create_target_response_map(label, label_size, 2)
        
        width = image.shape.dims[1].value
        height = image.shape.dims[0].value
        channels = image.shape.dims[2].value
        
        target = tf.py_function(self.create_target_response_map, [label, width, height], [tf.float32])
        # assert target.shape != shaped_output.shape, "While computing loss of NN shape of ground truth must be same as shape of network result"
        
        target = tf.reshape(target,(channels,height,width))        
        
        # number of neurons in each output layer
        N = width * height
        
        # count of positive pixels
        
        N_p = tf.math.count_nonzero(image[:, :, 0])      
        second_error = 0
        error = 0.0
                   
        # get array of weight factor with the same shape as target 
        initial = tf.constant(1,dtype=tf.float32, shape=(height,width))
        tmp_initial = initial
        condition = tf.greater(target[0,:, :], tf.constant(0,dtype=tf.float32),name="greater")
        weight_factor_array = tf.add(initial, tf.where(condition, (tmp_initial + self.weight_factor - 1), tmp_initial, name="where_condition"), name="assign" )

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
           
    def GetObjectBounds(self):
        ideal_size = (2.0 * self.radius + 1.0) / self.circle_ratio * self.scale
        # bound above
        ext_above = ((1.0 - self.boundaries) * ideal_size) / 2.0 + self.boundaries * ideal_size
        bound_above = ideal_size + ext_above
        
        # bound below
        diff = ideal_size / 2.0
        ext_below = ((1 - self.boundaries)* diff) / 2.0 + self.boundaries * diff
        bound_below = ideal_size - ext_below
        
        return bound_above, bound_below, ideal_size
    
    def create_target_response_map(self, labels, width, height):
                
        # maps => array of shape (channels, orig_height, orig_width) 
        maps = cv2.split(np.zeros((height,width,8)))
        # self.index = 0
        # result = tf.scan(self.scan_label_function, labels, initializer=0) 
        bound_above, bound_below, ideal = self.GetObjectBounds()
        for i in range(len(labels)):            
            label = labels[i]
            if label[0] == -1:
                continue
            # 0       1       2       3       4       5       6     7           8           9
            # fblx    fbly    fbrx    fbry    rblx    rbly    ftly  center_x    center_y    largest_dim
            
            #size = self.get_size_of_bounding_box(labels)
            if label[9] >= bound_below and label[9] <= bound_above:
                x = int(label[7] / self.scale)
                y = int(label[8] / self.scale)
                
                scaling_ratio = 1.0 / self.scale
                # print((self.orig_height,self.orig_width))
                #radius = ((circle_ration / scale) * szie ) - 1
                
                cv2.circle(maps[0], ( x, y ), int(self.radius), 1, -1)
                cv2.GaussianBlur(maps[0], (3, 3), 100)

                # x_acc = x * scaling_ratio
                # y_acc = y * scaling_ratio
                for c in range(1,8):
                    
                    for l in range(-self.radius,self.radius,1):
                        for j in range(-self.radius,self.radius,1):
                            xp = x + j
                            yp = y + l
                            
                            if xp >= 0 and xp < width and yp >= 0 and yp < height:
                                if maps[0][yp][xp] > 0.0:
                                    if c ==1 or c == 3 or c == 5:
                                        maps[c][yp][xp] = 0.5 + (label[c-1] - x - j * self.scale) / ideal
                                    elif c == 2 or c == 4 or c == 6 or c == 7:
                                        maps[c][yp][xp] = 0.5 + (label[c-1] - y - l * self.scale) / ideal
        
        return np.asarray(maps,dtype=np.float32)

class NetworkCreator():
    
    def __init__(self):
        self.device = cfg.DEVICE_NAME
        # self.is_training = tf.placeholder_with_default(cfg.IS_TRAINING, (), name='input_is_training_placeholder')  
        # self.is_training = True
        self.BatchSize = cfg.BATCH_SIZE
        self.weight_factor = cfg.WEIGHT_FACTOR
        self.initializer = tf.initializers.glorot_uniform()
        self.net_s_2 = None
        self.net_s_4 = None
        self.net_s_8 = None
        self.net_s_16 = None

    def conv2d(self, inputs , filters , stride_size, dilation, name, activation=True ):
        out = tf.nn.conv2d( inputs , filters , strides=[ 1 , stride_size , stride_size , 1 ] ,dilations=[1, dilation, dilation, 1], padding="SAME", name=name+'_convolution' ) 
        if activation:
            return tf.nn.relu( out , name=name+'_relu_activation') 
        else:
            return out
        
    def maxpool(self, inputs , pool_size , stride_size, name ):
        return tf.nn.max_pool2d( inputs , ksize=[ 1 , pool_size , pool_size , 1 ] , padding='SAME' , strides=[ 1 , stride_size , stride_size , 1 ], name=name+'pool' )

    def get_weight(self, shape , name ):
        return tf.Variable( self.initializer( shape ) , name=name , trainable=True , dtype=tf.float32 )

    def init_weights(self):
        shapes_2 = [
            [3,3,3,64],
            [3,3,64,64],
            [3,3,64,128],
            [3,3,128,128],
            [3,3,128,128],
            [3,3,128,128],
            [1,1,128,8]
        ]
        shapes_4 = [
            [3,3,128,256],
            [3,3,256,256],
            [3,3,256,256],
            [1,1,256,8]
        ]
        shapes_8 = [
            [3,3,256,512],
            [3,3,512,512],
            [3,3,512,512],
            [1,1,512,8]
        ]
        shapes_16 = [
            [3,3,512,512],
            [3,3,512,512],
            [3,3,512,512],
            [1,1,512,8]
        ]
        
        count = 0
        self.weights_2 = []
        for i in range( len( shapes_2 ) ):
            self.weights_2.append( self.get_weight( shapes_2[ i ] , 'weight{}'.format( count ) ) )
            count += 1
            
        self.weights_4 = []
        for i in range( len( shapes_4 ) ):
            self.weights_4.append( self.get_weight( shapes_4[ i ] , 'weight{}'.format( count ) ) )
            count += 1
            
        self.weights_8 = []
        for i in range( len( shapes_8 ) ):
            self.weights_8.append( self.get_weight( shapes_8[ i ] , 'weight{}'.format( count ) ) )
            count += 1
            
        self.weights_16 = []
        for i in range( len( shapes_16 ) ):
            self.weights_16.append( self.get_weight( shapes_16[ i ] , 'weight{}'.format( count ) ) )
            count += 1
            
    def batch_norm_wrapper(self, inputs, is_training, decay = 0.999):
        scale = tf.Variable(tf.ones([inputs.shape[-1]]))
        beta = tf.Variable(tf.zeros([inputs.shape[-1]]))
        pop_mean = tf.Variable(tf.zeros([inputs.shape[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.shape[-1]]), trainable=False)

        if is_training:
            batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
            train_mean = pop_mean.assign(pop_mean * decay + batch_mean * (1 - decay))
            train_var = pop_var.assign(pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                    batch_mean, batch_var, beta, scale, 1e-3)
        else:
            return tf.nn.batch_normalization(inputs,
                pop_mean, pop_var, beta, scale, 1e-3)
        
        # with tf.name_scope('bn'):
        #     beta = tf.Variable(tf.constant(0.0, shape=[inputs.shape[-1]]),
        #                                 name='beta', trainable=True)
        #     gamma = tf.Variable(tf.constant(1.0, shape=[inputs.shape[-1]]),
        #                                 name='gamma', trainable=True)
        #     batch_mean, batch_var = tf.nn.moments(inputs, [0,1,2], name='moments')
        #     ema = tf.train.ExponentialMovingAverage(decay)

        #     def mean_var_with_update():
        #         ema_apply_op = ema.apply([batch_mean, batch_var])
        #         with tf.control_dependencies([ema_apply_op]):
        #             return tf.identity(batch_mean), tf.identity(batch_var)

        #     mean, var = tf.cond(is_training,
        #                         mean_var_with_update,
        #                         lambda: (ema.average(batch_mean), ema.average(batch_var)))
        #     normalized = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
        
        # return normalized
        
    def model(self,input, training):
        x = tf.cast( input , dtype=tf.float32 )        
        with tf.name_scope("scale_2"):
            net = self.conv2d(x, self.weights_2[0], 1, 1, "layer1")
            #net = self.batch_norm_wrapper(net, training)
            net = self.conv2d(net, self.weights_2[1], 2, 1, "layer2")
            #net = self.batch_norm_wrapper(net, training)
            net = self.conv2d(net, self.weights_2[2], 1, 1, "layer3")
            #net = self.batch_norm_wrapper(net, training)
            net = self.conv2d(net, self.weights_2[3], 1, 1, "layer4")
            #net = self.batch_norm_wrapper(net, training)
            net = self.conv2d(net, self.weights_2[4], 1, 3, "layer5")
            #net = self.batch_norm_wrapper(net, training)
            net = self.conv2d(net, self.weights_2[5], 1, 6, "layer6")
            #net = self.batch_norm_wrapper(net, training)
            self.net_s_2 = self.conv2d(net, self.weights_2[6], 1, 1, "output_2", False)
        
        net = self.maxpool(net, 2, 2, "layer7")
        #net = self.batch_norm_wrapper(net, training)

        with tf.name_scope("scale_4"):
            net = self.conv2d(net, self.weights_4[0], 1, 1, "layer8")
            #net = self.batch_norm_wrapper(net, training)
            net = self.conv2d(net, self.weights_4[1], 1, 1, "layer9")
            #net = self.batch_norm_wrapper(net, training)
            net = self.conv2d(net, self.weights_4[2], 1, 3, "layer10")
            #net = self.batch_norm_wrapper(net, training)
            self.net_s_4 = self.conv2d(net, self.weights_4[3], 1, 1, "outpu_4", False)
        
        net = self.maxpool(net, 2, 2, "layer11")
        #net = self.batch_norm_wrapper(net, training)

        with tf.name_scope("scale_8"):
            net = self.conv2d(net, self.weights_8[0], 1, 1, "layer12")
            #net = self.batch_norm_wrapper(net, training)
            net = self.conv2d(net, self.weights_8[1], 1, 1, "layer13")
            #net = self.batch_norm_wrapper(net, training)
            net = self.conv2d(net, self.weights_8[2], 1, 3, "layer14")
            #net = self.batch_norm_wrapper(net, training)
            self.net_s_8 = self.conv2d(net, self.weights_8[3], 1, 1, "outpu_8", False)
            
        net = self.maxpool(net, 2, 2, "layer15")
        #net = self.batch_norm_wrapper(net, training)
        
        with tf.name_scope("scale_16"):
            net = self.conv2d(net, self.weights_16[0], 1, 1, "layer16")
            #net = self.batch_norm_wrapper(net, training)
            net = self.conv2d(net, self.weights_16[1], 1, 1, "layer17")
            #net = self.batch_norm_wrapper(net, training)
            net = self.conv2d(net, self.weights_16[2], 1, 3, "layer18")
            #net = self.batch_norm_wrapper(net, training)
            self.net_s_16 = self.conv2d(net, self.weights_16[3], 1, 1, "outpu_16", False)
            
    def network_loss_function(self, labels):
                        
        with tf.name_scope('loss_2_a'):
            loss_2_model = NetworkLoss(self.BatchSize, 2, 0.3, 0.33, 2.0)
            loss_2_result = loss_2_model.compute_loss(self.net_s_2,labels)
                      
        with tf.name_scope('loss_4_a'):
            loss_4_model = NetworkLoss(self.BatchSize, 2, 0.3, 0.33, 4.0)
            loss_4_result = loss_4_model.compute_loss(self.net_s_4,labels)
            
        with tf.name_scope('loss_8_a'):
            loss_8_model = NetworkLoss(self.BatchSize, 2, 0.3, 0.33, 8.0)
            loss_8_result = loss_8_model.compute_loss(self.net_s_8,labels)
                    
        with tf.name_scope('loss_16_a'):
            loss_16_model = NetworkLoss(self.BatchSize, 2, 0.3, 0.33, 16.0)
            loss_16_result = loss_16_model.compute_loss(self.net_s_16,labels)           
   
        return loss_2_result, loss_4_result, loss_8_result, loss_16_result

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

    def init_optimizers(self, learning):
        self.optimizer_2 = tf.optimizers.Adam(name="adam_optimizer_2",learning_rate=learning)
        self.optimizer_4 = tf.optimizers.Adam(name="adam_optimizer_4",learning_rate=learning)
        self.optimizer_8 = tf.optimizers.Adam(name="adam_optimizer_8",learning_rate=learning)
        self.optimizer_16 = tf.optimizers.Adam(name="adam_optimizer_16",learning_rate=learning)
    
    
    #@tf.function
    def train_step(self, model, inputs, label):
            
        if self.learning_changed:
            self.init_optimizers(self.learning)
        

        # GradientTape need to be persistent because we want to compute multiple gradients and it is no allowed by default
        with tf.GradientTape(persistent=True) as tape:
            model(inputs,self.is_training)
            loss_2, loss_4, loss_8, loss_16 = self.network_loss_function(label)
            loss = tf.reduce_sum([loss_2,loss_4,loss_8, loss_16],name="global_loss")
            
        # var_scale_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='scale_2')
        # var_scale_4 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='scale_4')
        # var_scale_8 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='scale_8')
        # var_scale_16 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='scale_16')
        # update_ops = tf.Graph().get_collection('update_ops')
        # with tf.control_dependencies(update_ops):
        grads_2 = tape.gradient( loss_2 , self.weights_2 )
        grads_4 = tape.gradient( loss_4 , self.weights_4 )
        grads_8 = tape.gradient( loss_8 , self.weights_8 )
        grads_16 = tape.gradient( loss_16 , self.weights_16 )
        
        # after gradient computation, we delete GradientTape object so it could be garbage collected
        del tape
        
        self.optimizer_2.apply_gradients(zip(grads_2,self.weights_2))
        self.optimizer_4.apply_gradients(zip(grads_4,self.weights_4))
        self.optimizer_8.apply_gradients(zip(grads_8,self.weights_8))
        self.optimizer_16.apply_gradients(zip(grads_16,self.weights_16))
        
        return loss
     
    def train(self, loader):
        
        test_acc = 1
        epoch = 1
        self.init_weights()
        # tf.train.write_graph(session.graph_def,r"C:\Users\Lukas\Documents\Object detection\model",'model.pbtxt')
        # writer = tf.summary.FileWriter(r"C:\Users\Lukas\Documents\Object detection\model", session.graph)
        learning = cfg.LEARNING_RATE
        update_edge = 0.1
        learning_changed = True
        while test_acc > cfg.MAX_ERROR:
            self.learning = learning
            self.is_training = True
            self.learning_changed = True
            for i in range(cfg.ITERATIONS):                
                image_batch, labels_batch, = loader.get_train_data(self.BatchSize)
                # train_fn = self.train_step_fn()        
                _ = self.train_step(self.model, image_batch, labels_batch)
                self.learning_changed = False
            
            self.is_training = False
            image_batch, labels_batch, = loader.get_train_data(self.BatchSize)
            self.model(image_batch,False)
            loss_2, loss_4, loss_8, loss_16 = self.network_loss_function(labels_batch)
            test_acc = tf.reduce_sum([loss_2,loss_4,loss_8, loss_16],name="global_loss") 
            print("Epoch:", (epoch), "test error: {:.5f}".format(test_acc))
            epoch += 1
            
            if test_acc < update_edge:
                self.learning = learning / 10
                update_edge = update_edge / 10
                self.learning_changed = True
                print("Learning rate updated to", learning)

    # def train(self, loader):
            
    #     graph_path = os.path.dirname(os.path.abspath(__file__)) + r"\graphs\tensorboard"
    #     iterations = cfg.ITERATIONS
        
    #     image_placeholder = tf.placeholder(tf.float32, [None, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_CHANNELS],name="input_image_placeholder")
    #     labels_placeholder = tf.placeholder(tf.float32, [None, None, 10], name="input_label_placeholder")
    #     self.learning = tf.placeholder(tf.float32,(),name="input_learning_rate")
    #     #self.is_training = tf.placeholder(tf.bool, name='input_is_training_placeholder')
    #     #self.is_training = tf.placeholder_with_default(cfg.IS_TRAINING, (), name='input_is_training_placeholder')  

    #     self.create_detection_network_2(image_placeholder)        
    #     loss_2, loss_4, loss_8, loss_16 = self.network_loss_function(labels_placeholder)
    #     loss = tf.reduce_sum([loss_2,loss_4,loss_8, loss_16],name="global_loss")    
        
    #     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #     with tf.control_dependencies(update_ops):
    #         optimize = self.network_otimizer(loss)
        
    #     saver = tf.train.Saver(name='model_saver')
    #     init = tf.global_variables_initializer()
        
    #     with tf.Session() as session:
    #         session.run(init)

    #         test_acc = 1
    #         epoch = 1
    #         tf.train.write_graph(session.graph_def,r"C:\Users\Lukas\Documents\Object detection\model",'model.pbtxt')
    #         writer = tf.summary.FileWriter(r"C:\Users\Lukas\Documents\Object detection\model", session.graph)
    #         learning = cfg.LEARNING_RATE
    #         update_edge = 0.1
    #         print("Learning rate for AdamOtimizer", learning)
    #         while test_acc > cfg.MAX_ERROR:
    #             for i in range(iterations):
    #                 image_batch, labels_batch, = loader.get_train_data(self.BatchSize)                    
    #                 session.run(optimize, 
    #                                 feed_dict={image_placeholder: image_batch, labels_placeholder: labels_batch, self.is_training: True, self.learning: learning})

    #             image_batch, labels_batch = loader.get_train_data(self.BatchSize)
    #             test_acc = session.run(loss, 
    #                             feed_dict={image_placeholder: image_batch, labels_placeholder: labels_batch, self.is_training: False, self.learning: learning})
                
    #             print("Epoch:", (epoch), "test error: {:.5f}".format(test_acc))
    #             epoch += 1

    #             if math.isnan(test_acc):
    #                 s_2, s_4, s_8, s_16 = session.run([self.net_s_2,self.net_s_4,self.net_s_8,self.net_s_16], 
    #                             feed_dict={image_placeholder: image_batch[0], labels_placeholder: labels_batch[0], self.is_training: False, self.learning: learning})
    #                 np.save(r"C:\Users\Lukas\Documents\Object detection\test\train_s_2.txt",s_2)
    #                 np.save(r"C:\Users\Lukas\Documents\Object detection\test\train_s_4.txt",s_4)
    #                 np.save(r"C:\Users\Lukas\Documents\Object detection\test\train_s_8.txt",s_8)
    #                 np.save(r"C:\Users\Lukas\Documents\Object detection\test\train_s_16.txt",s_16)
    #                 break
                
    #             if test_acc < update_edge:
    #                 learning = learning / 10
    #                 update_edge = update_edge / 10
    #                 print("Learning rate updated to", learning)

    #             # if epoch % 20 == 0:
    #             #     learning = learning / 10
    #             #     print("Learning rate updated to", learning)
                
    #         saver.save(session, cfg.MODEL_PATH)            
    #         freeze.freeze_and_save()              
            
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



    