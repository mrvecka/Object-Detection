import tensorflow as tf
import tensorflow.keras.backend as K
from Network.network_loss import NetworkLoss
from Network.object_detection_network import ObjectDetectionModel, ObjectDetectionModel2 
from Services.timer import Timer

import config as cfg
import os
import cv2
import numpy as np
import math
import copy   
                         
class NetworkCreator():
    
    def __init__(self):
        self.device = cfg.DEVICE_NAME
        self.BatchSize = cfg.BATCH_SIZE
        
        self.learning = cfg.LEARNING_RATE
        
        self.optimizer2 = tf.optimizers.Adam(name="adam_optimizer_2",learning_rate=self.get_learning_rate)
        self.optimizer4 = tf.optimizers.Adam(name="adam_optimizer_4",learning_rate=self.get_learning_rate)
        self.optimizer8 = tf.optimizers.Adam(name="adam_optimizer_8",learning_rate=self.get_learning_rate)
        self.optimizer16 = tf.optimizers.Adam(name="adam_optimizer_16",learning_rate=self.get_learning_rate)
        
        self.loss2 = NetworkLoss( "loss_function_2",2)
        self.loss4 = NetworkLoss( "loss_function_4",4)
        self.loss8 = NetworkLoss( "loss_function_8",8)
        self.loss16 = NetworkLoss( "loss_function_16",16)

    
    def train_step(self, inputs, label):
            
        # GradientTape need to be persistent because we want to compute multiple gradients and it is no allowed by default
        # persistent=True
        with tf.GradientTape() as tape:
            out_2, out_4, out_8, out_16 = self.model.train_on_batch(inputs)
            loss = self.network_loss[0]([out_2, out_4, out_8, out_16], label)
            
        grads = tape.gradient( loss , self.model.trainable_variables)
        self.optimizer[0].apply_gradients(zip(grads, self.model.trainable_variables))
        
        return loss
    
    @tf.function
    def compute_loss(self, label, output):
        loss_2 = self.loss2(label[0], output[0])
        loss_4 = self.loss4(label[1], output[1])
        loss_8 = self.loss8(label[2], output[2])
        loss_16 = self.loss16(label[3], output[3])
        return [loss_2, loss_4, loss_8, loss_16]
    
    @tf.function
    def loss(self,model,x,y,training):
        y_ = model(x,training)
        loss = self.compute_loss(y, y_)

        return loss
    
    def compute_gradient(self, model, inputs, targets):
        #print(model.block_2)
        with tf.GradientTape(persistent= True) as tape:

            loss_value = self.loss(model, inputs, targets, training=True)
         
        grad2 = tape.gradient(loss_value[0], model.block_2)
        grad4 = tape.gradient(loss_value[1], model.block_4)
        grad8 = tape.gradient(loss_value[2], model.block_8)
        grad16 = tape.gradient(loss_value[3], model.block_16)
        del tape  
        
        self.optimizer2.apply_gradients(zip(grad2, model.block_2)) 
        self.optimizer4.apply_gradients(zip(grad4, model.block_4)) 
        self.optimizer8.apply_gradients(zip(grad8, model.block_8)) 
        self.optimizer16.apply_gradients(zip(grad16, model.block_16)) 
       
        return loss_value
    
    def get_learning_rate(self):
        return self.learning
    
    def continue_training(self, acc):
       
        if acc[0] > cfg.MAX_ERROR or acc[1] > cfg.MAX_ERROR or acc[2] > cfg.MAX_ERROR or acc[3] > cfg.MAX_ERROR:
            return True
        else:
            return False 
        
    def loss_is_nan(self, acc):
        if np.isnan(acc[0]) or np.isnan(acc[1]) or np.isnan(acc[2]) or np.isnan(acc[3]):
            return True
        else:
            return False 
    
    def train(self, loader, model):
        
        update_edge = cfg.UPDATE_EDGE
        iteration = cfg.ITERATIONS
        epoch = 1
        acc = [9,9,9,9]
        errors = []
        t_global = Timer()
        t_global.start()

        while self.continue_training(acc):
            epoch_loss_avg = []
            t = Timer()
            t.start()
            for i in range(iteration):                
                # train_fn = self.train_step_fn()  
                image_batch, labels_batch, names = loader.get_train_data(self.BatchSize) 
                
                # update_ops = tf.Graph().get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
                # with tf.control_dependencies(update_ops):
                loss_value = self.compute_gradient(model,image_batch, labels_batch)
                if self.loss_is_nan(loss_value):
                    print(loss_value)
                    print(names)
                    model.save_weights(cfg.MODEL_WEIGHTS)
                    return
                
                epoch_loss_avg.append(loss_value)
                # _ = self.model.train_on_batch(image_batch,[labels_batch,labels_batch,labels_batch,labels_batch])
                # loss = self.network_loss[0]([out_2, out_4, out_8, out_16], label)

                #print("Iteration: ",i)
            
            # image_batch, labels_batch, = loader.get_train_data(self.BatchSize)
            # losses = self.model.test_on_batch(image_batch,[labels_batch,labels_batch,labels_batch,labels_batch])
            # acc = tf.reduce_sum(losses).numpy()

            # acc = self.network_loss[0](out, labels_batch)
            #errors.append(acc)
            _ = t.stop()
            acc = np.mean(epoch_loss_avg, axis=0)
            print("Epoch {:d}: Loss 2: {:.6f}, Loss 4: {:.6f}, Loss 8: {:.6f}, Loss 16: {:.6f}  Epoch duration: ".format(epoch,acc[0],acc[1],acc[2],acc[3]) + t.get_formated_time())
            model.save_weights(cfg.MODEL_WEIGHTS)

            if epoch == 30:
                self.learning = self.learning / 10
            if epoch == 80:
                self.learning = self.learning / 10
                

            #print(f"Epoch: {epoch:4d} test error: {acc:0.5f} Epoch duration: " + t.get_formated_time()) # make time hh:mm:ss
            # if acc < update_edge:
            #     self.learning = self.learning /10
            #     update_edge = update_edge /10
            #     print("learning rate changed")
            epoch += 1
            
                
        _ = t_global.stop()
        #print(f"Final test error: {acc:0.5f} Training duration: " + t_global.get_formated_time())
        print(errors) 
     
     
    def start_train(self, loader):
        

        # self.optimizer = [tf.optimizers.Adam(name="adam_optimizer_2",learning_rate=self.get_learning_rate),
        # tf.optimizers.Adam(name="adam_optimizer_4",learning_rate=self.get_learning_rate),
        # tf.optimizers.Adam(name="adam_optimizer_8",learning_rate=self.get_learning_rate),
        # tf.optimizers.Adam(name="adam_optimizer_16",learning_rate=self.get_learning_rate)] 
        model = ObjectDetectionModel2([3,3],'ObjectDetectionModel')
        # model.build((None,cfg.IMG_HEIGHT,cfg.IMG_WIDTH,cfg.IMG_CHANNELS))

        # model.build((cfg.BATCH_SIZE,cfg.IMG_HEIGHT,cfg.IMG_WIDTH,cfg.IMG_CHANNELS))
        # model.build(tf.keras.Input(shape=(cfg.IMG_HEIGHT,cfg.IMG_WIDTH,cfg.IMG_CHANNELS)))
        # model.summary()
        # self.model.compile(optimizer=tf.optimizers.Adam(name="adam_optimizer",learning_rate=self.get_learning_rate),loss=[NetworkLoss( "loss_function",2),NetworkLoss( "loss_function",4),
        #                      NetworkLoss( "loss_function",8),NetworkLoss( "loss_function",16)])
        # _model.summary()
        
        # x_train, y_train = loader.get_prepared_data()
        # steps_per_epoch=cfg.ITERATIONS
        # self.model.fit(x_train,[y_train,y_train,y_train,y_train],2,epochs=30)
        # self.network_loss = [NetworkLoss( "loss_function",2),NetworkLoss( "loss_function",4),
        #                      NetworkLoss( "loss_function",8),NetworkLoss( "loss_function",16)]
        # self.network_loss = NetworkLoss( "loss_function",2)
        self.train(loader,model)
        model.summary()
        model.save_weights(cfg.MODEL_WEIGHTS)
        

                      
            
    def save_results(self, maps, scale):
        result = cv2.split(np.squeeze(maps,axis=0))
        
            
        base_path = r".\.\result_test_s" + str(scale)
        if not fw.check_and_create_folder(base_path):
            print("Unable to create folder for results. Tried path: ", base_path)
            return
        
        path = base_path+r"\response_map_0.jpg"
        cv2.imwrite(path, (maps[0,:,:,0] - maps[0,:,:,0].min()) * (255/(maps[0,:,:,0].max() - maps[0,:,:,0].min())))
        path = base_path+r"\response_map_1.jpg"
        cv2.imwrite(path, 255* result[1])
        path = base_path+r"\response_map_2.jpg"
        cv2.imwrite(path, 255*result[2])
        path = base_path+r"\response_map_3.jpg"
        cv2.imwrite(path, 255*result[3])
        path = base_path+r"\response_map_4.jpg"
        cv2.imwrite(path, 255*result[4])
        path = base_path+r"\response_map_5.jpg"
        cv2.imwrite(path, 255*result[5])
        path = base_path+r"\response_map_6.jpg"
        cv2.imwrite(path, 255*result[6])
        path = base_path+r"\response_map_7.jpg"
        cv2.imwrite(path, 255*result[7])


    