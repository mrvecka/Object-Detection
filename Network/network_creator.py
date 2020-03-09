import tensorflow as tf

from Network.network_loss import NetworkLoss
from Network.object_detection_network import ObjectDetectionModel,ODM_MaxPool_Layer,ODM_Conv2D_Layer
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
        
        self.model = None
        self.optimizer = None
        self.network_loss = None
    
    def train_step(self, inputs, label):
            
        # GradientTape need to be persistent because we want to compute multiple gradients and it is no allowed by default
        # persistent=True
        with tf.GradientTape() as tape:
            out_2, out_4, out_8, out_16 = self.model(inputs,True)
            loss = self.network_loss([out_2, out_4, out_8, out_16], label)
            
        grads_2 = tape.gradient( loss , self.model.trainable_variables)
        
        # after gradient computation, we delete GradientTape object so it could be garbage collected        
        self.optimizer.apply_gradients(zip(grads_2, self.model.trainable_variables))

        
        return loss
    
    def get_learning_rate(self):
        return self.learning
    
    def train(self, loader,test_acc,epoch,update_edge,max_error):
        

        iteration = cfg.ITERATIONS
        errors = []
        t_global = Timer()
        t_global.start()
        while test_acc > max_error:
            t = Timer()
            t.start()
            for i in range(iteration):                
                # train_fn = self.train_step_fn()  
                image_batch, labels_batch, = loader.get_train_data(self.BatchSize)    
                _ = self.train_step(image_batch, labels_batch)

                #print("Iteration: ",i)
            
            image_batch, labels_batch, = loader.get_train_data(self.BatchSize)
            out_2, out_4, out_8, out_16 = self.model(image_batch,False)
            test_acc = self.network_loss([out_2, out_4, out_8, out_16], labels_batch)
            acc = test_acc.numpy()
            errors.append(acc)
            _ = t.stop()
            print(f"Epoch: {epoch:4d} test error: {acc:0.5f} Epoch duration: " + t.get_formated_time()) # make time hh:mm:ss
                
            epoch += 1
            
            # if test_acc < update_edge:
            #     self.learning = self.learning / 10
            #     update_edge = update_edge / 10
            #     print("Learning rate updated to", self.learning) 
                
        acc = test_acc.numpy()
        _ = t_global.stop()
        print(f"Final test error: {acc:0.5f} Training duration: " + t_global.get_formated_time())
        print(errors) 
     
     
    def start_train(self, loader):
        
        self.learning = cfg.LEARNING_RATE

        self.optimizer = tf.optimizers.Adam(name="adam_optimizer_2",learning_rate=self.get_learning_rate) 
        self.model = ObjectDetectionModel([3,3],'ObjectDetectionModel')
        self.network_loss = NetworkLoss( "loss_function")
        self.model.compile()
        self.train(loader,1,1,cfg.UPDATE_EDGE,cfg.MAX_ERROR)
        self.model.save_weights(cfg.MODEL_WEIGHTS)

                      
            
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



    