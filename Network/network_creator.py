import tensorflow as tf

from Network.network_loss import NetworkLoss
from Network.object_detection_network import ObjectDetectionModel,ODM_MaxPool_Layer,ODM_Conv2D_Layer

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
         
    def network_loss_function(self,out_2, out_4, out_8, out_16, labels):
                        
        # tf.config.experimental_run_functions_eagerly(True)
        with tf.name_scope('loss_2'):
            loss_2_model = NetworkLoss(self.BatchSize, 2.0, "loss_scale_2")
            loss_2_result = loss_2_model(out_2,labels)
                  
        del loss_2_model    
        with tf.name_scope('loss_4'):
            loss_4_model = NetworkLoss(self.BatchSize, 4.0, "loss_scale_4")
            loss_4_result = loss_4_model(out_4,labels)
        
        del loss_4_model  
        with tf.name_scope('loss_8'):
            loss_8_model = NetworkLoss(self.BatchSize, 8.0, "loss_scale_8")
            loss_8_result = loss_8_model(out_8,labels)
            
        del loss_8_model        
        with tf.name_scope('loss_16'):
            loss_16_model = NetworkLoss(self.BatchSize, 16.0, "loss_scale_16")
            loss_16_result = loss_16_model(out_16,labels)           
        # tf.config.experimental_run_functions_eagerly(False)
   
        del loss_16_model
        return loss_2_result, loss_4_result, loss_8_result, loss_16_result
    
    def train_step(self, model, inputs, label, optimizers):
            
        # GradientTape need to be persistent because we want to compute multiple gradients and it is no allowed by default
        with tf.GradientTape(persistent=True) as tape:
            out_2, out_4, out_8, out_16 = model(inputs,True)
            loss_2, loss_4, loss_8, loss_16 = self.network_loss_function(out_2, out_4, out_8, out_16, label)
            loss = tf.reduce_sum([loss_2,loss_4,loss_8, loss_16],name="global_loss")
            
        grads_2 = tape.gradient( loss_2 , model.out_2_trainable_variables)
        grads_4 = tape.gradient( loss_4 , model.out_4_trainable_variables )
        grads_8 = tape.gradient( loss_8 , model.out_8_trainable_variables )
        grads_16 = tape.gradient( loss_16 , model.out_16_trainable_variables )
        
        # after gradient computation, we delete GradientTape object so it could be garbage collected
        del tape
        
        optimizers[0].apply_gradients(zip(grads_2,model.out_2_trainable_variables))
        optimizers[1].apply_gradients(zip(grads_4,model.out_4_trainable_variables))
        optimizers[2].apply_gradients(zip(grads_8,model.out_8_trainable_variables))
        optimizers[3].apply_gradients(zip(grads_16,model.out_16_trainable_variables))
        
        return loss
    
    
    def get_learning_rate(self):
        return self.learning
    
    
    def train(self, model, loader, optimizers,test_acc,epoch,update_edge,max_error):
        

        iteration = cfg.ITERATIONS
        errors = []
        while test_acc > max_error:
            for i in range(iteration):                
                # train_fn = self.train_step_fn()  
                image_batch, labels_batch, = loader.get_train_data(self.BatchSize)    
                _ = self.train_step(model, image_batch, labels_batch,optimizers)
                #print("Iteration: ",i)
            
            image_batch, labels_batch, = loader.get_train_data(self.BatchSize)
            out_2, out_4, out_8, out_16 = model(image_batch,False)
            loss_2, loss_4, loss_8, loss_16 = self.network_loss_function(out_2, out_4, out_8, out_16, labels_batch)
            test_acc = tf.reduce_sum([loss_2,loss_4,loss_8, loss_16],name="global_loss") 
            errors.append(test_acc.numpy())
            print("Epoch:", (epoch), "test error: ", test_acc.numpy())
                
            epoch += 1
            
            if test_acc < update_edge:
                self.learning = self.learning / 10
                update_edge = update_edge / 10
                print("Learning rate updated to", self.learning) 
                
        print(errors) 
     
     
    def start_train(self, loader):
        
        self.learning = cfg.LEARNING_RATE

        optimizer_2 = tf.optimizers.Adam(name="adam_optimizer_2",learning_rate=self.get_learning_rate)
        optimizer_4 = tf.optimizers.Adam(name="adam_optimizer_4",learning_rate=self.get_learning_rate)
        optimizer_8 = tf.optimizers.Adam(name="adam_optimizer_8",learning_rate=self.get_learning_rate)
        optimizer_16 = tf.optimizers.Adam(name="adam_optimizer_16",learning_rate=self.get_learning_rate) 
          
        model = ObjectDetectionModel([3,3],'ObjectDetectionModel')        
        model.compile()
        self.train(model, loader, [optimizer_2,optimizer_4,optimizer_8,optimizer_16],1,1,0.01,cfg.MAX_ERROR)
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



    