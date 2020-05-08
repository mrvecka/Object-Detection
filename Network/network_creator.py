import tensorflow as tf
import tensorflow.keras.backend as K
from Network.network_loss import NetworkLoss
from Network.object_detection_network import ObjectDetectionModel 
from Services.timer import Timer

import config as cfg
import os
import sys
import cv2
import numpy as np
import math
import copy   
import keras2onnx
                         
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
    
    @tf.function
    def compute_gradient(self, model, inputs, targets):
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
        
        tf.keras.backend.set_floatx('float32')
        
        epoch = 1
        acc = [9,9,9,9]
        t_global = Timer()
        t_global.start()

        while self.continue_training(acc):
            epoch_loss_avg = []
            t = Timer()
            t.start()
            for i in range(cfg.ITERATIONS):                
                image_batch, labels_batch, _ = loader.get_train_data(self.BatchSize) 
                loss_value = self.compute_gradient(model,image_batch, labels_batch)
                self.check_computed_loss(loss_value)
                
                epoch_loss_avg.append(loss_value)

            _ = t.stop()
            acc = np.mean(epoch_loss_avg, axis=0)
            print("Epoch {:d}: Loss 2: {:.6f}, Loss 4: {:.6f}, Loss 8: {:.6f}, Loss 16: {:.6f}  Epoch duration: ".format(epoch,acc[0],acc[1],acc[2],acc[3]) + t.get_formated_time())
            
            self.save_model(model, epoch)
            self.update_learning_rate(epoch)               
            epoch += 1
                           
        _ = t_global.stop()
        self.save_model(model, 0)
    
    def check_computed_loss(self, loss_values):
        if self.loss_is_nan(loss_values):
            print("One of loss values is NaN, program will be terminated!")
            print(loss_values)
            # print(names)
            sys.exit()
     
    def update_learning_rate(self, epoch):
        if epoch in cfg.UPDATE_LEARNING_RATE:
            self.learning = self.learning / 10

    
    def save_model(self, model, epoch):
        if epoch % cfg.SAVE_MODEL_EVERY == 0:
            model.save_weights(cfg.MODEL_WEIGHTS)
            
            model_json = model.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)
    
     
    def start_train(self, loader):
        
        model = ObjectDetectionModel([3,3],'ObjectDetectionModel')
        self.train(loader,model)
        model.summary()
        




    