import tensorflow as tf

from Network.network_loss import NetworkLoss
from Network.object_detection_network import ObjectDetectionModel,ODM_MaxPool_Layer,ODM_Conv2D_Layer

tf.keras.layers.BatchNormalization._USE_V2_BEHAVIOR = False

import config as cfg
import os
import cv2
import numpy as np
import Services.helper as help
import math
import copy   
                         
class NetworkCreator():
    
    def __init__(self):
        self.device = cfg.DEVICE_NAME
        self.BatchSize = cfg.BATCH_SIZE
                     
    def network_loss_function(self,out_2, out_4, out_8, out_16, labels):
                        
        with tf.name_scope('loss_2_a'):
            loss_2_model = NetworkLoss(self.BatchSize, 2.0, "loss_scale_2")
            loss_2_result = loss_2_model(out_2,labels)
                      
        with tf.name_scope('loss_4_a'):
            loss_4_model = NetworkLoss(self.BatchSize, 4.0, "loss_scale_2")
            loss_4_result = loss_4_model(out_4,labels)
            
        with tf.name_scope('loss_8_a'):
            loss_8_model = NetworkLoss(self.BatchSize, 8.0, "loss_scale_2")
            loss_8_result = loss_8_model(out_8,labels)
                    
        with tf.name_scope('loss_16_a'):
            loss_16_model = NetworkLoss(self.BatchSize, 16.0, "loss_scale_2")
            loss_16_result = loss_16_model(out_16,labels)           
   
        return loss_2_result, loss_4_result, loss_8_result, loss_16_result
    
    def train_step(self, model, inputs, label):
            
        # GradientTape need to be persistent because we want to compute multiple gradients and it is no allowed by default
        with tf.GradientTape(persistent=True) as tape:
            out_2, out_4, out_8, out_16 = model(inputs,True)
            loss_2, loss_4, loss_8, loss_16 = self.network_loss_function(out_2, out_4, out_8, out_16, label)
            loss = tf.reduce_sum([loss_2,loss_4,loss_8, loss_16],name="global_loss")
            
        grads_2 = tape.gradient( loss_2 , model.out_2_trainable_variables )
        grads_4 = tape.gradient( loss_4 , model.out_4_trainable_variables )
        grads_8 = tape.gradient( loss_8 , model.out_8_trainable_variables )
        grads_16 = tape.gradient( loss_16 , model.out_16_trainable_variables )
        
        
        self.optimizer_2.apply_gradients(zip(grads_2,model.out_2_trainable_variables))
        self.optimizer_4.apply_gradients(zip(grads_4,model.out_4_trainable_variables))
        self.optimizer_8.apply_gradients(zip(grads_8,model.out_8_trainable_variables))
        self.optimizer_16.apply_gradients(zip(grads_16,model.out_16_trainable_variables))
        
        # after gradient computation, we delete GradientTape object so it could be garbage collected
        del tape
        return loss
     
    def get_learning_rate(self):
        return self.learning
    
    def train(self, loader):
        
        self.learning = cfg.LEARNING_RATE
        test_acc = 1
        epoch = 1 
        
        self.optimizer_2 = tf.optimizers.Adam(name="adam_optimizer_2",learning_rate=self.get_learning_rate)
        self.optimizer_4 = tf.optimizers.Adam(name="adam_optimizer_4",learning_rate=self.get_learning_rate)
        self.optimizer_8 = tf.optimizers.Adam(name="adam_optimizer_8",learning_rate=self.get_learning_rate)
        self.optimizer_16 = tf.optimizers.Adam(name="adam_optimizer_16",learning_rate=self.get_learning_rate)        
        
        update_edge = 0.01
        model = ObjectDetectionModel([3,3],'Object Detection Model')
        model.build(input_shape=(None,cfg.IMG_HEIGHT,cfg.IMG_WIDTH,cfg.IMG_CHANNELS))
        
        # image_batch, labels_batch, = loader.get_train_data(self.BatchSize)    
        # model._set_inputs(image_batch)
        # model.save(cfg.MODEL_PATH_PB,save_format="tf")         
        
        # new_model = tf.keras.models.load_model(cfg.MODEL_PATH_H5)
        # print(new_model.summary())
        
        # print(model.summary())
        # ['odm__conv2d__layer_5', 'odm__conv2d__layer_9', 'odm__conv2d__layer_13', 'odm__conv2d__layer_17']
        # model.compile(optimizer=tf.optimizers.Adam(name="adam_optimizer",learning_rate=self.get_learning_rate),
        #               loss = {"odm__conv2d__layer_5":NetworkLoss(self.BatchSize, 2.0, "loss_scale_2"),
        #                       "odm__conv2d__layer_9":NetworkLoss(self.BatchSize, 4.0, "loss_scale_4"),
        #                       "odm__conv2d__layer_13":NetworkLoss(self.BatchSize, 8.0, "loss_scale_8"),
        #                       "odm__conv2d__layer_17":NetworkLoss(self.BatchSize, 16.0, "loss_scale_16")},
        #               loss_weights={'odm__conv2d__layer_5':1.,'odm__conv2d__layer_9':1.,'odm__conv2d__layer_13':1.,'odm__conv2d__layer_17':1.},
        #               metrics={"odm__conv2d__layer_5":'accuracy',"odm__conv2d__layer_9":'accuracy',"odm__conv2d__layer_13":'accuracy',"odm__conv2d__layer_17":'accuracy'})
        
        # image_batch, labels_batch, = loader.get_train_data(4)
        # history = model.fit(image_batch,[labels_batch,labels_batch,labels_batch,labels_batch],
        #                     batch_size=2,epochs=10,verbose=2)
        
        errors = []
        while test_acc > cfg.MAX_ERROR:
            for i in range(cfg.ITERATIONS):                
                # train_fn = self.train_step_fn()  
                image_batch, labels_batch, = loader.get_train_data(self.BatchSize)    
                _ = self.train_step(new_model, image_batch, labels_batch)
            
            image_batch, labels_batch, = loader.get_train_data(self.BatchSize)
            out_2, out_4, out_8, out_16 = new_model(image_batch,False)
            loss_2, loss_4, loss_8, loss_16 = self.network_loss_function(out_2, out_4, out_8, out_16, labels_batch)
            test_acc = tf.reduce_sum([loss_2,loss_4,loss_8, loss_16],name="global_loss") 
            errors.append(test_acc)
            print("Epoch:", (epoch), "test error: {:.5f}".format(test_acc))
            if epoch == 10:
                break
                
            epoch += 1
            
            if test_acc < update_edge:
                self.learning = self.learning / 10
                update_edge = update_edge / 10
                print("Learning rate updated to", self.learning)  
                
        new_model.save_weights(cfg.MODEL_PATH_H5)

 
               
       
    
       
            
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



    