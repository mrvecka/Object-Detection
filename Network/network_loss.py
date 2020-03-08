import tensorflow as tf
import config as cfg
import cv2
import numpy as np

class NetworkLoss(tf.keras.losses.Loss):
    def __init__(self,batch, scale, loss_name, reduction=tf.keras.losses.Reduction.AUTO):
        super().__init__(reduction=reduction, name=loss_name)
        self.radius = cfg.RADIUS
        self.circle_ratio = cfg.CIRCLE_RATIO
        self.boundaries = cfg.BOUNDARIES
        self.weight_factor = cfg.WEIGHT_FACTOR
        self.scale = scale
        self.batch_size = batch
    
    def get_config(self):
        base_config = super().get_config()
        config = {'batch':self.batch,'scale':self.scale,'loss_name':self.loss_name,'reduction':tf.keras.losses.Reduction.AUTO}
        return dict(list(base_config.items()) + list(config.items()))
    
    def call(self,images,labels):
        errors = []
        for i in range(self.batch_size):          
            current_img = images[i]
            current_lbl = labels[i]
            img_error = self.scan_image_function(current_img, current_lbl)
            errors.append(img_error)

        errors_as_tensor = tf.convert_to_tensor(errors,dtype=tf.float32)
        loss = tf.reduce_sum(errors_as_tensor)
        return loss
    
    def scan_image_function(self, image, label):

        width = image.shape.dims[1].value
        height = image.shape.dims[0].value
        channels = image.shape.dims[2].value
        
        target = self.create_target_response_map(label, width, height)        
        target = tf.reshape(target,(channels,height,width))        
        
        # number of neurons in each output layer
        N = width * height

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
                
        maps = cv2.split(np.zeros((height,width,8)))
        bound_above, bound_below, ideal = self.GetObjectBounds()
        for i in range(len(labels)):            
            label = labels[i]
            if label[0] == -1:
                continue
            # 0       1       2       3       4       5       6     7           8           9
            # fblx    fbly    fbrx    fbry    rblx    rbly    ftly  center_x    center_y    largest_dim
            
            if label[9] >= bound_below and label[9] <= bound_above:
                x = int(label[7] / self.scale)
                y = int(label[8] / self.scale)
                
                scaling_ratio = 1.0 / self.scale                
                cv2.circle(maps[0], ( x, y ), int(self.radius), 1, -1)
                cv2.GaussianBlur(maps[0], (3, 3), 100)

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