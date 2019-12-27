from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorboard as tb
# from Network.network_creator import NetworkCreator
# import Models.bb3txt as bb
import cv2
import Services.loader as load

def GetObjectBounds(r, cr, bo, scale):
    ideal_size = (2 * r + 1) / cr * scale
    # bound above
    ext_above = ((1 - bo) * ideal_size) / 2 + bo * ideal_size
    bound_above = ideal_size + ext_above
    
    # bound below
    diff = ideal_size / 2
    ext_below = ((1 - bo)* diff) /2 + bo * diff
    bound_below = ideal_size - ext_below
    
    return bound_above, bound_below, ideal_size

def create_target_response_map(labels, r, circle_ratio, boundaries, scale, height, width):
                
        # maps => array of shape (channels, orig_height, orig_width) 
        maps = cv2.split(np.zeros((height,width,8)))
        # self.index = 0
        # result = tf.scan(self.scan_label_function, labels, initializer=0) 
        bound_above, bound_below, ideal = GetObjectBounds(r,circle_ratio,boundaries,scale)
        for i in range(len(labels)):            
            label = labels[i]
            if label[0] == -1:
                break
            # 0       1       2       3       4       5       6     7           8           9
            # fblx    fbly    fbrx    fbry    rblx    rbly    ftly  center_x    center_y    largest_dim
            
            #size = self.get_size_of_bounding_box(labels)
            if label[9] >= bound_below and label[9] <= bound_above:
                x = label[7] / scale
                y = label[8] / scale
                
                scaling_ratio = 1.0 / scale
                # print((self.orig_height,self.orig_width))
                #radius = ((circle_ration / scale) * szie ) - 1
                
                cv2.circle(maps[0], (int(x), int(y)), int(r), 255, -1)
                
                # x_acc = x * scaling_ratio
                # y_acc = y * scaling_ratio
                for c in range(1,7):
                    cv2.circle(maps[c], (int(x), int(y)), int(r), 1, -1)
                    cv2.GaussianBlur(maps[c], (3, 3), 100)
                    
                    for l in range(-r,r,1):
                        for j in range(-r,r,1):
                            xp = int(x) + j
                            yp = int(y) + l
                            
                            if xp >= 0 and xp < width and yp >= 0 and yp < height:
                                if maps[c][yp][xp] > 0.0:
                                    if c ==1 or c == 3 or c == 5:
                                        maps[c][yp][xp] = 0.5 + (label[c-1] - x - j * scale) / ideal
                                    elif c == 2 or c == 4 or c == 6 or c == 7:
                                        maps[c][yp][xp] = 0.5 + (label[c-1] - y - l * scale) / ideal

        result = cv2.merge(maps)
        
        return np.asarray(result,dtype=np.float32)

loader = load.Loader()
loader.load_specific_label("000046")

image_batch, labels_batch, image_paths, calib_matrices = loader.get_test_data(1)

maps2 = create_target_response_map(labels_batch[0], 2, 0.3, 0.33, 2,64,128)
# maps4 = create_target_response_map(labels_batch[0], 2, 0.3, 0.33, 4, 32,64)
# maps8 = create_target_response_map(labels_batch[0], 2, 0.3, 0.33, 8, 16,32)
# maps16 = create_target_response_map(labels_batch[0], 2, 0.3, 0.33, 16, 8,16)

first2 = maps2[:,:,0]
second2 = maps2[:,:,1]
# first4 = maps4[:,:,0]
# first8 = maps8[:,:,0]
# first16 = maps16[:,:,0]
cv2.imshow("response_2", first2)
cv2.imshow("response_2 second",second2)
# cv2.imshow("response_4", first4)
# cv2.imshow("response_8", first8)
# cv2.imshow("response_16", first16)
cv2.waitKey(0)
            

