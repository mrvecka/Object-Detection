import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
# from Network.network_creator import NetworkCreator
# import Models.bb3txt as bb
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

def euclidean_distance(p1, p2):
    tmp = pow(p2[0] - p1[0]) + pow(p2[1] - p1[1])
    return np.sqrt(tmp)

def scale_box_size():
    loader = load.Loader()
    loader.load_data()
    
    bound_above_2, bound_below_2, _ = GetObjectBounds(2,0.3,0.33,2)
    bound_above_4, bound_below_4, _ = GetObjectBounds(2,0.3,0.33,4)
    bound_above_8, bound_below_8, _ = GetObjectBounds(2,0.3,0.33,8)
    bound_above_16, bound_below_16, _ = GetObjectBounds(2,0.3,0.33,16)

    heights_scale_2 = []
    heights_scale_4 = []
    heights_scale_8 = []
    heights_scale_16 = []

    for im_model in loader.Data:        
        for label in im_model.labels:
            if label.largest_dim >= bound_below_2 and label.largest_dim <= bound_above_2:
                dist = np.abs(label.fbl_y - label.ftl_y)
                heights_scale_2.append(dist)
            if label.largest_dim >= bound_below_4 and label.largest_dim <= bound_above_4:
                dist = np.abs(label.fbl_y - label.ftl_y)
                heights_scale_4.append(dist)
            if label.largest_dim >= bound_below_8 and label.largest_dim <= bound_above_8:
                dist = np.abs(label.fbl_y - label.ftl_y)
                heights_scale_8.append(dist)
            if label.largest_dim >= bound_below_16 and label.largest_dim <= bound_above_16:
                dist = np.abs(label.fbl_y - label.ftl_y)
                heights_scale_16.append(dist)
                
    _min = np.mean(heights_scale_2) - 2*np.std(heights_scale_2)
    _max = np.mean(heights_scale_2) + 2*np.std(heights_scale_2)
    print("heights 2 min:",_min," max:",_max)
    
    _min = 0
    _max = 0
    _min = np.mean(heights_scale_4) - 2*np.std(heights_scale_4)
    _max = np.mean(heights_scale_4) + 2*np.std(heights_scale_4)
    print("heights 4 min:",_min," max:",_max)
    
    _min = 0
    _max = 0  
    _min = np.mean(heights_scale_8) - 2*np.std(heights_scale_8)
    _max = np.mean(heights_scale_8) + 2*np.std(heights_scale_8)
    print("heights 8 min:",_min," max:",_max)
    
    _min = 0
    _max = 0 
    _min = np.mean(heights_scale_16) - 2*np.std(heights_scale_16)
    _max = np.mean(heights_scale_16) + 2*np.std(heights_scale_16)
    print("heights 16 min:",_min," max:",_max)


if __name__ == "__main__":    
    scale_box_size() 