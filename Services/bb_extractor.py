import config as cfg
import numpy as np
from Models.boxModel import BoxModel, ResultBoxModel
import cv2

import Services.geometry as geom

def extract_bounding_box(result, label, calib_matrix):
    
    maps = cv2.split(np.squeeze(result,axis=0))
    image = np.asmatrix(maps[0])
    im_max = image.max()
    
    tresh_value = (im_max * cfg.RESULT_TRESHOLD) / 100 
    image_model = ResultBoxModel()
    for y in range(image.shape[0]):
        
        for x in range(image.shape[1]):
            # pixels
            if image[y,x] >= tresh_value:

                fbl_x = maps[1][y,x]
                fbl_y = maps[2][y,x]
                fbr_x = maps[3][y,x]
                fbr_y = maps[4][y,x]
                rbl_x = maps[5][y,x]
                rbl_y = maps[6][y,x]
                ftl_y = maps[7][y,x]

                data = [[fbl_x, fbr_x, rbl_x],
                        [fbl_y, fbr_y, rbl_y],
                        [1, 1, 1]]
                
                world_space = geom.image_to_world_space(data, calib_matrix, [0,1,0], 0)
                w_fbl = world_space[:,0]
                w_fbr = world_space[:,1]
                w_rbl = world_space[:,2]
                
                w_rbr = w_fbr + (w_rbl - w_fbl)
                
                front_normal = w_fbl - w_rbl
                front_d = np.dot(front_normal, w_fbl)
                w_ftl = geom.image_to_world_space(np.reshape([fbl_x,ftl_y,1],(3,1)), calib_matrix, [0,1,0], front_d)
                bottom_to_top = np.subtract(w_ftl, np.reshape(w_fbl,(3,1)))


                # now we have reconstructed bottom rectangle but it is paralelogram
                
                # center of parallelogram
                mass_center = (w_fbl + w_rbr) / 2.0
                # half diagonals
                d1 = w_fbl - mass_center
                length_d1 = np.linalg.norm(d1)
                
                d2 = w_fbr - mass_center
                length_d2 = np.linalg.norm(d2)
                
                delta = abs(length_d1 - length_d2) / 2.0
                
                d1_new = []
                d2_new = []
                if length_d1 > length_d2:
                    # first diagonal is shorter
                    d1_new = d1 * (1 - delta / length_d1)
                    d2_new = d2 * (1 + delta / length_d2)
                else:
                    d1_new = d1 * (1 + delta / length_d1)
                    d2_new = d2 * (1 - delta / length_d2)
                
                w_fbl = np.reshape(mass_center + d1_new, (3,1))
                w_fbr = np.reshape(mass_center + d2_new, (3,1))
                w_rbl = np.reshape(mass_center - d2_new, (3,1))
                w_rbr = np.reshape(mass_center - d1_new, (3,1))
                
                w_ftl = np.reshape(w_fbl, (3,1)) + bottom_to_top
                w_ftr = np.reshape(w_fbr, (3,1)) + bottom_to_top
                w_rtl = np.reshape(w_rbl, (3,1)) + bottom_to_top
                w_rtr = np.reshape(w_rbr, (3,1)) + bottom_to_top
                
                
                data = np.asarray([w_fbl, w_fbr, w_rbl, w_rbr, w_ftl, w_ftr, w_rtl, w_rtr])
                transposed = np.squeeze(data.transpose())
                
                data = np.ones((4,8))
                data[0:3,0:8] = transposed
                points = geom.world_space_to_image(data, calib_matrix)
                
                box = BoxModel()
                box.confidence = (image[y,x] * 100) / im_max
                box.fbl = (int(points[0,0]), int(points[1,0]))
                box.fbr = (int(points[0,1]), int(points[1,1]))
                box.rbl = (int(points[0,2]), int(points[1,2]))
                box.rbr = (int(points[0,3]), int(points[1,3]))
                box.ftl = (int(points[0,4]), int(points[1,4]))
                box.ftr = (int(points[0,5]), int(points[1,5]))
                box.rtl = (int(points[0,6]), int(points[1,6]))
                box.rtr = (int(points[0,7]), int(points[1,7]))
                
                image_model.boxes.append(box)
                
    return image_model
                
                

                
            
    
    
    