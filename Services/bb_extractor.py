import config as cfg
import numpy as np
from Models.boxModel import BoxModel, ResultBoxModel

import geometry as geom

def extract_bounding_box(maps, image_path, calib_path):
    image = np.asmatrix(maps[0])
    im_max = image.max
    
    tresh_value = (im_max * cfg.RESULT_TRESHOLD) / 100 
    image_model = ResultBoxModel()
    image_model.file_name = image_path
    for y in range(len(image)):
        
        for x in range(len(image[y])):
            # pixels
            if image[y][x] >= tresh_value:

                fbl_x = x + maps[1][y][x]
                fbl_y = y + maps[2][y][x]
                fbr_x = x + maps[3][y][x]
                fbr_y = y + maps[4][y][x]
                rbl_x = x + maps[5][y][x]
                rbl_y = y + maps[6][y][x]
                ftl_y = y + maps[7][y][x]

                data = [[fbl_x, fbr_x, rbl_x],
                        [fbl_y, fbr_y, rbl_y],
                        [1, 1, 1]]
                
                world_space = geom.image_to_world_space(data, calib_path, [0,1,0] , 0)
                w_fbl = world_space[:,0]
                w_fbr = world_space[:,1]
                w_rbl = world_space[:,2]
                
                w_rbr = w_fbr + (w_rbl - w_fbl)
                
                front_normal = w_fbl - w_rbl
                front_d = np.dot(front_normal, w_fbl)
                w_ftl = geom.image_to_world_space([[fbl_x],[ftl_y],[1]], calib_path, [0,1,0], front_d)
                bottom_to_top = w_ftl - w_fbl


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
                
                w_fbl = mass_center + d1_new
                w_fbr = mass_center + d2_new
                w_rbl = mass_center - d2_new
                w_rbr = mass_center - d1_new
                
                w_ftl = w_fbl + bottom_to_top
                w_ftr = w_fbr + bottom_to_top
                w_rtl = w_rbl + bottom_to_top
                w_rtr = w_rbr + bottom_to_top
                
                
                data = np.asarray([w_fbl, w_fbr, w_rbl, w_rbr, w_ftl, w_ftr, w_rtl, w_rtr])
                transposed = data.transpose()
                
                points = geom.world_space_to_image(transposed, calib_path)
                
                box = BoxModel()
                box.confidence = (image[y][x] * 100) / im_max
                box.fbl = (points[:,0][0,0], points[:,0][1,0])
                box.fbr = (points[:,1][0,0], points[:,1][1,0])
                box.rbl = (points[:,2][0,0], points[:,2][1,0])
                box.rbr = (points[:,3][0,0], points[:,3][1,0])
                box.ftl = (points[:,4][0,0], points[:,4][1,0])
                box.ftr = (points[:,5][0,0], points[:,5][1,0])
                box.rtl = (points[:,6][0,0], points[:,6][1,0])
                box.rtr = (points[:,7][0,0], points[:,7][1,0])
                
                image_model.boxes.append(box)
                
                

                
            
    
    
    