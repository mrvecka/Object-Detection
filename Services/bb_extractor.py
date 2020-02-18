import config as cfg
import numpy as np
from Models.boxModel import BoxModel, ResultBoxModel
import cv2

import Services.geometry as geom

def extract_bounding_box(result, label, calib_matrix, img_path, scale, ideal):
    
    print("EXTRACTING BOUNDING BOXES STARTED ON SCALE ", scale)
    prob = result[0,:,:,0]
    image = np.asmatrix(prob)
    im_max = image.max()
    thresh_value = (im_max * cfg.RESULT_TRESHOLD) / 100 
    maps = threshold_result(prob, thresh_value)
    
    tmp = np.zeros((9,image.shape[0], image.shape[1]))
    tmp[0] = prob
    maps,object_count = denormalize_to_true_value(result, tmp, scale, ideal)
    print("FOUND: ", object_count, " OBJECTS")
    image_model = ResultBoxModel()
    for y in range(image.shape[0]):
        
        for x in range(image.shape[1]):
            # pixels
            if maps[0][y,x] > 0:

                fbl_x = maps[1][y,x]
                fbl_y = maps[2][y,x]
                fbr_x = maps[3][y,x]
                fbr_y = maps[4][y,x]
                rbl_x = maps[5][y,x]
                rbl_y = maps[6][y,x]
                ftl_y = maps[7][y,x]
                object_index = maps[8][y,x]

                data = np.asarray([[fbl_x, fbr_x, rbl_x],
                [fbl_y, fbr_y, rbl_y],
                [1, 1, 1]])
                # calib_matrix = np.matmul(calib_matrix,rotation_matrix) 
                # world_space = image_to_world_space(image_space_homo[:,0], calib_matrix, [0,1,0], 0)
                w_rbl = geom.image_to_world_space(data[:,2], calib_matrix, [0,1,0], 0)
                w_fbl = geom.image_to_world_space(data[:,0], calib_matrix, [0,1,0], 0)
                w_fbr = geom.image_to_world_space(data[:,1], calib_matrix, [0,1,0], 0)
                w_rbr = w_fbr + (w_rbl - w_fbl)
                front_normal = w_rbl - w_fbl
                front_normal = np.reshape(front_normal, (1,3))
                front_d = -np.dot(front_normal, w_fbl)
                w_ftl = geom.image_to_world_space(np.reshape([data[0,0],ftl_y,1],(3,1)), calib_matrix, front_normal, front_d[0,0])
                bottom_to_top = w_ftl - np.reshape(w_fbl,(3,1))

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
                box.fbl = (int(points[0,0]), int(points[1,0]))
                box.fbr = (int(points[0,1]), int(points[1,1]))
                box.rbl = (int(points[0,2]), int(points[1,2]))
                box.rbr = (int(points[0,3]), int(points[1,3]))
                box.ftl = (int(points[0,4]), int(points[1,4]))
                box.ftr = (int(points[0,5]), int(points[1,5]))
                box.rtl = (int(points[0,6]), int(points[1,6]))
                box.rtr = (int(points[0,7]), int(points[1,7]))
                box.confidence = maps[0][y,x]
                box.object_index = object_index
                
                image_model.boxes.append(box)
                
    print("EXTRACTION FINISHED!")
    return image_model
                
def threshold_result(maps, threshold):
    
    for y in range(len(maps)):
        for x in range(len(maps[0])):
            if maps[y,x] < threshold:
                maps[y,x] = 0
                maps[y,x] = 0
                maps[y,x] = 0
                maps[y,x] = 0
                maps[y,x] = 0
                maps[y,x] = 0
                maps[y,x] = 0
                maps[y,x] = 0
                
    return maps

def denormalize_to_true_value(result, maps, scale, ideal):

    object_count = 0
    y_max = 0
    x_max = 0
    for y in range(len(maps[0])):
        for x in range(len(maps[0,y])):
            if maps[0][y,x] > 0:
                y_max, x_max = find_local_max_coordinates(maps[0], y,x, scale)
                if y_max != 0 and x_max != 0:
                    object_count += 1
                    
                # maps[c][yp][xp] = 0.5 + (label[c-1] - x - j * scale) / ideal
                maps[1][y,x] = (result[0,y,x,1] - 0.5) * ideal + x_max + x / scale
                maps[3][y,x] = (result[0,y,x,3] - 0.5) * ideal + x_max + x / scale
                maps[5][y,x] = (result[0,y,x,5] - 0.5) * ideal + x_max + x / scale
                
                # maps[c][yp][xp] = 0.5 + (label[c-1] - y - i * scale) / ideal                
                maps[2][y,x] = (result[0,y,x,2] - 0.5) * ideal + y_max + y / scale
                maps[4][y,x] = (result[0,y,x,4] - 0.5) * ideal + y_max + y / scale
                maps[6][y,x] = (result[0,y,x,6] - 0.5) * ideal + y_max + y / scale
                maps[7][y,x] = (result[0,y,x,7] - 0.5) * ideal + y_max + y / scale
                maps[8][y,x] = object_count
                # (M - 0.5) - I + x + x_max*S
                    
    return maps, object_count
                
def find_local_max_coordinates(prob_map, y, x, scale):
    
    height = len(prob_map)
    width = len(prob_map[0])
    y_max = y
    x_max = x
    max_val = prob_map[y,x]

    size = 8
    if scale == 2:
        size = 16
    if scale == 4:
        size = 8
    if scale == 8:
        size = 4
    if scale == 16:
        size = 3


    # left up
    for _y in range(y-size,y + size,1):
        for _x in range(x -size,x + size ,1):
            if _x > 0 and _y > 0 and _x < width and _y < height:
                if prob_map[_y,_x] > max_val:
                    y_max = _y
                    x_max = _x
                    max_val = prob_map[_y,_x]
                    
                
    return y_max, x_max

def get_info_from_pgp(file_path):
    slash_index = file_path.rindex('\\')
    dot_index = file_path.rindex('.')
    name = file_path[slash_index:dot_index]
    with open(cfg.PGP_FOLDER + r'\pgps_info.txt', 'r') as infile_label:
        for line in infile_label:
            line = line.rstrip(r'\n')
            data = line.split(' ')
            if data == name:
                P = np.asmatrix([[float(data[1]), float(data[2]),  float(data[3]),  float(data[4])],
                    [float(data[5]), float(data[6]),  float(data[7]),  float(data[8])],
                    [float(data[9]), float(data[10]), float(data[11]), float(data[12])]])
                
                gp = np.asmatrix([float(data[13]),float(data[14]),float(data[15]),float(data[16])])
                
                return p, gp
                
    return None, None

def showResults(result, img_path, scale, ideal):
    
    maps = result[0,:,:,0]
    image = np.asmatrix(maps)
    im_max = image.max()
    thresh_value = (im_max * cfg.RESULT_TRESHOLD) / 100 
    maps = threshold_result(maps, thresh_value)
    
    tmp = np.zeros((8,image.shape[0], image.shape[1]))
    tmp[0] = maps
    tmp = denormalize_to_true_value(result, tmp, scale, ideal)
    
    
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            
            if tmp[0][y,x] > 0:
                # fbl_x = 766.9976689964997 
                # fbl_y = 292.1749884809996
                # fbr_x = 897.5057514572361 
                # fbr_y = 292.8354111658222 
                # rbl_x = 866.5701895091134 
                # rbl_y = 380.04597547426044 
                # ftl_y = 172.0861783040727
                          
                fbl_x = int(tmp[1][y,x])
                fbl_y = int(tmp[2][y,x])
                fbr_x = int(tmp[3][y,x])
                fbr_y = int(tmp[4][y,x])
                rbl_x = int(tmp[5][y,x])
                rbl_y = int(tmp[6][y,x])
                ftl_y = int(tmp[7][y,x])
                
                # front
                cv2.line(img, (fbl_x,fbl_y), (fbr_x,fbr_y), (0,0,255), 2) 
                cv2.line(img, (fbl_x,fbl_y), (rbl_x,rbl_y), (0,0,255), 2) 
                cv2.line(img, (fbl_x,fbl_y), (fbl_x,ftl_y), (0,0,255), 2)
                
    cv2.imshow("result without NMS", img)
    cv2.waitKey(0)
            
    
    
    
    
                