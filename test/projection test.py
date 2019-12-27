import Services.loader as load
import numpy as np
from Models.boxModel import BoxModel, ResultBoxModel
import Services.drawer as drawer

def get_rotation_matrix(rotation):
    matrix = [[ np.cos(rotation),   0, np.sin(rotation),    2.87],
              [ 0,                1.0, 0,                   1.61],
              [-np.sin(rotation),   0, np.cos(rotation),    7.64],
              [0,                   0, 0,                   1]]
    
    # matrix = [[ np.cos(rotation),   0, np.sin(rotation),    -3.59],
    #         [ 0,                1.0, 0,                   1.69],
    #         [-np.sin(rotation),   0, np.cos(rotation),    12.01],
    #         [0,                   0, 0,                   1]]

    return np.asmatrix(matrix)

def image_to_world_space(data, calib_matrix, normal, d):
    
    data = np.reshape(data, (3,1))
    normal = np.reshape(normal,(1,3))
    P_3 = np.reshape(calib_matrix[0:3,0:3],(3,3))
    P_1 = np.reshape(calib_matrix[:,3],(3,1))
    
    inverse_P_3 = np.linalg.inv(P_3) 
    
    eye = np.matmul((-1 * inverse_P_3),P_1)
    # eye should be 3x1
    
    normal_eye = np.matmul(normal, eye)
    inverse_P_3_y = np.matmul(inverse_P_3, data)
    
    normal_inverse_P_3_y = np.matmul(normal, inverse_P_3_y)
    _lambda = -(normal_eye[0,0] + d) / normal_inverse_P_3_y[0,0]
    
    x = eye + _lambda * inverse_P_3_y
    
    return x

def world_space_to_image(data, calib_matrix):
    
    points = np.matmul(calib_matrix, data)
    points = points / points[2,:]
    
    return points[0:2,:]

def test_projection(label, calib_matrix):
    
    
    
    image_model = ResultBoxModel()
    
    rotation_matrix = get_rotation_matrix(-1.54)
    calib_matrix = np.matmul(calib_matrix, rotation_matrix)
    
        
    fbl_x = label[0]
    fbl_y = label[1]
    fbr_x = label[2]
    fbr_y = label[3]
    rbl_x = label[4]
    rbl_y = label[5]
    ftl_y = label[6]
    data = np.asarray([[fbl_x, fbr_x, rbl_x],
            [fbl_y, fbr_y, rbl_y],
            [1, 1, 1]])
    # calib_matrix = np.matmul(calib_matrix,rotation_matrix) 
    # world_space = image_to_world_space(image_space_homo[:,0], calib_matrix, [0,1,0], 0)
    w_rbl = image_to_world_space(data[:,2], calib_matrix, [0,1,0], 0)
    w_fbl = image_to_world_space(data[:,0], calib_matrix, [0,1,0], 0)
    w_fbr = image_to_world_space(data[:,1], calib_matrix, [0,1,0], 0)
    w_rbr = w_fbr + (w_rbl - w_fbl)
    front_normal = w_rbl - w_fbl
    front_normal = np.reshape(front_normal, (1,3))
    front_d = -np.dot(front_normal, w_fbl)
    w_ftl = image_to_world_space(np.reshape([data[0,0],ftl_y,1],(3,1)), calib_matrix, front_normal, front_d[0,0])
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
    points = world_space_to_image(data, calib_matrix)

    box = BoxModel()
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



if __name__ == "__main__":
    loader = load.Loader()
    # loader.load_data()
    loader.load_specific_label("000046")
    image_batch, label_batch, image_paths, calib_matrices = loader.get_test_data(1)
    b_boxes_model = test_projection(label_batch[0][1],calib_matrices)
    b_boxes_model.file_name = image_paths[0]
    drawer.draw_bounding_boxes(b_boxes_model)