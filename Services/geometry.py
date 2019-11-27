import numpy as np
import loader as load

def rotation_4x4(rotation,x,y,z):
    
    rt = np.asmatrix([[np.cos(rotation), 0.0, np.sin(rotation), x],
                   [0.0, 1.0, 0.0, y],
                   [-np.sin(rotation), 0.0, np.cos(rotation), z],
                   [0, 0, 0, 1]])
 
 
    return rt

def get_points_matrix(P,R,label) -> [[]]:

    #                       rbl                     fbl                 fbr                 ftl
    corners = np.asmatrix([ [-label.dim_length/2,   label.dim_length/2, label.dim_length/2, label.dim_length/2],
                            [0,                     0,                  0,                  -label.dim_height],
                            [label.dim_width/2,     label.dim_width/2,  -label.dim_width/2, label.dim_width/2],
                            [1,                     1,                  1,                  1]])

    p_3x4 = P * R        
    corners = p_3x4 * corners

    corners = corners / corners[2]

    return corners

def image_to_world_space(data, path, normal, d):
    loader = load.Loader()
    matrix = loader.load_calibration(path)
    
    P_3 = matrix[0:2,0:2]
    P_1 = matrix[:,3]
    
    inverse_P_3 = np.linalg.inv(P_3) 
    
    eye = (-1 * inverse_P_3) * P_1
    
    normal_eye = normal * eye
    inverse_P_3_y = inverse_P_3 * data
    
    normal_inverse_P_3_y = normal * inverse_P_3_y
    _lambda = -(normal_eye + d) / normal_inverse_P_3_y
    
    x = eye + inverse_P_3_y * _lambda
    
    return x

def world_space_to_image(data, path):
    loader = load.Loader()
    matrix = loader.load_calibration(path)
    
    points = matrix * data
    points = points / points[2]
    
    return points[0:1,:]

def get_points_distance(p1, p2):  
    squere_sum = pow(p1[0] - p2[0],2) + pow(p1[1] - p2[1],2) + pow(p1[2] - p2[2],2)
    result = np.sqrt(squere_sum)
    return result
