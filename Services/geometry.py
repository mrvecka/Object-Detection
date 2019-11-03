import numpy as np

def rotation_4x4(rotation,x,y,z):
    
    rt = np.asmatrix([[np.cos(rotation), 0.0, np.sin(rotation), x],
                   [0.0, 1.0, 0.0, y],
                   [-np.sin(rotation), 0.0, np.cos(rotation), z],
                   [0, 0, 0, 1]])
 
 
    return rt