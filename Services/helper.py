import numpy as np

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def change_first_x_last_dim(data):
    tmp = np.moveaxis(data, 0, -1)
    return tmp