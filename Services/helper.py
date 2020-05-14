__date__   = '14/05/2020'
__author__ = 'Lukas Mrvecka'
__email__  = 'lukas.mrvecka.st@vsb.cz'

import numpy as np

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def normalize_to_image(data):
    return (data - np.min(data)) * (255/(np.max(data) - np.min(data))) 

def change_first_x_last_dim(data):
    tmp = np.moveaxis(data, 0, -1)
    return tmp