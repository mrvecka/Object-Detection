import numpy as np

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))