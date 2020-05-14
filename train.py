__date__   = '14/05/2020'
__author__ = 'Lukas Mrvecka'
__email__  = 'lukas.mrvecka.st@vsb.cz'

import Services.loader as load
from Network.network_creator import NetworkCreator
import tensorflow as tf
import config as cfg

import tensorflow as tf
from tensorflow.python.client import device_lib


def StartTrain():
    loader = load.Loader()
    batch = cfg.BATCH_SIZE
    if cfg.SPECIFIC_DATA != "":
        loader.load_specific_label(cfg.SPECIFIC_DATA)
        batch = 1
    else:
        loader.load_data()
           
    nc = NetworkCreator(batch)
    
    if cfg.USE_GPU:
        device_name = find_gpu(cfg.DEVICE_NAME)
        if device_name != None:
            with tf.device(device_name):
                nc.start_train(loader)
        else:
            nc.start_train(loader)
    else:
        nc.start_train(loader)
 

def find_gpu(user_device):
    local_device_protos = device_lib.list_local_devices()

    print("Available GPU's")
    device_name = ""
    
    for device in local_device_protos:
        if (device.device_type == "GPU"):
            print("Name:", device.name)
            if (device_name == "" or user_device == device.name):
                device_name = device.name

    if device_name != "":
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in gpu_devices:
                tf.config.experimental.set_memory_growth(device, True)
                
        print("Using device: ", device_name)
        return device_name
    
    print("Using device: CPU")
    return None

if __name__ == '__main__':
    StartTrain()