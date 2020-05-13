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
    nc.train(loader)

if __name__ == '__main__':
    StartTrain()