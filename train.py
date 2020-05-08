import Services.loader as load
from Network.network_creator import NetworkCreator
import tensorflow as tf
import config as cfg

def StartTrain():
    loader = load.Loader()
    loader.load_specific_label("000008")
    #loader.prepare_data(cfg.BATCH_SIZE)
    # loader.load_data()
    
    nc = NetworkCreator()
    with tf.device(cfg.DEVICE_NAME):
        nc.start_train(loader)
    

if __name__ == '__main__':
    StartTrain()