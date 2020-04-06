import Services.loader as load
from Network.network_creator import NetworkCreator
import tensorflow as tf
def StartTrain():
    loader = load.Loader()
    loader.load_specific_label("000025")
    # loader.load_data()
    
    nc = NetworkCreator()
    with tf.device('/gpu:0'):
        nc.start_train(loader)
    

if __name__ == '__main__':
    StartTrain()