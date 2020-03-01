import Services.loader as load
from Network.network_creator import NetworkCreator

def StartTrain():
    loader = load.Loader()
    loader.load_data()
    
    nc = NetworkCreator()
    nc.train(loader)
    

if __name__ == '__main__':
    StartTrain()

