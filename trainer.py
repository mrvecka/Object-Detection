import loader as load
from ground_plane_extractor import GroundPlaneExtractor
from Network.network_creator import NetworkCreator



def StartTrain():
    loader = load.Loader()
    loader.load_data()
    # # data was loaded to loader.Data property
    # print(loader.Data)

    #extract ground plane
    # print('extract ground plane')
    # gp_extract = GroundPlaneExtractor()
    # gp_extract.start_ransac()
    # print(gp_extract.result_gp)
    
    
    nc = NetworkCreator()
    nc.start_training(loader)
    
    

if __name__ == '__main__':
    StartTrain()

