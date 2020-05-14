__date__   = '14/05/2020'
__author__ = 'Lukas Mrvecka'
__email__  = 'lukas.mrvecka.st@vsb.cz'

class ResultBoxModel():
    
    def __init__(self):
        self.boxes = []
        self.file_name = ''

class BoxesWithEthalons():
    
    def __init__(self):
        self.boxes = []        
        self.ethalon = ()


class BoxModel():

    def __init__(self):
        self.confidence= 0
        
        self.fbl = ()
        self.fbr = ()
        self.rbl = ()
        self.rbr = ()
        self.ftl = ()
        self.ftr = ()
        self.rtl = ()
        self.rtr = ()
        # 3x8
        self.world_points = None
        self.image_points = None
        self.object_index = 0
