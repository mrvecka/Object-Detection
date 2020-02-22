
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
        self.confidence: 0
        
        self.fbl = ()
        self.fbr = ()
        self.rbl = ()
        self.rbr = ()
        self.ftl = ()
        self.ftr = ()
        self.rtl = ()
        self.rtr = ()
        
        self.object_index = 0
