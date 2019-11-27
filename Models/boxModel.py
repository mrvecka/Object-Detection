
class ResultBoxModel():
    def __init__(self):
        self.file_name: ''
        self.boxes = []


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
        
        # self.fbl_x = 0
        # self.fbl_y = 0
        
        # self.fbr_x = 0
        # self.fbr_y = 0
        
        # self.ftl_x = 0
        # self.ftl_y = 0
        
        # self.ftr_x = 0
        # self.ftr_y = 0
        
        # self.rbl_x = 0
        # self.rbl_y = 0
        
        # self.rbr_x = 0
        # self.rbr_y = 0
        
        # self.rtl_x = 0
        # self.rtl_y = 0
        
        # self.rtr_x = 0
        # self.rtr_y = 0