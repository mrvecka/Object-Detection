
class LabelModel:
    def __init__(self):
        self.label =  ''
        self.truncated = 0 # 0/1
        self.occluded = 0 # 0/1/2/3
        self.alpha = 0.0
        self.x_top_left = 0.0 #pixel
        self.y_top_left = 0.0
        self.x_bottom_right = 0.0
        self.y_bottom_right = 0.0
        self.dim_width = 0.0 # meter
        self.dim_height = 0.0
        self.dim_length = 0.0
        self.location_x = 0.0
        self.location_y = 0.0
        self.location_z = 0.0
        self.rotation = 0.0 # rad
        