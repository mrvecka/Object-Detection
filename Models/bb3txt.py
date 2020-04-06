import numpy as np
import config as cfg
import Services.fileworker as fwb
import Services.geometry as geom

class BB3Txt():

    def __init__(self):
        self.file_name: ''
        self.label: ''
        self.confidence: 0
        self.fbl_x: 0
        self.fbl_y: 0
        self.fbr_x: 0
        self.fbr_y: 0
        self.rbl_x: 0
        self.rbl_y: 0
        self.ftl_y: 0
        
        self.bb_center_x = 0
        self.bb_center_y = 0
        
        self.largest_dim = 0

    def to_string(self):
        data = self.file_name + ' ' + self.label + ' ' + str(self.fbl_x) + ' ' + str(self.fbl_y) + ' ' + str(self.fbr_x) + ' ' + str(self.fbr_y) + ' ' + str(self.rbl_x) + ' ' + str(self.rbl_y) + ' ' + str(self.ftl_y) + ' ' + str(self.bb_center_x) + ' ' + str(self.bb_center_y) + ' ' + str(self.largest_dim)
        return data

def create_empty_object(file_name):
    bb3 = BB3Txt()
    bb3.file_name = file_name
    bb3.label = "-1"
    bb3.confidence = -1
    
    bb3.rbl_x = -1
    bb3.rbl_y = -1
    bb3.fbl_x = -1
    bb3.fbl_y = -1
    bb3.fbr_x = -1
    bb3.fbr_y = -1
    bb3.ftl_y = -1
    bb3.bb_center_x = -1
    bb3.bb_center_y = -1
    bb3.bb_center_x = -1
    bb3.bb_center_y = -1    
    bb3.largest_dim = -1
    
    return bb3

def create_bb3txt_object(label, file_name, P, width, height) -> BB3Txt:
    bb3 = BB3Txt()
    bb3.file_name = file_name
    bb3.label = label.label
    bb3.confidence = 0
    # rbl = np.asmatrix([[-label.dim_length/2],[0],[label.dim_width/2],[1]])
    # fbl = np.asmatrix([[label.dim_length/2],[0],[label.dim_width/2],[1]])
    # fbr = np.asmatrix([[label.dim_length/2],[0],[-label.dim_width/2],[1]])

    # ftl = np.asmatrix([[label.dim_length/2],[-label.dim_height],[label.dim_width/2],[1]])

    rotation_4x4 = geom.rotation_4x4(label.rotation,label.location_x,label.location_y,label.location_z)


    # fbl_3x1 = p_3x4 * fbl
    # fbr_3x1 = p_3x4 * fbr
    # rbl_3x1 = p_3x4 * rbl
    # ftl_3x1 = p_3x4 * ftl

    # rbl_3x1 = rbl_3x1 / rbl_3x1[2,0]
    # fbl_3x1 = fbl_3x1 / fbl_3x1[2,0]
    # fbr_3x1 = fbr_3x1 / fbr_3x1[2,0]
    # ftl_3x1 = ftl_3x1 / ftl_3x1[2,0]

    corners = geom.get_points_matrix(P,rotation_4x4,label)

    bb3.rbl_x = corners[0,0]
    bb3.rbl_y = corners[1,0]

    bb3.fbl_x = corners[0,1]
    bb3.fbl_y = corners[1,1]

    bb3.fbr_x = corners[0,2]
    bb3.fbr_y = corners[1,2]

    bb3.ftl_y = corners[1,3]
    
    bb3.bb_center_x = label.x_top_left + (label.x_bottom_right - label.x_top_left) / 2
    bb3.bb_center_y = label.y_top_left + (label.y_bottom_right - label.y_top_left) / 2
    
    # scale center of object to CNN input image size
    scale_width_factor = (cfg.IMG_WIDTH * 100) / width
    bb3.bb_center_x = bb3.bb_center_x * (scale_width_factor / 100)
    scale_height_factor = (cfg.IMG_HEIGHT * 100) / height
    bb3.bb_center_y = bb3.bb_center_y * (scale_height_factor / 100)
    
    bb3.largest_dim = np.max([label.x_bottom_right - label.x_top_left, label.y_bottom_right - label.y_top_left])
    
    return bb3

def write_bb3_to_file(bb3_labels) -> None:

    bb3_path = cfg.BB3_FOLDER
    if not fwb.check_dir_exists(bb3_path):
        if not fwb.create_dir(bb3_path):
            return
    file_path =bb3_path + r'\\' + bb3_labels[0].file_name + '.txt'
    if fwb.check_file_exists(file_path):
        fwb.delete_file(file_path)
        
    f= open(file_path,"w+")

    for label in bb3_labels:
        data = label.to_string()
        f.write(data + '\n')

    f.close()
    