import numpy as np
import config as cfg
import Services.fileworker as fwb
import Services.geometry as geom

class BB3Txt():

    def __init__(self):
        self.file_name: ''
        self.label: ''
        self.coinfidence: 0
        self.fbl_x: 0
        self.fbl_y: 0
        self.fbr_x: 0
        self.fbr_y: 0
        self.rbl_x: 0
        self.rbl_y: 0
        self.ftl_y: 0

    def to_string(self):
        data = self.file_name + ' ' + self.label + ' ' + str(self.coinfidence) + ' ' + str(self.fbl_x) + ' ' + str(self.fbl_y) + ' ' + str(self.fbr_x) + ' ' + str(self.fbr_y) + ' ' + str(self.rbl_x) + ' ' + str(self.rbl_y) + ' ' + str(self.ftl_y)
        return data

def create_bb3txt_object(label, file_name, P) -> BB3Txt:
    bb3 = BB3Txt()
    bb3.file_name = file_name
    bb3.label = label.label
    bb3.coinfidence = 0
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

    corners = get_points_matrix(P,rotation_4x4,label)

    bb3.fbl_x = corners[0,0]
    bb3.fbl_y = corners[1,0]

    bb3.fbr_x = corners[0,1]
    bb3.fbr_y = corners[1,1]

    bb3.rbl_x = corners[0,2]
    bb3.rbl_y = corners[1,2]

    bb3.ftl_y = corners[1,3]

    return bb3

def get_points_matrix(P,R,label) -> [[]]:

    #                       rbl                     fbl                 fbr                 ftl
    corners = np.asmatrix([ [-label.dim_length/2,   label.dim_length/2, label.dim_length/2, label.dim_length/2],
                            [0,                     0,                  0,                  -label.dim_height],
                            [label.dim_width/2,     label.dim_width/2,  -label.dim_width/2, label.dim_width/2],
                            [1,                     1,                  1,                  1]])

    p_3x4 = P * R        
    corners = p_3x4 * corners

    corners = corners / corners[2]

    return corners

def write_bb3_to_file(bb3_labels) -> None:

    bb3_path =cfg.BASE_PATH + r'\\'+ cfg.BB3_FOLDER
    if not fwb.check_dir_exists(bb3_path):
        if not fwb.create_dir(bb3_path):
            return
    file_path =bb3_path + r'\\' + bb3_labels[0].file_name
    if fwb.check_file_exists(file_path):
        fwb.delete_file(file_path)
        
    f= open(file_path,"w+")

    for label in bb3_labels:
        data = label.to_string()
        f.write(data + '\n')

    f.close()
    