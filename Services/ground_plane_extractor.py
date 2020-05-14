__date__   = '14/05/2020'
__author__ = 'Lukas Mrvecka'
__email__  = 'lukas.mrvecka.st@vsb.cz'

import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import Services.fileworker as fw
import Services.geometry as geom

import random
from Models.labelModel import LabelModel
import numpy as np
import config as cfg

class GroundPlaneExtractor():

    def __init__(self):

        self.iterations = cfg.RANSAC_ITERATIONS
        self.gp_4_xn = [[]]
        self.label_path = cfg.LABEL_PATH            
        self.data_amount = 10000
        self.ground_points = []
        
        self.result_gp = []

    def start_ransac(self):
        
        self._load_label_data()
        
        
        self.gp_4_xn = np.asmatrix(np.ones((4, len(self.ground_points))))
        for i in range(len(self.ground_points)):
            self.gp_4_xn[0:3,i] = self.ground_points[i]
        
        self.extract_ground_plane()
        

    def extract_ground_plane(self):
        """
        compute ground plane koeficients
        Input:
            vertices: array of all loaded vertices on ground plane
            N: number of iterations trought vertices array
        """

        dst_min = np.inf
        gp_max = np.asmatrix(np.zeros((1,4)))

        for _ in range(self.iterations):
            #get random 3 vertices
            indexes = random.sample(range(0, len(self.ground_points)), 3)


            vert_0_array = self.ground_points[indexes[0]]
            vert_1_array = self.ground_points[indexes[1]]
            vert_2_array = self.ground_points[indexes[2]]
            gp = self._compute_koeficients(vert_0_array, vert_1_array, vert_2_array)
            if gp * self.gp_4_xn[:,indexes[0]] > 0.000000001 or \
                gp * self.gp_4_xn[:,indexes[1]] > 0.000000001 or \
                gp * self.gp_4_xn[:,indexes[2]] > 0.000000001:
                print('WARNING: Solution is not precise, skipping...')
                continue

            # Compute the sum of distances from this plane
            distances2 = np.power(gp * self.gp_4_xn, 2)
            dist2_sum = np.sum(distances2, axis=1)
            
            if dist2_sum[0,0] < dst_min:
                print('New min distance sum:', str(dist2_sum[0,0]))
                dst_min = dist2_sum[0,0]
                gp_max = gp

        self.result_gp = gp_max
    
    def _compute_koeficients(self,p1,p2,p3):

        l_1 = p2 - p1
        l_2 = p3 - p1
        
        normal = np.cross(l_1,l_2,axis=0)
        d = - (normal[0,0]*p1[0,0] + normal[1,0]*p1[1,0] + normal[2,0]*p1[2,0])
    
        return np.asmatrix([normal[0,0], normal[1,0], normal[2,0], d])
    
    def _load_label_data(self):        
        """
        Reads a label file to specific image.
        Read only Car labels

        Input:
            label_path: Row-major stored label separated by spaces, first element is the label name
            x: number of image
        Returns:
            LabelModel object
        """
        

        x = 0
        labels = []
        while x < self.data_amount:
            # print('loading file',x)
            local_label_path = self.label_path + r'\\' + str(x).zfill(6)+'.txt'
            if not fw.check_file_exists(local_label_path):
                return None

            with open(local_label_path, 'r') as infile_label:

                for line in infile_label:
                    line = line.rstrip('\n')
                    data = line.split(' ')

                    # Process just cars
                    if data[0] != 'Car': continue

                    # process just object that are good visible
                    if (int(data[2]) >= 2 or float(data[1]) > 0.75): continue

                    label = LabelModel()
                    label.dim_width = float(data[8])
                    label.dim_height = float(data[9])
                    label.dim_length = float(data[10])
                    label.location_x = float(data[11])
                    label.location_y = float(data[12])
                    label.location_z = float(data[13])
                    label.rotation = float(data[14])

                    self.ground_points.extend(self._get_bottom_points(label))
            
            x+=1

        #return labels
    
    def _get_bottom_points(self, label):
        #                       rbl                     fbl                 fbr                 rbr
        corners = np.asmatrix([ [-label.dim_length/2,   label.dim_length/2, label.dim_length/2, -label.dim_length/2 ],
                                [0,                     0,                  0,                  0                   ],
                                [label.dim_width/2,     label.dim_width/2,  -label.dim_width/2, -label.dim_width/2  ],
                                [1,                     1,                  1,                  1                   ]])

        R = geom.rotation_4x4(label.rotation,label.location_x,label.location_y,label.location_z)
        corners = R * corners
        corners_list = []
        corners_list.append(corners[0:3,0])
        corners_list.append(corners[0:3,1])
        corners_list.append(corners[0:3,2])
        corners_list.append(corners[0:3,3])

        return corners_list



if __name__ == '__main__':
    gpe = GroundPlaneExtractor()
    gpe.start_ransac()
    print(gpe.result_gp)
