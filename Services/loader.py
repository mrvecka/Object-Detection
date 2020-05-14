__date__   = '14/05/2020'
__author__ = 'Lukas Mrvecka'
__email__  = 'lukas.mrvecka.st@vsb.cz'

import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import cv2
import numpy as np
import random
import Services.fileworker as fw
import Services.helper as h
from Models.labelModel import LabelModel
from Models.dataModel import DataModel
import Models.bb3txt as bb
import config as cfg

class Loader:
    def __init__(self):
        self.Data = []
        self.image_path = ''
        self.label_path = ''
        self.calib_path = ''
        self.amount = 0
        self.start_from = 0
        self.image_extension = ''
        self.colored = False

        self.init()

    def init(self):
        
        if fw.check_dir_exists(cfg.IMAGE_PATH):
            self.image_path = cfg.IMAGE_PATH
        else:
            print("Image path '"+ cfg.IMAGE_PATH +"' not found!!!")
            
        if fw.check_dir_exists(cfg.LABEL_PATH):
            self.label_path = cfg.LABEL_PATH
        else:
            print("Label path '"+ cfg.LABEL_PATH +"' not found!!!")  
            
        if fw.check_dir_exists(cfg.CALIB_PATH):
            self.calib_path = cfg.CALIB_PATH
        else:
            print("Calibration path '"+ cfg.CALIB_PATH +"' not found!!!")
            
        self.amount = cfg.DATA_AMOUNT
        self.image_extension = cfg.IMAGE_EXTENSION
        if cfg.IMG_CHANNELS == 1:
            self.colored = False
        else:
            self.colored = True           
          
    def clear(self):
        self.Data = []
          
    def load_data(self):
        """
        Load data from files on specified path.
        Image file name is formated to be "000000" with specified extension
        
        Returns:
            Data are stored to properties
        """
        assert self.image_path != '', 'Image path not set. Nothing to work with. Check config file.'
        assert self.label_path != '', 'Label path not set. Nothing to work with. Check config file.'
        assert self.calib_path != '', 'Calibration path not set. Nothing to work with. Check config file.'
        print('Loading training files')
        
        image_pathss = []
        label_paths = []
        calib_paths = []
        
        # in img_files are absolut paths
        img_files = fw.get_all_files(self.image_path, self.image_extension)
        if self.amount == -1:
            amount_to_load = len(img_files)
        else:
            amount_to_load = self.amount
        
        for i in range(len(img_files)):
            file_ = img_files[i]
            dot_index = file_.find('.')
            file_name = file_[:dot_index]
            
            image_path = self.image_path + '\\' + file_
            label_path = self.label_path + '\\' + file_name + '.txt'
            if not fw.check_file_exists(label_path):
                continue
                    
            calib_path = self.calib_path + '\\' + file_name + '.txt'
            if not fw.check_file_exists(calib_path):
                continue
            
            image, width, height = self._load_image(image_path)
            if image is None:
                continue
                
            # calibration
            calib_matrix = self.load_calibration(calib_path)
            if calib_matrix is None:
                continue
            
            # label
            labels = self._load_label(label_path, file_name, calib_matrix, width, height)
            if labels is None or len(labels) == 0:
                continue
          
            data = DataModel()
            data.image = image
            data.image_path = image_path
            data.image_name = file_
            data.labels = labels
            data.calib_matrix = calib_matrix

            self.Data.append(data)
            if len(self.Data) == amount_to_load:
                break
            
            printProgressBar(len(self.Data), amount_to_load, prefix = 'Progress:', suffix = 'Complete', length = 50)
            
        # self.create_pgp_file()
        printProgressBar(amount_to_load, amount_to_load, prefix = 'Progress:', suffix = 'Complete', length = 50)
        print("Loaded: ",len(self.Data)," training files")
        

    def _load_image(self, image_path):
        """
        Reads a image with number x and based on parameter colored result will have 1 or 3 chanels

        Input:
            image_path: Path to folder with images
            x: number of image
        Returns:
            image as matrix
        """
        if self.colored:
            im = cv2.imread(image_path, cv2.IMREAD_COLOR)
        else:
            im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if im.any() == None:
            return None

        resized = cv2.resize(im, (cfg.IMG_WIDTH,cfg.IMG_HEIGHT), interpolation=cv2.INTER_AREA) # opencv resize function takes as desired shape (width,height) !!!
        #normalized = h.normalize(resized)
        return resized, im.shape[1], im.shape[0]

    def _load_label(self, label_path, file_name, calib_matrix, width, height):
        """
        Reads a label file to specific image.
        Read only Car labels

        Input:
            label_path: Row-major stored label separated by spaces, first element is the label name
            x: number of image
        Returns:
            LabelModel object
        """

        # check if bb3_files folder exists
        # if exists then load from this file they are pre processed to 33btxt format and ready to use
        # if not load kitti label and create bb3txt file, next time this file will be used
        bb3_path = cfg.BB3_FOLDER
        if fw.check_dir_exists(bb3_path):
            bb3_file_path = bb3_path + '\\'+file_name+'.txt'
            if fw.check_file_exists(bb3_file_path):
                result = self._load_from_bb3_folder(bb3_file_path)
                return result
            else:
                # if not exists try add 
                return self.load_one_label(label_path, file_name, calib_matrix, width, height)
        else:
            # if not exists try add 
            return self.load_one_label(label_path, file_name, calib_matrix, width, height)
                
    def load_calibration(self, calib_path):
        """
        Reads a camera matrix P (3x4) stored in the row-major scheme.

        Input:
            calib_path: Row-major stored matrix separated by spaces, first element is the matrix name
            x: number of image
        Returns:
            camera matrix P 4x4
        """

        with open(calib_path, 'r') as infile_calib:
            for line in infile_calib:
                if line[:2] == 'P2':
                    line_data = line.rstrip('\n')
                    data = line_data.split(' ')

                    if data[0] != 'P2:':
                        print('ERROR: We need left camera matrix (P2)!')
                        exit(1)

                    P = np.asmatrix([[float(data[1]), float(data[2]),  float(data[3]),  float(data[4])],
                                    [float(data[5]), float(data[6]),  float(data[7]),  float(data[8])],
                                    [float(data[9]), float(data[10]), float(data[11]), float(data[12])]])
                    return P

        return None

    def _load_from_bb3_folder(self, path):
        labels = []
        with open(path, 'r') as infile_label:

            for line in infile_label:
                line = line.rstrip(r'\n')
                data = line.split(' ')

                label = bb.BB3Txt()
                label.file_name = data[0]
                label.label = data[1]
                label.fbl_x = float(data[2])
                label.fbl_y = float(data[3])
                label.fbr_x = float(data[4])
                label.fbr_y = float(data[5])
                label.rbl_x = float(data[6])
                label.rbl_y = float(data[7])
                label.ftl_y = float(data[8])
                label.bb_center_x = float(data[9])
                label.bb_center_y = float(data[10])
                
                label.largest_dim = float(data[11])
                
                labels.append(label)

        return labels
     
    def load_one_label(self, label_path, file_name, calib_matrix, width, height):

        labels = []
        with open(label_path, 'r') as infile_label:

            for line in infile_label:
                line = line.rstrip('\n')
                data = line.split(' ')

                # First element of the data is the label. We don't want to process 'Misc' and
                # 'DontCare' labels
                if data[0] != 'Car': continue

                # We do not want to include objects, which are occluded or truncated too much
                if (int(data[2]) >= 2 or float(data[1]) > 0.75): continue

                label = LabelModel()
                label.label = data[0]
                label.truncated = int(float(data[1]))
                label.occluded = int(float(data[2]))
                label.alpha = float(data[3])
                label.x_top_left = int(float(data[4]))
                label.y_top_left = int(float(data[5]))
                label.x_bottom_right = int(float(data[6]))
                label.y_bottom_right = int(float(data[7]))
                label.dim_width = float(data[8])
                label.dim_height = float(data[9])
                label.dim_length = float(data[10])
                label.location_x = float(data[11])
                label.location_y = float(data[12])
                label.location_z = float(data[13])
                label.rotation = float(data[14])

                bb3_label = bb.create_bb3txt_object(label, file_name, calib_matrix, width, height)
                labels.append(bb3_label)
            
        if len(labels) == 0:
            empty = bb.create_empty_object(file_name)
            labels.append(empty)
          

        bb.write_bb3_to_file(labels)
                          
        return labels
            
    def load_specific_label(self, file_name):
        
        image_path = cfg.IMAGE_PATH + '\\' + file_name + '.png'
        label_path = cfg.LABEL_PATH + '\\' + file_name + '.txt'
        calib_path = cfg.CALIB_PATH + '\\' + file_name + '.txt'
        
        if not fw.check_file_exists(image_path):
            assert "Path for image not found"
        
        if not fw.check_file_exists(label_path):
            assert "Path for label not found"
                    
        if not fw.check_file_exists(calib_path):
            assert "Path for calib matrix not found"
        
        image, width, height = self._load_image(image_path)
        calib_matrix = self.load_calibration(calib_path)
        labels = self.load_one_label(label_path, file_name, calib_matrix, width, height)
        
        
        data = DataModel()
        data.image = image
        data.image_path = image_path
        data.image_name = file_name
        data.labels = labels
        data.calib_matrix = calib_matrix

        self.Data.append(data)   
      
    def prepare_data(self, batch_size):
        data = random.sample(self.Data, batch_size)
        result_image = []
        result_image_names = []
        gt_2 = []
        gt_4 = []
        gt_8 = []
        gt_16 = []
        for x in range(batch_size):
            if x >= batch_size:
                break
            result_image.append(data[x].image)
            scale2, scale4, scale8, scale16 = self.create_ground_truth(self.labels_array_for_training(data[x].labels)) 
            gt_2.append(scale2)
            gt_4.append(scale4)
            gt_8.append(scale8)
            gt_16.append(scale16)
            result_image_names.append(data[x].image_name)

            
        #result_label = self.complete_uneven_arrays(result_label)

        self.result_images = np.asarray(result_image)
        self.ground_truths = [np.asarray(gt_2), np.asarray(gt_4), np.asarray(gt_8), np.asarray(gt_16)]
        self.image_names = np.asarray(result_image_names)

        #return np.asarray(result_image), [np.asarray(gt_2), np.asarray(gt_4), np.asarray(gt_8), np.asarray(gt_16)], np.asarray(result_image_names)
           
    def get_train_data(self, batch_size):
        
        data = random.sample(self.Data, batch_size)
        result_image = []
        result_image_names = []
        gt_2 = []
        gt_4 = []
        gt_8 = []
        gt_16 = []
        for x in range(batch_size):
            if x >= batch_size:
                break
            result_image.append(data[x].image)
            scale2, scale4, scale8, scale16 = self.create_ground_truth(self.labels_array_for_training(data[x].labels)) 
            gt_2.append(scale2)
            gt_4.append(scale4)
            gt_8.append(scale8)
            gt_16.append(scale16)
            result_image_names.append(data[x].image_name)

            
        # result_label = self.complete_uneven_arrays(result_label)

        return np.asarray(result_image), [np.asarray(gt_2), np.asarray(gt_4), np.asarray(gt_8), np.asarray(gt_16)], np.asarray(result_image_names)
    
    def create_ground_truth(self, label):
        scale2 = self.create_target_response_map(label,128,64,cfg.RADIUS,cfg.CIRCLE_RATIO,cfg.BOUNDARIES,2)        
        scale4 = self.create_target_response_map(label,64,32,cfg.RADIUS,cfg.CIRCLE_RATIO,cfg.BOUNDARIES,4)        
        scale8 = self.create_target_response_map(label,32,16,cfg.RADIUS,cfg.CIRCLE_RATIO,cfg.BOUNDARIES,8)        
        scale16 = self.create_target_response_map(label,16,8,cfg.RADIUS,cfg.CIRCLE_RATIO,cfg.BOUNDARIES,16)
        # np.save('scale2.npy', scale2)
        return scale2, scale4, scale8, scale16

    def GetObjectBounds(self,radius,circle_ratio,boundaries,scale):
        ideal_size = (2.0 * radius + 1.0) / circle_ratio * scale
        # bound above
        ext_above = ((1.0 - boundaries) * ideal_size) / 2.0 + boundaries * ideal_size
        bound_above = ideal_size + ext_above
        
        # bound below
        diff = ideal_size / 2.0
        ext_below = ((1 - boundaries)* diff) / 2.0 + boundaries * diff
        bound_below = ideal_size - ext_below
        
        return bound_above, bound_below, ideal_size
    
    def create_target_response_map(self, labels, width, height, radius,circle_ratio,boundaries,scale):
        
        result = np.zeros((height,width,8))        
        maps = cv2.split(result)
  
        bound_above, bound_below, ideal = self.GetObjectBounds(radius,circle_ratio,boundaries,scale)
        for i in range(len(labels)):            
            label = labels[i]
            if label[0] == -1:
                # 2. dimension of array have to be same acros 1. dimension, we complete the missing elements with -1, now they will be ignored
                continue
            # 0       1       2       3       4       5       6     7           8           9
            # fblx    fbly    fbrx    fbry    rblx    rbly    ftly  center_x    center_y    largest_dim
            
            if label[9] >= bound_below and label[9] <= bound_above:
                x = int(label[7] / scale)
                y = int(label[8] / scale)
                
                scaling_ratio = 1.0 / scale                
                cv2.circle(maps[0], ( x, y ), int(radius), 1, -1)
                cv2.GaussianBlur(maps[0], (3, 3), 100)

                for k in range(1,8):
                    for l in range(-radius,radius,1):
                        for j in range(-radius,radius,1):
                            xp = x + j
                            yp = y + l
                            if xp >= 0 and xp < width and yp >= 0 and yp < height:
                                if maps[0][yp][xp] > 0.0:
                                    if k ==1 or k == 3 or k == 5:
                                        maps[k][yp][xp] = 0.5 + (label[k-1] - x - j * scale) / ideal
                                    elif k == 2 or k == 4 or k == 6 or k == 7:
                                        maps[k][yp][xp] = 0.5 + (label[k-1] - y - l * scale) / ideal
        
        
        # stack created maps together
        result[:,:,0] = maps[0]
        result[:,:,1] = maps[1]
        result[:,:,2] = maps[2]
        result[:,:,3] = maps[3]
        result[:,:,4] = maps[4]
        result[:,:,5] = maps[5]
        result[:,:,6] = maps[6]
        result[:,:,7] = maps[7]
        
        return result
    
    
    def get_prepared_data(self):
            
        # data = random.sample(self.Data, batch_size)
        result_image = []
        result_label = []
        for x in range(len(self.Data)):
            result_image.append(self.Data[x].image)
            labels = self.labels_array_for_training(self.Data[x].labels)
            result_label.append(labels)
            
        result_label = self.complete_uneven_arrays(result_label)
        return np.asarray(result_image), np.asarray(result_label)
    
    def get_test_data(self, batch_size):
        
        data = random.sample(self.Data, batch_size)
        result_image = []
        result_label = []
        result_images_paths = []
        result_image_names = []
        result_calib_matrices = []
        for x in range(batch_size):
            result_image.append(data[x].image)
            labels = self.labels_array_for_training(data[x].labels)
            result_label.append(labels)
            result_images_paths.append(data[x].image_path)
            result_image_names.append(data[x].image_name)
            result_calib_matrices.append(data[x].calib_matrix)
            
        result_label = self.complete_uneven_arrays(result_label)
        return np.asarray(result_image), np.asarray(result_label), np.asarray(result_images_paths),np.asarray(result_image_names), np.asarray(result_calib_matrices)
    
    def get_test_data_sequentialy(self, i):
        
        data = self.Data[i]
        result_image = []
        result_label = []
        result_images_paths = []
        result_calib_matrices = []
        for x in range(1):
            result_image.append(data.image)
            labels = self.labels_array_for_training(data.labels)
            result_label.append(labels)
            result_images_paths.append(data.image_path)
            result_calib_matrices.append(data.calib_matrix)
            
        result_label = self.complete_uneven_arrays(result_label)
        return np.asarray(result_image), np.asarray(result_label), np.asarray(result_images_paths), np.asarray(result_calib_matrices)
    
    def labels_array_for_training(self,labels):
        label_array = []
        for i in range(len(labels)):
            label = labels[i]
            label_array.append([float(label.fbl_x), float(label.fbl_y), float(label.fbr_x), float(label.fbr_y), float(label.rbl_x), float(label.rbl_y), float(label.ftl_y), float(label.bb_center_x), float(label.bb_center_y), float(label.largest_dim)])
        
        return label_array
    
    def complete_uneven_arrays(self, array, insert_val = -1.0):
        lens = np.array([len(item) for item in array])
        mask = lens[:,None] > np.arange(lens.max())
        out = np.full((mask.shape[0],mask.shape[1],10),insert_val,dtype=np.float32)
        out[mask] = np.concatenate(array)
        return out
    
    def create_pgp_file(self):
        if not fw.check_dir_exists(cfg.PGP_FOLDER):
            fw.create_dir(cfg.PGP_FOLDER)
            
        if fw.check_file_exists(cfg.PGP_FOLDER + r'\pgps_info.txt'):
            fw.delete_file(cfg.PGP_FOLDER + r'\pgps_info.txt')
        
        f= open(cfg.PGP_FOLDER + r'\pgps_info.txt',"w+")

        for i in range(len(self.Data)):
            label = self.Data[i]
            text = label.image_name + ' ' + str(label.calib_matrix[0,0])+ ' ' +  str(label.calib_matrix[0,1])+ ' ' +  str(label.calib_matrix[0,2])+ ' ' +  str(label.calib_matrix[0,3])+ ' ' +  str(label.calib_matrix[1,0])+ ' ' +  str(label.calib_matrix[1,1])+ ' ' +  str(label.calib_matrix[1,2])+ ' ' +  str(label.calib_matrix[1,3])+ ' ' +  str(label.calib_matrix[2,0])+ ' ' +  str(label.calib_matrix[2,1])+ ' ' +  str(label.calib_matrix[2,2])+ ' ' +  str(label.calib_matrix[2,3]) + ' ' +  str(0) + ' ' +  str(1) + ' ' +  str(0) + ' ' +  str(0)
            f.write(text + '\n')

        f.close()
                
    
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        

    
if __name__ == '__main__':
    loader = Loader()
    # loader.load_specific_label("000008")
    loader.load_data()

    
                 