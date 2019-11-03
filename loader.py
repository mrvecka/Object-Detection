import cv2
import numpy as np
import random
from Models.labelModel import LabelModel
from Models.dataModel import DataModel
import config as cfg
import Models.bb3txt as bb
import Services.fileworker as fw

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
        if cfg.IMAGE_PATH != '':
            self.image_path = cfg.IMAGE_PATH
        else:
            self.image_path = cfg.BASE_PATH + r'\images\training\image'

        if cfg.LABEL_PATH != '':
            self.label_path = cfg.LABEL_PATH
        else:
            self.label_path = cfg.BASE_PATH + r'\label\training\label'

        if cfg.CALIB_PATH != '':
            self.calib_path = cfg.CALIB_PATH
        else:
            self.calib_path = cfg.BASE_PATH + r'\calib\training\calib'

        self.amount = cfg.DATA_AMOUNT
        self.start_from = cfg.START_FROM
        self.image_extension = cfg.IMAGE_EXTENSION
        if cfg.IMG_CHANNELS == 1:
            self.colored = False
        else:
            self.colored = True           
            

    def load_data(self):
        """
        Load data from files on specified path.
        Image file name is formated to be "000000" with specified extension

        Input:
            image_path: Path to the images folder
            label_path: Path to the label folder. Load labels from txt files with specific structure according to KITTI documentation
            calib_path: Path to the foder with calibrations files. Load calibration for P2 - left colored images
            amount: Amount of loaded data. For testing purpose.
            extension: Extension of image file.
            colored: Indicate that images are loaded either colored or grayscale
        Returns:
            Data are stored to properties
        """
        assert self.image_path != '', 'Image path not set. Nothing to work with. Check config file.'
        assert self.label_path != '', 'Label path not set. Nothing to work with. Check config file.'
        assert self.calib_path != '', 'Calibration path not set. Nothing to work with. Check config file.'
        print("Start loading training files")

        x = self.start_from
        while x < self.amount + self.start_from:
            # print('loading image',x)
            try:
                image = self._load_image(self.image_path, x, self.colored, self.image_extension)
                if image is None:
                    x+=1
                    continue
            except Exception as e:
                #raise Exception('FAILED LOAD IMAGE FILE','LOADER')
                print('FAILED LOAD IMAGE FILE',x)
                print(e.args)
                x+=1
                continue

            # print('loading calib',x)
            try:
                calib_matrix = self._load_calibration(self.calib_path, x)
                if calib_matrix is None:
                    x+=1
                    continue
            except Exception as e:
                print('FAILED LOAD CALIB FILE',x)
                print(e.args)
                x+=1
                continue

            # print('loading label',x)
            try:
                labels = self._load_label(self.label_path, x)
                if labels is None:
                    x+=1
                    continue
            except Exception as e:
                print('FAILED LOAD LABEL FILE',x)
                print(e.args)
                x+=1
                continue

            if len(labels) == 0:
                x+=1
                continue
            
            data = DataModel()
            data.image = image
            data.labels = labels
            data.calib_matrix = calib_matrix

            self.Data.append(data)
            x+=1
            
        print("Done loading", len(self.Data))

    def _load_image(self, image_path, x, colored, extension):
        """
        Reads a image with number x and based on parameter colored result will have 1 or 3 chanels

        Input:
            image_path: Path to folder with images
            x: number of image
        Returns:
            image as matrix
        """
        image_path = image_path + r'\\' + str(x).zfill(6)+'.'+extension

        if not fw.check_file_exists(image_path):
            return None

        #print(image_path)
        if colored:
            im = cv2.imread(image_path, cv2.IMREAD_COLOR)
        else:
            im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if im.any() == None:
            return None

        cfg.IMG_ORIG_WIDTH = im.shape[1]
        cfg.IMG_ORIG_HEIGHT = im.shape[0]
        return cv2.resize(im, (cfg.IMG_WIDTH,cfg.IMG_HEIGHT), interpolation=cv2.INTER_AREA)

    def _load_label(self, label_path, x):
        """
        Reads a label file to specific image.
        Read only Car labels

        Input:
            label_path: Row-major stored label separated by spaces, first element is the label name
            x: number of image
        Returns:
            LabelModel object
        """
        label_path = label_path + r'\\' + str(x).zfill(6)+'.txt'
        if not fw.check_file_exists(label_path):
            return None

        # check if bb3_files folder exists
        # if exists then load from this file they are pre processed to 33btxt format and ready to use
        # if not load kitti label and create bb3txt file, next time this file will be used
        bb3_path = cfg.BASE_PATH + r'\\' + cfg.BB3_FOLDER
        if fw.check_dir_exists(bb3_path):
            bb3_file_path = bb3_path + '\\' + str(x).zfill(6)+'.txt'
            if fw.check_file_exists(bb3_file_path):
                result = self._load_from_bb3_folder(bb3_file_path)
                return result
            else:
                # if not exists try add 
                return self.load_one_label(x)
                
    def _load_calibration(self, calib_path, x):
        """
        Reads a camera matrix P (3x4) stored in the row-major scheme.

        Input:
            calib_path: Row-major stored matrix separated by spaces, first element is the matrix name
            x: number of image
        Returns:
            camera matrix P 4x4
        """
        calib_path = calib_path + r'\\' + str(x).zfill(6)+'.txt'
        if not fw.check_file_exists(calib_path):
            return None

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
                label.coinfidence = float(data[2])
                label.fbl_x = float(data[3])
                label.fbl_y = float(data[4])
                label.fbr_x = float(data[5])
                label.fbr_y = float(data[6])
                label.rbl_x = float(data[7])
                label.rbl_y = float(data[8])
                label.ftl_y = float(data[9])

                labels.append(label)

        return labels
    
    def convert_labels_to_bb3(self):
        print('converting labels to bb3txt format')
        x = 0
        while x < self.amount:            
            labels = self.load_one_label(x)
            if labels != None and len(labels) != 0:
                bb.write_bb3_to_file(labels)
                
            x+=1
     
    def load_one_label(self,x):
        # print('loading calib',x)
        try:
            calib_matrix = self._load_calibration(self.calib_path, x)
            if calib_matrix is None:
                x+=1
                return None
        except Exception as e:
            print('FAILED LOAD CALIB FILE',x)
            print(e.args)
            x+=1
            return None

        label_path_local = self.label_path + r'\\' + str(x).zfill(6)+'.txt'
        if not fw.check_file_exists(label_path_local):
            return None

        labels = []
        with open(label_path_local, 'r') as infile_label:

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

                bb3_label = bb.create_bb3txt_object(label, str(x).zfill(6)+'.txt', calib_matrix)
                labels.append(bb3_label)
                
        return labels
           
    def get_train_data(self, batch_size):
        
        data = random.sample(self.Data, batch_size)
        result_image = []
        result_label = []
        result_object_count = []
        for x in range(batch_size):
            result_image.append(data[x].image)
            labels = self.labels_array_for_training(data[x].labels)
            result_label.append(labels)
            # tmp = np.asarray(result_label)
            # if tmp.dtype.name != 'float32' and tmp.dtype.name != 'float64':
            #     print(x)
            result_object_count.append(len(labels))
            
            

        result_label = self.complete_uneven_arrays(result_label)
        return np.asarray(result_image), np.asarray(result_label), np.asarray(result_object_count)
    
    def labels_array_for_training(self,labels):
        label_array = []
        for i in range(len(labels)):
            label = labels[i]
            label_array.append([float(label.fbl_x), float(label.fbl_y), float(label.fbr_x), float(label.fbr_y), float(label.rbl_x), float(label.rbl_y), float(label.ftl_y)])
        
        return label_array
    
    def complete_uneven_arrays(self, array, insert_val = -1.0):
        lens = np.array([len(item) for item in array])
        mask = lens[:,None] > np.arange(lens.max())
        out = np.full((mask.shape[0],mask.shape[1],7),insert_val,dtype=np.float32)
        out[mask] = np.concatenate(array)
        return out
    
if __name__ == '__main__':
    loader = Loader()
    loader.load_data()
    
    ba, bs, bd = loader.get_train_data(64)
    print(ba)
    
                 