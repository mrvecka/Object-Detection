__date__   = '14/05/2020'
__author__ = 'Lukas Mrvecka'
__email__  = 'lukas.mrvecka.st@vsb.cz'

from Network.object_detection_network import ObjectDetectionModel
import config as cfg
import numpy as np
import cv2
import tensorflow as tf
import Services.loader as load
from Services.timer import Timer


import Services.bb_extractor as extract
import Services.non_maxima_supression as NMS
import Services.drawer as drawer
import Services.evaluate as evaluator
import Services.fileworker as fw


def start_test():

    if cfg.SPECIFIC_TEST_DATA == "":
        print("Test data not specified!!! Check config file")
        return
    
    loader = load.Loader()
    # loader.load_data()
    
    loader.load_specific_label(cfg.SPECIFIC_TEST_DATA)
    loader.prepare_data(1)
    model = ObjectDetectionModel([3,3],'Testing Object Detection Model',1)
    model.build((1,cfg.IMG_HEIGHT,cfg.IMG_WIDTH,cfg.IMG_CHANNELS))
    # model.build(input_shape=(None,cfg.IMG_HEIGHT,cfg.IMG_WIDTH,cfg.IMG_CHANNELS))               
    image_batch, label_batch, image_paths,image_names, calib_matrices = loader.get_test_data(1)
    im, lbl, name = loader.get_train_data(1)
    
    base_path = r".\model"
    if not fw.check_dir_exists(base_path):
        print("Weights for model not found. Tried path ", base_path)
        print("In project folder create directory model which will contain model_weight.h5 file with saved weights")
        return
    
    model.load_weights(base_path + r"\model_weights.h5")
    # image_batch, label_batch, names = loader.get_train_data(1)
    t = Timer()
    t.start()
    out = model(image_batch,False)
    _ = t.stop()
    t.get_formated_time()
    print("Formated time ",t.get_formated_time())
    del model
    save_results(out[0].numpy(), 2)
    save_results(out[1].numpy(), 4)
    save_results(out[2].numpy(), 8)
    save_results(out[3].numpy(), 16)

    # compare_results(out[0].numpy(), label_batch[0],2)
    # compare_results(out[1].numpy(), label_batch[1],4)
    # compare_results(out[2].numpy(), lbl[2],8)
    # compare_results(out[3].numpy(), label_batch[3],16)
    
    extract_and_show(out[0].numpy(), out[1].numpy(), out[2].numpy(), out[3].numpy(),
                    label_batch, calib_matrices, image_paths)
    # show_triangle(response_maps_2,response_maps_4,response_maps_8,response_maps_16,image_paths)


def save_results(maps, scale):
    result = cv2.split(np.squeeze(maps, axis=0))
    tmp = (result[0] - result[0].min()) * (255/(result[0].max() - result[0].min()))
    tmp[tmp < 150] = 0
    
    base_path = r".\result_test_s" + str(scale)
    if not fw.check_and_create_folder(base_path):
        print("Unable to create folder for results. Tried path: ", base_path)
        return
    
    path = base_path +r"\response_map_0.jpg"
    cv2.imwrite(path, tmp)
    
    path = base_path+r"\response_map_1.jpg"
    cv2.imwrite(path, maps[0, :, :, 1])
    
    path = base_path+r"\response_map_2.jpg"
    cv2.imwrite(path, maps[0, :, :, 2])
    
    path = base_path+r"\response_map_3.jpg"
    cv2.imwrite(path, maps[0, :, :, 3])
    
    path = base_path+r"\response_map_4.jpg"
    cv2.imwrite(path, maps[0, :, :, 4])
    
    path = base_path+r"\response_map_5.jpg"
    cv2.imwrite(path, maps[0, :, :, 5])
    
    path = base_path+r"\response_map_6.jpg"
    cv2.imwrite(path, maps[0, :, :, 6])
    
    path = base_path+r"\response_map_7.jpg"
    cv2.imwrite(path, maps[0, :, :, 7])


def extract_and_show(response_maps_2, response_maps_4, response_maps_8, response_maps_16, 
                     label_batch, calib_matrices, image_paths):


    result = extract.extract_bounding_box(
        response_maps_2, label_batch[0], calib_matrices[0], image_paths[0], 2, 33)
    result = NMS.start_nms(result,2)
    evaluator.evaluate(result)
    if not result is None:
        result.file_name = image_paths[0]
        drawer.draw_bounding_boxes(result, 2)

    result = extract.extract_bounding_box(
        response_maps_4, label_batch[0], calib_matrices[0], image_paths[0], 4, 66)
    result = NMS.start_nms(result,4)
    evaluator.evaluate(result)
    if not result is None:
        result.file_name = image_paths[0]
        drawer.draw_bounding_boxes(result, 4)

    result = extract.extract_bounding_box(
        response_maps_8, label_batch[0], calib_matrices[0], image_paths[0], 8, 133)
    result = NMS.start_nms(result,8)
    evaluator.evaluate(result)
    if not result is None:
        result.file_name = image_paths[0]
        drawer.draw_bounding_boxes(result, 8)

    result = extract.extract_bounding_box(
        response_maps_16, label_batch[0], calib_matrices[0], image_paths[0], 16, 266)
    result = NMS.start_nms(result,16)
    evaluator.evaluate(result)
    if not result is None:
        result.file_name = image_paths[0]
        drawer.draw_bounding_boxes(result, 16)


def show_triangle(response_maps_2, response_maps_4, response_maps_8, response_maps_16, image_paths):

    extract.showResults(response_maps_2, image_paths[0], 2, 33)
    extract.showResults(response_maps_4, image_paths[0], 4, 66)
    extract.showResults(response_maps_8, image_paths[0], 8, 133)
    extract.showResults(response_maps_16, image_paths[0], 16, 266)
    
    
def compare_results(out, label, scale):
    prob = out[0,:,:,0]
    
    prob = (prob - prob.min()) * (255/(prob.max() - prob.min()))
    prob[prob < 150] = 0
    
    label_prob = label[0,:,:,0]
    label_prob = (label_prob - label_prob.min()) * (255/(label_prob.max() - label_prob.min()))
    label_prob[label_prob < 150] = 0
     
    compared = label_prob - prob
    compared = (compared - compared.min()) * (255/(compared.max() - compared.min()))
    
    # prob = cv2.resize(prob, (220,120), interpolation = cv2.INTER_AREA)
    # label_prob = cv2.resize(label_prob, (220,120), interpolation = cv2.INTER_AREA)
    # compared = cv2.resize(compared, (220,120))
    
    cv2.imshow("output",prob)
    cv2.imshow("label", label_prob)
    cv2.imshow("compared", compared)
    
    cv2.waitKey()
    
    base_path = r".\result_compare_s" + str(scale)
    if not fw.check_and_create_folder(base_path):
        print("Unable to create folder for results. Tried path: ", base_path)
        return
    
    path = base_path +r"\output.jpg"
    cv2.imwrite(path, prob)
    
    path = base_path +r"\ground_truth.jpg"
    cv2.imwrite(path, label_prob)
    
    path = base_path +r"\compared.jpg"
    cv2.imwrite(path, compared)
    


if __name__ == "__main__":
    start_test()
