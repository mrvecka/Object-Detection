from Network.object_detection_network import ObjectDetectionModel
import config as cfg
import numpy as np
import cv2
import tensorflow as tf
import Services.loader as load


import Services.bb_extractor as extract
import Services.non_maxima_supression as NMS
import Services.drawer as drawer
import Services.evaluate as evaluator
import Services.fileworker as fw


def start_test():

    loader = load.Loader()
    # loader.load_data()
    loader.load_specific_label("000003")
    model = ObjectDetectionModel([3,3],'Testing Object Detection Model')
    model.build(input_shape=(None,cfg.IMG_HEIGHT,cfg.IMG_WIDTH,cfg.IMG_CHANNELS))               
    model.load_weights(cfg.MODEL_PATH_H5)
            
        
    image_batch, label_batch, image_paths,image_names, calib_matrices = loader.get_test_data(1)
    out_2, out_4, out_8, out_16 = model(image_batch,False)
    del model
    # save_results(out_2.numpy(), 2)
    # save_results(out_4.numpy(), 4)
    # save_results(out_8.numpy(), 8)
    # save_results(out_16.numpy(), 16)

    extract_and_show(out_2.numpy(), out_4.numpy(), out_8.numpy(), out_16.numpy(),
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


def extract_and_show(response_maps_2, response_maps_4, response_maps_8, response_maps_16, label_batch, calib_matrices, image_paths):


    b_boxes_model = extract.extract_bounding_box(
        response_maps_2, label_batch[0], calib_matrices[0], image_paths[0], 2, 33)
    nms_result = NMS.start_nms(b_boxes_model,2)
    evaluator.evaluate(nms_result)
    if not nms_result is None:
        nms_result.file_name = image_paths[0]
        drawer.draw_bounding_boxes(nms_result, 2)

    b_boxes_model = extract.extract_bounding_box(
        response_maps_4, label_batch[0], calib_matrices[0], image_paths[0], 4, 66)
    nms_result = NMS.start_nms(b_boxes_model,4)
    evaluator.evaluate(nms_result)
    if not nms_result is None:
        nms_result.file_name = image_paths[0]
        drawer.draw_bounding_boxes(nms_result, 4)

    b_boxes_model = extract.extract_bounding_box(
        response_maps_8, label_batch[0], calib_matrices[0], image_paths[0], 8, 133)
    nms_result = NMS.start_nms(b_boxes_model,8)
    evaluator.evaluate(nms_result)
    if not nms_result is None:
        nms_result.file_name = image_paths[0]
        drawer.draw_bounding_boxes(nms_result, 8)

    b_boxes_model = extract.extract_bounding_box(
        response_maps_16, label_batch[0], calib_matrices[0], image_paths[0], 16, 266)
    nms_result = NMS.start_nms(b_boxes_model,16)
    evaluator.evaluate(nms_result)
    if not nms_result is None:
        nms_result.file_name = image_paths[0]
        drawer.draw_bounding_boxes(nms_result, 16)


def show_triangle(response_maps_2, response_maps_4, response_maps_8, response_maps_16, image_paths):

    extract.showResults(response_maps_2, image_paths[0], 2, 33)
    extract.showResults(response_maps_4, image_paths[0], 4, 66)
    extract.showResults(response_maps_8, image_paths[0], 8, 133)
    extract.showResults(response_maps_16, image_paths[0], 16, 266)


if __name__ == "__main__":
    start_test()
