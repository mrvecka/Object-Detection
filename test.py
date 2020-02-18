
import config as cfg
import numpy as np
import cv2
import tensorflow as tf
import Services.loader as load

import Services.bb_extractor as extract
import Services.non_maxima_supression as NMS
import Services.drawer as drawer


def start_test():

    loader = load.Loader()
    # loader.load_data()
    loader.load_specific_label("000010")
    # loader.load_specific_label("001067")
    new_saver = tf.train.import_meta_graph(cfg.MODEL_PATH + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    
    with tf.Session() as sess:
        
        new_saver.restore(sess,cfg.MODEL_PATH)
            
        graph = tf.get_default_graph()
        image_placeholder = graph.get_tensor_by_name(
            "input_image_placeholder:0")
        is_training = graph.get_tensor_by_name(
            "input_is_training_placeholder:0") 

        predict_2 = graph.get_tensor_by_name("scale_2/output2/Conv2D:0")
        predict_4 = graph.get_tensor_by_name("scale_4/output4/Conv2D:0")
        predict_8 = graph.get_tensor_by_name("scale_8/output8/Conv2D:0")
        predict_16 = graph.get_tensor_by_name("scale_16/output16/Conv2D:0")

        image_batch, label_batch, image_paths,image_names, calib_matrices = loader.get_test_data(1)

        response_maps_2, response_maps_4, response_maps_8, response_maps_16 = sess.run(
            [predict_2, predict_4, predict_8, predict_16], feed_dict={image_placeholder: image_batch,  is_training: False})
        save_results(response_maps_2, 2)
        save_results(response_maps_4, 4)
        save_results(response_maps_8, 8)
        save_results(response_maps_16, 16)

        extract_and_show(response_maps_2, response_maps_4, response_maps_8,
                         response_maps_16, label_batch, calib_matrices, image_paths)
        # show_triangle(response_maps_2,response_maps_4,response_maps_8,response_maps_16,image_paths)


def save_results(maps, scale):
    result = cv2.split(np.squeeze(maps, axis=0))
    tmp = (result[0] - result[0].min()) * \
        (255/(result[0].max() - result[0].min()))
    tmp[tmp < 150] = 0
    path = r"C:\Users\Lukas\Documents\Object detection\result_test_s" + \
        str(scale)+r"\response_map_0.jpg"
    cv2.imwrite(path, tmp)
    path = r"C:\Users\Lukas\Documents\Object detection\result_test_s" + \
        str(scale)+r"\response_map_1.jpg"
    cv2.imwrite(path, (maps[0, :, :, 1] - maps[0, :, :, 1].min())
                * (255/(maps[0, :, :, 1].max() - maps[0, :, :, 1].min())))
    path = r"C:\Users\Lukas\Documents\Object detection\result_test_s" + \
        str(scale)+r"\response_map_2.jpg"
    cv2.imwrite(path, maps[0, :, :, 2])
    path = r"C:\Users\Lukas\Documents\Object detection\result_test_s" + \
        str(scale)+r"\response_map_3.jpg"
    cv2.imwrite(path, maps[0, :, :, 3])
    path = r"C:\Users\Lukas\Documents\Object detection\result_test_s" + \
        str(scale)+r"\response_map_4.jpg"
    cv2.imwrite(path, maps[0, :, :, 4])
    path = r"C:\Users\Lukas\Documents\Object detection\result_test_s" + \
        str(scale)+r"\response_map_5.jpg"
    cv2.imwrite(path, maps[0, :, :, 5])
    path = r"C:\Users\Lukas\Documents\Object detection\result_test_s" + \
        str(scale)+r"\response_map_6.jpg"
    cv2.imwrite(path, maps[0, :, :, 6])
    path = r"C:\Users\Lukas\Documents\Object detection\result_test_s" + \
        str(scale)+r"\response_map_7.jpg"
    cv2.imwrite(path, maps[0, :, :, 7])


def extract_and_show(response_maps_2, response_maps_4, response_maps_8, response_maps_16, label_batch, calib_matrices, image_paths):

    b_boxes_model = extract.extract_bounding_box(
        response_maps_2, label_batch[0], calib_matrices[0], image_paths[0], 2, 33)
    nms_result = NMS.start_nms(b_boxes_model,2)
    if not nms_result is None:
        nms_result.file_name = image_paths[0]
        drawer.draw_bounding_boxes(nms_result, 2)

    b_boxes_model = extract.extract_bounding_box(
        response_maps_4, label_batch[0], calib_matrices[0], image_paths[0], 4, 66)
    nms_result = NMS.start_nms(b_boxes_model,4)
    if not nms_result is None:
        nms_result.file_name = image_paths[0]
        drawer.draw_bounding_boxes(nms_result, 4)

    b_boxes_model = extract.extract_bounding_box(
        response_maps_8, label_batch[0], calib_matrices[0], image_paths[0], 8, 133)
    nms_result = NMS.start_nms(b_boxes_model,8)
    if not nms_result is None:
        nms_result.file_name = image_paths[0]
        drawer.draw_bounding_boxes(nms_result, 8)

    b_boxes_model = extract.extract_bounding_box(
        response_maps_16, label_batch[0], calib_matrices[0], image_paths[0], 16, 266)
    nms_result = NMS.start_nms(b_boxes_model,16)
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
