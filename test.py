
import config as cfg
import numpy as np
import cv2
import tensorflow as tf
import Services.loader as load

import Services.bb_extractor as extract
import Services.drawer as drawer

def start_test():
    
    loader = load.Loader()
    #loader.load_data()
    loader.load_specific_label("000046")
    # loader.load_specific_label("001067")
    
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(cfg.MODEL_PATH + '.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('model/'))
         
        graph = tf.get_default_graph()
        image_placeholder = graph.get_tensor_by_name("input_image_placeholder:0")
        is_training = graph.get_tensor_by_name("input_is_training_placeholder:0")
        
        predict_2 = graph.get_tensor_by_name("output2_convolution:0")
        predict_4 = graph.get_tensor_by_name("output4_convolution:0")
        predict_8 = graph.get_tensor_by_name("output8_convolution:0")
        predict_16 = graph.get_tensor_by_name("output16_convolution:0")
    
        image_batch, label_batch, image_paths, calib_matrices = loader.get_test_data(1)
            
        response_maps_2, response_maps_4, response_maps_8, response_maps_16 = sess.run([predict_2, predict_4, predict_8, predict_16], feed_dict={image_placeholder: image_batch,  is_training: False})
        save_results(response_maps_2,2)            
        save_results(response_maps_4,4)            
        save_results(response_maps_8,8)         
        save_results(response_maps_16,16)
        
        b_boxes_model = extract.extract_bounding_box(response_maps_2, label_batch[0], calib_matrices[0],image_paths[0], 2, 33)
        if not b_boxes_model is None:
            b_boxes_model.file_name = image_paths[0]
            drawer.draw_bounding_boxes(b_boxes_model)
            
        
        b_boxes_model = extract.extract_bounding_box(response_maps_4, label_batch[0], calib_matrices[0],image_paths[0], 4, 66)
        if not b_boxes_model is None:
            b_boxes_model.file_name = image_paths[0]
            drawer.draw_bounding_boxes(b_boxes_model)
        
        b_boxes_model = extract.extract_bounding_box(response_maps_8, label_batch[0], calib_matrices[0],image_paths[0], 8, 133)
        if not b_boxes_model is None:
            b_boxes_model.file_name = image_paths[0]
            drawer.draw_bounding_boxes(b_boxes_model)
        
        b_boxes_model = extract.extract_bounding_box(response_maps_16, label_batch[0], calib_matrices[0],image_paths[0], 16, 266)
        if not b_boxes_model is None:
            b_boxes_model.file_name = image_paths[0]
            drawer.draw_bounding_boxes(b_boxes_model)
        
def save_results(maps, scale):
    result = cv2.split(np.squeeze(maps,axis=0))
    tmp = (result[0] - result[0].min()) * (255/(result[0].max() - result[0].min()))
    tmp[tmp<150] = 0
    path = r"C:\Users\Lukas\Documents\Object detection\result_test_s"+str(scale)+r"\response_map_0.jpg"
    cv2.imwrite(path, tmp)
    path = r"C:\Users\Lukas\Documents\Object detection\result_test_s"+str(scale)+r"\response_map_1.jpg"
    cv2.imwrite(path, (result[1] - result[1].min()) * (255/(result[1].max() - result[1].min())))
    path = r"C:\Users\Lukas\Documents\Object detection\result_test_s"+str(scale)+r"\response_map_2.jpg"
    cv2.imwrite(path, (result[2] - result[2].min()) * (255/(result[2].max() - result[2].min())))
    path = r"C:\Users\Lukas\Documents\Object detection\result_test_s"+str(scale)+r"\response_map_3.jpg"
    cv2.imwrite(path, (result[3] - result[3].min()) * (255/(result[3].max() - result[3].min())))
    path = r"C:\Users\Lukas\Documents\Object detection\result_test_s"+str(scale)+r"\response_map_4.jpg"
    cv2.imwrite(path, (result[4] - result[4].min()) * (255/(result[4].max() - result[4].min())))
    path = r"C:\Users\Lukas\Documents\Object detection\result_test_s"+str(scale)+r"\response_map_5.jpg"
    cv2.imwrite(path, (result[5] - result[5].min()) * (255/(result[5].max() - result[5].min())))
    path = r"C:\Users\Lukas\Documents\Object detection\result_test_s"+str(scale)+r"\response_map_6.jpg"
    cv2.imwrite(path, (result[6] - result[6].min()) * (255/(result[6].max() - result[6].min())))
    path = r"C:\Users\Lukas\Documents\Object detection\result_test_s"+str(scale)+r"\response_map_7.jpg"
    cv2.imwrite(path, (result[7] - result[7].min()) * (255/(result[7].max() - result[7].min())))
    
if __name__ == "__main__":
    start_test()