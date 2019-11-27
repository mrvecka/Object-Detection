import config as cfg
import numpy as np
import cv2
import tensorflow as tf
import loader as load


def start_test():
    
    loader = load.Loader()
    loader.load_data()
    
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(cfg.MODEL_PATH + '.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('model/'))
        
        graph = tf.get_default_graph()
        image_placeholder = graph.get_tensor_by_name("input_image_placeholder:0")
        is_training = graph.get_tensor_by_name("input_is_training_placeholder:0")
        
        predict = graph.get_tensor_by_name("output1_final:0")
    
        image_batch, _, _, image_paths = loader.get_train_data(1)
            
        response_maps = sess.run(predict, feed_dict={image_placeholder: image_batch,  is_training: False})
        result = cv2.split(np.squeeze(response_maps,axis=0))
        path1 = r"C:\Users\Lukas\Documents\Object detection\result_test\response_map0.jpg"
        cv2.imwrite(path1, 255 * result[0])
        path1 = r"C:\Users\Lukas\Documents\Object detection\result_test\response_map1.jpg"
        cv2.imwrite(path1, 255* result[1])
        path1 = r"C:\Users\Lukas\Documents\Object detection\result_test\response_map2.jpg"
        cv2.imwrite(path1, 255*result[2])
        path1 = r"C:\Users\Lukas\Documents\Object detection\result_test\response_map3.jpg"
        cv2.imwrite(path1, 255*result[3])
        path1 = r"C:\Users\Lukas\Documents\Object detection\result_test\response_map4.jpg"
        cv2.imwrite(path1, 255*result[4])
        path1 = r"C:\Users\Lukas\Documents\Object detection\result_test\response_map5.jpg"
        cv2.imwrite(path1, 255*result[5])
        path1 = r"C:\Users\Lukas\Documents\Object detection\result_test\response_map6.jpg"
        cv2.imwrite(path1, 255*result[6])
        path1 = r"C:\Users\Lukas\Documents\Object detection\result_test\response_map7.jpg"
        cv2.imwrite(path1, 255*result[7])
        
        print(image_paths[0])
    

if __name__ == "__main__":
    start_test()