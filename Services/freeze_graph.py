import config as cfg
import tensorflow as tf

def freeze_and_save():

    new_saver = tf.train.import_meta_graph(cfg.MODEL_PATH + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    
    try:
        with tf.Session() as sess:
            
            new_saver.restore(sess,cfg.MODEL_PATH)
            
            output_node_names = ["scale_2/output2/Conv2D","scale_4/output4/Conv2D","scale_8/output8/Conv2D","scale_16/output16/Conv2D"]
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names)
            
            with tf.gfile.GFile(cfg.FROZEN_MODEL_PATH, "wb") as f:
                f.write(output_graph_def.SerializeToString())        
    except:
        print("Cannot freeze tensorflow graph. Check output node names!")
        
    