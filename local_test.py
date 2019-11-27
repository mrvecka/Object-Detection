from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorboard as tb
# from Network.network_creator import NetworkCreator
# import Models.bb3txt as bb
import cv2
import loader as load

from tensorflow-gpu import version; print(version.VERSION)

# import os
# import sys
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# array_val = np.zeros((5,5))
# array_val[2][2] = 1

# # array_val = np.asarray([[0,0,0,0,0],
# #              [0,0,0,0,0],
# #              [0,0,1,0,0],
# #              [0,0,0,0,0],
# #              [0,0,0,0,0]])


# loader = load.Loader()
# loader.load_data()

# print(tf.VERSION)
# print(tb.version.VERSION)

# img = cv2.imread(r"C:\Users\Lukas\Documents\Object detection\result\response_map2.jpg",cv2.IMREAD_GRAYSCALE)
# img2 = img * 255

# cv2.imshow("multilied", img2)
# cv2.waitKey(0)




# print(array_val)
# ground = np.zeros((256,512))

# cv2.circle(ground, (50,100 ), 5, (255, 255, 255), -1)
# ground = cv2.GaussianBlur(ground, (5, 5), 1)

# cv2.imshow("bla",ground)
# cv2.waitKey(0)



# print(tf.reduce_sum(5))

# def test_fn(count,sess):
    

#     # def run_init(sess):
#     #     init = tf.global_variables_initializer()
#     #     sess.run(init)
#     #     return 0
    
#     # tf.cond( tf.equal(create_sess,tf.constant(True)), lambda: "neinicializujem", lambda: run_init(sess) )
#     _ = tf.print(count,[count], "value of Count: ", output_stream=sys.stdout)
#     data = count.eval(session=sess)
#     print(count[0].eval())
#     _ = tf.print(data, [data],"evaluated data: ", output_stream=sys.stdout)
#     data_tmp = tf.reduce_sum(data)

#     _ = tf.print(data_tmp, [data_tmp], "summarized: ", output_stream=sys.stdout)
    
#     maps = np.zeros((8,30,30))
    
#     i = tf.constant(0, dtype=tf.float32)
#     while_condition = lambda i: tf.less(i, count[3])
#     def body(i):
#         # do something here which you want to do in your loop
#         # increment i
        
#         cv2.circle(maps[0], (int(count[3]), int(count[5])), 2, (255, 255, 255), -1)
        
#         return [tf.add(i, 1)]

#     # do the loop:
#     r = tf.while_loop(while_condition, body, [i])
#     return r
    
    
#     # a = sess.run(count[3])
#     # b = tf.print(a,[a], "bla")
#     # sess.run(b)
#     # a = tf.cond(tf.equal(count[3], 5), lambda: tf.reduce_sum(count[5]), lambda: tf.reduce_sum(count[0]))
    
#     # return a
    


# lbl = [1,1,1,5,1,6,1,1]


# count = tf.Variable(tf.zeros(8),tf.float32)
# create_sess = tf.placeholder(tf.bool)

# init = tf.global_variables_initializer()
# sess = tf.InteractiveSession()
        
# sess.run(init)      

# test = test_fn(count,sess)

# sess.close()


# with tf.Session() as session:
#     session.run(init)

# # initialise the variables
#     # after model is fully trained
#     # path = saver.save(session, r"C:\Users\Lukas\Documents\Python Projects\Tensorflow-Projects\Image_Clasification\graphs\my_net.ckpt")
#     #writer = tf.summary.FileWriter(graph_path, session.graph)

#     print(session.run(test, 
#                 feed_dict={count: lbl,create_sess:True}))
    
    
#     #print(reshaped.eval())

