import tensorflow as tf


def normalization_layer(data,is_training):
    # normalization
    out_layer = tf.layers.batch_normalization(data, training=is_training)
    
    return out_layer

def normalize_input_data(data,is_training):
    mean, variance = tf.nn.moments(data, [1,2,3], keep_dims=True)
    out_layer = tf.nn.batch_normalization(data, mean, variance, None, None, 0.001, "normalization")
    return out_layer

def create_detection_network_layer(name, input_data, filter_kernel_shape, num_input_channels, num_output_channels, dilation, stride, is_training):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filter_shape = [filter_kernel_shape[0], filter_kernel_shape[1], num_input_channels, num_output_channels]

    # initialize weights anddd bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filter_shape, stddev=0.1), name=name+'_weights', trainable=True)

    # bias = tf.Variable(tf.truncated_normal(([num_output_channels]), stddev=0.1), name=name+'_b', trainable=False)
    bias = tf.constant(0.1,shape=[num_output_channels],name=name+'_biases')
    
    # setup the convolutional layer network
    out_layer = tf.nn.conv2d(input_data, weights, strides=[1, stride, stride, 1], padding='SAME', dilations=[1, dilation, dilation, 1], name=name+'_convolution')

    # add bias
    out_layer = tf.add(out_layer, bias, name=name+'_bias_add_op')

    # apply a relu non-linea activation
    out_layer = tf.nn.relu(out_layer, name=name+'_relu_activation')
    
    return out_layer

def create_detection_network_output_layer(name, input_data, filter_kernel_shape, num_input_channels, num_output_channels, dilation, stride, is_training):
        
    # setup the filter input shape for tf.nn.conv_2d
    conv_filter_shape = [filter_kernel_shape[0], filter_kernel_shape[1], num_input_channels, num_output_channels]

    # initialize weights anddd bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filter_shape, stddev=0.1), name=name+'_weights', trainable=True)

    # bias = tf.Variable(tf.truncated_normal(([num_output_channels]), stddev=0.1), name=name+'_b', trainable=False)
    bias = tf.constant(0.1,shape=[num_output_channels],name=name+'_biases')
    
    # setup the convolutional layer network
    out_layer = tf.nn.conv2d(input_data, weights, strides=[1, stride, stride, 1], padding='SAME', dilations=[1, dilation, dilation, 1], name=name+'_convolution')
    
    return out_layer

def create_detection_network_pool_layer(layer, pool_shape, name):
    # perform max pooling
    ksize = [1, pool_shape[0],pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    layer = tf.nn.max_pool(layer, ksize=ksize, strides=strides, padding='SAME', name=name+'pool')
    
    return layer

def create_detection_network_fcn(net, is_training, name):
    flattered = tf.reshape(net, [-1, 8*8*512])

    # setup some weights and bias values for this layer, then activate with ReLU
    weight = tf.Variable(tf.truncated_normal([8 * 8 * 512, 3072], stddev=0.1), name=name+'weight', trainable=True)
    bias = tf.Variable(tf.truncated_normal([8], stddev=0.1), name=name+'bias')
    dense_layer1 = tf.add(tf.matmul(flattered, weight), bias)

    dense_layer1 = tf.layers.batch_normalization(dense_layer1, training=is_training)

    dense_layer1 = tf.nn.relu(dense_layer1)
    