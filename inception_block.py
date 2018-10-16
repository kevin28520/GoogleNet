import tensorflow as tf

def inception_module(input):
    input = tf.expand_dims(input, axis=-1)
    print('input shape: ', input.get_shape().as_list())

    conv_1x1 = tf.layers.conv2d(input, filters=64, kernel_size=(1, 1), strides=1, padding='same', activation=tf.nn.relu)
    print('conv_1x1 shape: ', conv_1x1.get_shape().as_list())

    conv_3x3_reduce = tf.layers.conv2d(input, filters=96, kernel_size=(1, 1), strides=1, padding='same', activation=tf.nn.relu)

    conv_3x3 = tf.layers.conv2d(conv_3x3_reduce, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu)
    print('conv_3x3 shape: ', conv_3x3.get_shape().as_list())
    
    conv_5x5_reduce = tf.layers.conv2d(input, filters=16, kernel_size=(1,1), strides=1, padding='same', activation=tf.nn.relu)
    conv_5x5 = tf.layers.conv2d(conv_5x5_reduce, filters=32, kernel_size=(5,5), strides=(1,1), padding='same', activation=tf.nn.relu)
    print('conv_5x5 shape: ', conv_5x5.get_shape().as_list())
    
    maxpooling = tf.layers.max_pooling2d(input, pool_size=(3,3), strides=(1,1), padding='same')
    maxpooling_proj = tf.layers.conv2d(maxpooling, filters=32, kernel_size=(1,1), strides=(1,1), padding='same', activation=tf.nn.relu)
    print('maxpooling_projection shape: ', maxpooling_proj.get_shape().as_list())
    
    concatenation = tf.concat(values=[conv_1x1, conv_3x3, conv_5x5, maxpooling_proj], axis=-1)
    print('concatenated shape: ', concatenation.get_shape().as_list())

    return concatenation
