import tensorflow as tf

'''
    This script contains many different networks. 
    ALso contains first experiments on Convolution layers were created using the native tf.nn.conv2d
    and native tf.nn.max_pool

    Later on, we use the latest library from tf.layers. 
'''

# --- Layer building blocks --------------
def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, strides, padding, activation, name):
    # setup the filter input shape for tf.nn.conv_2d
    #print("Conv {} [{},{}] stride [{},{}] with {} channels ".format(name, filter_shape[0], filter_shape[1], strides[0], strides[1], num_filters))
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]
    
    # initialise weights and bias for the filter
    # weights = tf.get_variable('{}_W'.format(name), conv_filt_shape, tf.truncated_normal(conv_filt_shape, stddev=0.03))
    # bias = tf.get_variable('{}_b'.format(name), [num_filters], tf.truncated_normal([num_filters]))
    weights = tf.get_variable('{}_W'.format(name), conv_filt_shape, dtype=tf.float32)
    bias = tf.get_variable('{}_b'.format(name), [num_filters], dtype=tf.float32)
    

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, strides[0], strides[1], 1], padding=padding)

    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    if activation:
        out_layer = activation(out_layer)

    return out_layer

def create_new_pooling_layer(input_data, pool_shape, strides, padding, name):
    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, strides[0], strides[1], 1]
    out_layer = tf.nn.max_pool(input_data, ksize=ksize, strides=strides, padding=padding)

    return out_layer

def create_new_conv_and_pooling_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    print("Conv {} [{},{}] stride [1,1] with {} channels ".format(name, filter_shape[0], filter_shape[1], num_filters))
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]
    

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name='{}_W'.format(name))
    bias = tf.Variable(tf.truncated_normal([num_filters]), name='{}_b'.format(name))

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')

    return out_layer

# ----------------- Network builds --------------
def build_network(name, input_x, input_dimension, output_size, is_training=True, keep_prob=1):
    if  (name.lower() == "landmark"):
        return build_landmark_network(input_x, input_dimension, output_size)
    # if  (name.lower() == "double_landmark"):
    #     return build_double_landmark(input_x, output_size)
    elif (name.lower() == "simple"):
        return build_simple_network(input_x, input_dimension, output_size)
    elif (name.lower() == "mnist"):
        return build_mnist_cnn(input_x, input_dimension, output_size)
    elif (name.lower() == "simple_alexnet"):
        return build_simple_alexnet(input_x, input_dimension, output_size)
    elif (name.lower() == "alexnet"):
        return build_alexnet(input_x, input_dimension, output_size)
    elif (name.lower() == "whale"):
        return build_whale_head_localizer(input_x, input_dimension, output_size, is_training, keep_prob)        
    elif (name.lower() == "local"):
        return build_localizer_net(input_x, input_dimension, output_size, is_training, keep_prob)
    else:
        return None

def build_registration_network(xy):
    '''
        Localisation network
    '''
    model_name = 'DIRnet'
    # layer 1
    conv1 = create_new_conv_layer(xy, 2, 64, [3, 3], [1, 1], padding='SAME', activation=tf.nn.relu, name='conv1') # should be normalized
    pool1 = tf.nn.avg_pool(conv1, [1,2,2,1], [1,2,2,1], 'SAME', name='pool1')
    # layer 2
    conv2 = create_new_conv_layer(pool1, 64, 128, [3, 3], [1, 1], padding='SAME', activation=tf.nn.relu, name='conv2') # should be normalized
    conv3 = create_new_conv_layer(conv2, 128, 128, [3, 3], [1, 1], padding='SAME', activation=tf.nn.relu, name='conv3') # should be normalized
    pool2 = tf.nn.avg_pool(conv3, [1,2,2,1], [1,2,2,1], 'SAME', name='pool2')
    # layer 3
    conv4 = create_new_conv_layer(pool2, 128, 2, [3, 3], [1, 1], padding='SAME', activation=None, name='conv4') 
    pool3 = tf.nn.avg_pool(conv4, [1,2,2,1], [1,2,2,1], 'SAME', name='pool3')

    pool3 = tf.reshape(pool3, [-1, 16, 16, 2])
    return model_name, pool3

def build_mnist_cnn(x_shaped, input_dimension, output_size): 
    model_name = "mnist_cnn"
    layer1 = create_new_conv_and_pooling_layer(x_shaped, 1, 8, [2, 2], [1, 1], name='layer1')
    print("Layer 1 shape", layer1.shape)
    layer2 = create_new_conv_and_pooling_layer(layer1, 8, 16, [2, 2], [1, 1], name='layer2')
    print("Layer 2 shape", layer2.shape)

    flattened = tf.reshape(layer2, [-1, 1 * 1 * 16])
    print(flattened)

    # setup some weights and bias values for this layer, then activate with ReLU
    wd1 = tf.Variable(tf.truncated_normal([1 * 1 * 16, 100], stddev=0.03), name='wd1')
    print(wd1)
    bd1 = tf.Variable(tf.truncated_normal([100], stddev=0.01), name='bd1')
    print(bd1)
    dense_layer1 = tf.matmul(flattened, wd1) + bd1
    dense_layer1 = tf.nn.leaky_relu(dense_layer1)
    #dense_layer1 = tf.nn.sigmoid(dense_layer1)
    print(dense_layer1)

    # another layer with softmax activations
    wd2 = tf.Variable(tf.truncated_normal([100, output_size], stddev=0.03), name='wd2')
    print(wd2)
    bd2 = tf.Variable(tf.truncated_normal([output_size], stddev=0.01), name='bd2')
    print(bd2)
    dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
    print(dense_layer2)
    y_ = tf.nn.softmax(dense_layer2, name='y_')
    print(y_)
    return model_name, y_

def build_yolo_network(input_x, input_dimension, output_size):
    '''
    Fast YOLO uses a neural network with fewer convolutional layers (9 instead of 24) 
    and fewer filters in those layers. Other than the size of the network, 
    all training and testing parameters are the same between YOLO and Fast YOLO.
    '''
    model_name = "yolo"
    
def build_simple_network(input_x, input_dimension, output_size):
    '''
        Taken from https://wiki.tum.de/display/lfdv/Facial+Landmark+Detection
    '''
    model_name = "simple_conv"
    layer1 = create_new_conv_and_pooling_layer(input_x, 1, 16, [5, 5], [2, 2], name='layer1')
    layer2 = create_new_conv_and_pooling_layer(layer1, 16, 48, [3, 3], [2, 2], name='layer2')
    layer3 = create_new_conv_and_pooling_layer(layer2, 48, 64, [3, 3], [2, 2], name='layer3')
    layer4 = create_new_conv_layer(layer3, 64, 64, [2, 2], [1,1], 'SAME', activation=tf.nn.relu, name='layer4')
    print(layer4)
    
    flattened = tf.reshape(layer4, [-1, layer4.shape[1] * layer4.shape[2] * 64])
    
    layer4_size = input_dimension//8
    wd1 = tf.Variable(tf.truncated_normal([layer4_size * layer4_size * 64, 100], stddev=0.03), name='wd1')
    bd1 = tf.Variable(tf.truncated_normal([100], stddev=0.01), name='bd1')
    dense_layer1 = tf.matmul(flattened, wd1) + bd1
    dense_layer1 = tf.nn.relu(dense_layer1)
    print(dense_layer1)

    # Regression layer, no activation
    wd2 = tf.Variable(tf.truncated_normal([100, output_size], stddev=0.03), name='wd2')
    bd2 = tf.Variable(tf.truncated_normal([output_size], stddev=0.01), name='bd2')
    dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
    return model_name, dense_layer2

def build_convolution_layers_api(input_x, prefix):
    # Layer 1
    conv1 = tf.layers.conv2d(inputs=input_x,filters=32,kernel_size=3, strides=(1,1), 
        padding="VALID", activation=tf.nn.relu ,name='conv1_'+prefix)
    # conv1 = tf.layers.batch_normalization(conv1, training=True,  scale=True)
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2, padding='VALID', name='pool1_'+prefix)


    # Layer 2
    conv2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=3, strides=(1,1), 
        padding="VALID", activation=tf.nn.relu ,name='conv2_'+prefix)
    conv3 = tf.layers.conv2d(inputs=conv2,filters=64,kernel_size=3, strides=(1,1), 
        padding="VALID", activation=tf.nn.relu ,name='conv3_'+prefix)
    pool2 = tf.layers.max_pooling2d(conv3, 2, 2, padding='VALID', name='pool2_'+prefix)

    # Layer 3
    conv4 = tf.layers.conv2d(inputs=pool2,filters=64,kernel_size=3, strides=(1,1), 
        padding="VALID", activation=tf.nn.relu ,name='conv4_'+prefix)
    conv5 = tf.layers.conv2d(inputs=conv4,filters=64,kernel_size=3, strides=(1,1), 
        padding="VALID", activation=tf.nn.relu ,name='conv5_'+prefix)
    pool3 = tf.layers.max_pooling2d(conv5, 2, 2, padding='VALID', name='pool3_'+prefix)

    # Layer 4
    conv6 = tf.layers.conv2d(inputs=pool3,filters=128,kernel_size=3, strides=(1,1), 
        padding="VALID", activation=tf.nn.relu ,name='conv6_'+prefix)
    conv7 = tf.layers.conv2d(inputs=conv6,filters=128,kernel_size=3, strides=(1,1), 
        padding="VALID", activation=tf.nn.relu ,name='conv7_'+prefix)
    pool4 = tf.layers.max_pooling2d(conv7, 2, 1, padding='VALID', name='pool4_'+prefix)

    # Layer 5
    conv8 = tf.layers.conv2d(inputs=pool4,filters=256,kernel_size=3, strides=(1,1), 
        padding="VALID", activation=tf.nn.relu ,name='conv8_'+prefix)
    
    return conv8

def build_localizer_net(input_x, input_dimension, output_size, is_training, keep_prob=0.8):
    input_multiplier = int(input_x.shape[3]) # this is basically the input channel
    multiplier = 2
    model_name = "localizer_{}".format(multiplier)
    # Layer 1
    with tf.variable_scope("layer1"):
        conv1 = create_new_conv_layer(input_x, 1 * input_multiplier, 16 * multiplier, [3, 3], [1, 1], padding='VALID', activation=None, name='conv')
        conv1 = tf.layers.batch_normalization(conv1, training=is_training)
        conv1 = tf.nn.relu(conv1)
        pool1 = create_new_pooling_layer(conv1, [3,3], [2,2], 'VALID', name='pool')
    
    # Layer 2
    with tf.variable_scope("layer2"):
        conv2 = create_new_conv_layer(pool1, 16 * multiplier, 64 * multiplier, [3, 3], [1, 1], padding='VALID', activation=None, name='conv')
        conv2 = tf.layers.batch_normalization(conv2, training=is_training)
        conv2 = tf.nn.relu(conv2)
        pool2 = create_new_pooling_layer(conv2, [3,3], [2,2], 'VALID', name='pool')

    # Layer 3
    with tf.variable_scope("layer3"):
        conv3 = create_new_conv_layer(pool2, 64 * multiplier, 64 * multiplier, [3, 3], [1, 1], padding='VALID', activation=None, name='conv')
        conv3 = tf.layers.batch_normalization(conv3, training=is_training)
        conv3 = tf.nn.relu(conv3)
        pool3 = create_new_pooling_layer(conv3, [3,3], [2,2], 'VALID', name='pool')
    
    # Layer 4
    with tf.variable_scope("layer4"):
        conv4 = create_new_conv_layer(pool3, 64 * multiplier, 64 * multiplier, [3, 3], [1, 1], padding='VALID', activation=None, name='conv')
        conv4 = tf.layers.batch_normalization(conv4, training=is_training)
        conv4 = tf.nn.relu(conv4)
        pool4 = create_new_pooling_layer(conv4, [3,3], [2,2], 'VALID', name='pool')
    
    # Layer 5
    with tf.variable_scope("layer5"):
        conv5 = create_new_conv_layer(pool4, 64 * multiplier, 64 * multiplier, [3, 3], [1, 1], padding='VALID', activation=None, name='conv')
        conv5 = tf.layers.batch_normalization(conv5, training=is_training)
        conv5 = tf.nn.relu(conv5)
        pool5 = create_new_pooling_layer(conv5, [3,3], [2,2], 'VALID', name='pool')
    
    print("pool 5 shape", pool5.shape)
    flattened = tf.reshape(pool5, [-1, pool5.shape[1] * pool5.shape[2] * 64 * multiplier])

    with tf.variable_scope("dense1"):
        dense1 = tf.layers.dense(flattened, 1024, activation=tf.nn.relu, use_bias=True)
        dense1 = tf.layers.dropout(dense1, rate=1-keep_prob)

    with tf.variable_scope("dense2"):
        # Regression layer, no activation
        dense2 = tf.layers.dense(dense1, output_size, activation=None, use_bias=True)

    return model_name, dense2

'''
    Based on https://blog.deepsense.ai/deep-learning-right-whale-recognition-kaggle/
'''
def build_whale_head_localizer(input_x, input_dimension, output_size, is_training, keep_prob=0.5):
    input_multiplier = int(input_x.shape[3]) # this is basically the input channel
    multiplier = 2
    model_name = "whale_localizer_{}".format(multiplier)
    # Layer 1
    conv1 = create_new_conv_layer(input_x, 1 * input_multiplier, 16 * multiplier, [3, 3], [1, 1], padding='VALID', activation=None, name='conv1')
    conv1 = tf.layers.batch_normalization(conv1, training=is_training)
    conv1 = tf.nn.relu(conv1)
    pool1 = create_new_pooling_layer(conv1, [3,3], [2,2], 'VALID', name='pool1')
    
    # Layer 2
    conv2 = create_new_conv_layer(pool1, 16 * multiplier, 64 * multiplier, [3, 3], [1, 1], padding='VALID', activation=None, name='conv2')
    conv2 = tf.layers.batch_normalization(conv2, training=is_training)
    conv2 = tf.nn.relu(conv2)
    pool2 = create_new_pooling_layer(conv2, [3,3], [2,2], 'VALID', name='pool2')

    # Layer 3
    conv3 = create_new_conv_layer(pool2, 64 * multiplier, 64 * multiplier, [3, 3], [1, 1], padding='VALID', activation=None, name='conv3')
    conv3 = tf.layers.batch_normalization(conv3, training=is_training)
    conv3 = tf.nn.relu(conv3)
    pool3 = create_new_pooling_layer(conv3, [3,3], [2,2], 'VALID', name='pool3')
    
    # Layer 4
    conv4 = create_new_conv_layer(pool3, 64 * multiplier, 64 * multiplier, [3, 3], [1, 1], padding='VALID', activation=None, name='conv4')
    conv4 = tf.layers.batch_normalization(conv4, training=is_training)
    conv4 = tf.nn.relu(conv4)
    pool4 = create_new_pooling_layer(conv4, [3,3], [2,2], 'VALID', name='pool4')
    
    # Layer 5
    conv5 = create_new_conv_layer(pool4, 64 * multiplier, 64 * multiplier, [3, 3], [1, 1], padding='VALID', activation=None, name='conv5')
    conv5 = tf.layers.batch_normalization(conv5, training=is_training)
    conv5 = tf.nn.relu(conv5)
    pool5 = create_new_pooling_layer(conv5, [3,3], [2,2], 'VALID', name='pool5')
    
    print("pool 5 shape", pool5.shape)
    flattened = tf.reshape(pool5, [-1, pool5.shape[1] * pool5.shape[2] * 64 * multiplier])

    # Regression layer, no activation
    dense1 = tf.layers.dense(flattened, output_size, activation=None, use_bias=True)
    return model_name, dense1


def build_double_landmark(img1, img2, output_size):
    model_name = "double_landmark"
    
    pipe1 = build_convolution_layers(img1, 'ED')
    pipe2 = build_convolution_layers(img2, 'ES')

    merged_pipe = tf.concat([pipe1, pipe2], 3)
    print('Merged layers', merged_pipe.shape)

    conv_m1 = create_new_conv_layer(merged_pipe, 256, 512, [3, 3], [1, 1], padding='SAME', activation=tf.nn.relu, name='conv_m1')
    conv_m2 = create_new_conv_layer(conv_m1, 512, 512, [3, 3], [1, 1], padding='VALID', activation=tf.nn.relu, name='conv_m2')
    pool_m1 = create_new_pooling_layer(conv_m2, [2,2], [1,1], 'VALID', name='pool_m1')
    # Layer 5
    conv_m3 = create_new_conv_layer(pool_m1, 512, 256, [3, 3], [1, 1], padding='SAME', activation=tf.nn.relu, name='conv_m3')
    
    # Layer 6
    print("conv_m3 shape", conv_m3.shape)
    # print("conv_m3 shape", conv_m3.shape[1], conv_m3.shape[2], conv_m3.shape[3])
    flattened = tf.reshape(conv_m3, [-1, conv_m3.shape[1] * conv_m3.shape[2] * 256])

    dense1 = tf.layers.dense(flattened, 1024, activation=tf.nn.relu, use_bias=True)
    dense1 = tf.layers.dropout(dense1, rate=0.5)

    dense2 = tf.layers.dense(dense1, 2048, activation=tf.nn.relu, use_bias=True)
    dense2 = tf.layers.dropout(dense2, rate=0.5)

    # Regression layer, no activation
    dense3 = tf.layers.dense(dense2, output_size, activation=None, use_bias=True)
    
    return model_name, dense3

def build_convolution_layers(input_x, prefix):
    # Layer 1
    conv1 = create_new_conv_layer(input_x, 1, 32, [3, 3], [1, 1], padding='VALID', activation=tf.nn.relu, name='conv1_'+prefix)
    pool1 = create_new_pooling_layer(conv1, [2,2], [2,2], 'VALID', name='pool1_'+prefix)
    # Layer 2
    conv2 = create_new_conv_layer(pool1, 32, 64, [3, 3], [1, 1], padding='VALID', activation=tf.nn.relu, name='conv2_'+prefix)
    conv3 = create_new_conv_layer(conv2, 64, 64, [3, 3], [1, 1], padding='VALID', activation=tf.nn.relu, name='conv3_'+prefix)
    pool2 = create_new_pooling_layer(conv3, [2,2], [2,2], 'VALID', name='pool2_'+prefix)
    # Layer 3
    conv4 = create_new_conv_layer(pool2, 64, 64, [3, 3], [1, 1], padding='VALID', activation=tf.nn.relu, name='conv4_'+prefix)
    conv5 = create_new_conv_layer(conv4, 64, 64, [3, 3], [1, 1], padding='VALID', activation=tf.nn.relu, name='conv5_'+prefix)
    pool3 = create_new_pooling_layer(conv5, [2,2], [2,2], 'VALID', name='pool3_'+prefix)
    # Layer 4
    conv6 = create_new_conv_layer(pool3, 64, 128, [3, 3], [1, 1], padding='VALID', activation=tf.nn.relu, name='conv6_'+prefix)
    conv7 = create_new_conv_layer(conv6, 128, 128, [3, 3], [1, 1], padding='VALID', activation=tf.nn.relu, name='conv7_'+prefix)
    pool4 = create_new_pooling_layer(conv7, [2,2], [1,1], 'VALID', name='pool4_'+prefix)
    
    return  pool4

def build_landmark_network(input_x, input_dimension, output_size):
    '''
    Code: https://github.com/yinguobing/cnn-facial-landmark/blob/master/landmark.py
    '''
    model_name = "landmark_network"
    # Layer 1
    conv1 = create_new_conv_layer(input_x, 1, 32, [3, 3], [1, 1], padding='VALID', activation=tf.nn.relu, name='conv1')
    pool1 = create_new_pooling_layer(conv1, [2,2], [2,2], 'VALID', name='pool1')
    # Layer 2
    conv2 = create_new_conv_layer(pool1, 32, 64, [3, 3], [1, 1], padding='VALID', activation=tf.nn.relu, name='conv2')
    conv3 = create_new_conv_layer(conv2, 64, 64, [3, 3], [1, 1], padding='VALID', activation=tf.nn.relu, name='conv3')
    pool2 = create_new_pooling_layer(conv3, [2,2], [2,2], 'VALID', name='pool2')
    # Layer 3
    conv4 = create_new_conv_layer(pool2, 64, 64, [3, 3], [1, 1], padding='VALID', activation=tf.nn.relu, name='conv4')
    conv5 = create_new_conv_layer(conv4, 64, 64, [3, 3], [1, 1], padding='VALID', activation=tf.nn.relu, name='conv5')
    pool3 = create_new_pooling_layer(conv5, [2,2], [2,2], 'VALID', name='pool3')
    # Layer 4
    conv6 = create_new_conv_layer(pool3, 64, 128, [3, 3], [1, 1], padding='VALID', activation=tf.nn.relu, name='conv6')
    conv7 = create_new_conv_layer(conv6, 128, 128, [3, 3], [1, 1], padding='VALID', activation=tf.nn.relu, name='conv7')
    pool4 = create_new_pooling_layer(conv7, [2,2], [1,1], 'VALID', name='pool4')
    # Layer 5
    conv8 = create_new_conv_layer(pool4, 128, 256, [3, 3], [1, 1], padding='VALID', activation=tf.nn.relu, name='conv8')
    # Layer 6
    print("conv8 shape", conv8.shape)
    print("conv8 shape", conv8.shape[1], conv8.shape[2], conv8.shape[3])
    flattened = tf.reshape(conv8, [-1, conv8.shape[1] * conv8.shape[2] * 256])

    #wd1 = tf.Variable(tf.truncated_normal([21 * 21 * 256, 1024], stddev=0.03), name='wd1')
    # wd1 = tf.Variable(tf.truncated_normal([5 * 5 * 256, 1024], stddev=0.03), name='wd1')
    wd1 = tf.Variable(tf.truncated_normal([int(flattened.shape[1]), 1024], stddev=0.03), name='wd1')
    bd1 = tf.Variable(tf.truncated_normal([1024], stddev=0.01), name='bd1')
    dense_layer1 = tf.matmul(flattened, wd1) + bd1
    dense_layer1 = tf.nn.relu(dense_layer1)
    print(dense_layer1)

    # Regression layer, no activation
    wd2 = tf.Variable(tf.truncated_normal([1024, output_size], stddev=0.03), name='wd2')
    bd2 = tf.Variable(tf.truncated_normal([output_size], stddev=0.01), name='bd2')
    dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
    return model_name, dense_layer2

def build_simple_alexnet(x_shaped, input_dimension, output_size): 
    model_name = "simple_alexnet"
    
    # Layer 1
    conv1 = create_new_conv_layer(x_shaped, 1, 96, [11, 11], [4, 4], padding='VALID', activation=tf.nn.relu, name='conv1')
    # 'norm1'    Cross Channel Normalization
    pool1 = create_new_pooling_layer(conv1, [3,3], [2,2], 'VALID', name='pool1')

    # Layer 2
    conv2 = create_new_conv_layer(pool1, 96, 256, [5, 5], [1, 1], padding='SAME', activation=tf.nn.relu, name='conv2')
    # 'norm1'    Cross Channel Normalization
    pool2 = create_new_pooling_layer(conv2, [3,3], [2,2], 'VALID', name='pool2')

    # Layer 3
    conv3 = create_new_conv_layer(pool2, 256, 384, [3, 3], [1, 1], padding='SAME', activation=tf.nn.relu, name='conv3')
    conv4 = create_new_conv_layer(conv3, 384, 384, [3, 3], [1, 1], padding='SAME', activation=tf.nn.relu, name='conv4')
    conv5 = create_new_conv_layer(conv4, 384, 256, [3, 3], [1, 1], padding='SAME', activation=tf.nn.relu, name='conv5')
    pool3 = create_new_pooling_layer(conv5, [3,3], [2,2], 'VALID', name='pool3')

    print("pool3 shape", pool3.shape)
    print("pool3 shape", pool3.shape[1], pool3.shape[2], pool3.shape[3])
    flattened = tf.reshape(pool3, [-1, pool3.shape[1] * pool3.shape[2] * 256])

    # dense 1
    wd1 = tf.Variable(tf.truncated_normal([2 * 2 * 256, 4096], stddev=0.03), name='wd1')
    bd1 = tf.Variable(tf.truncated_normal([4096], stddev=0.01), name='bd1')
    dense_layer1 = tf.matmul(flattened, wd1) + bd1
    dense_layer1 = tf.nn.relu(dense_layer1)
    # dropout here...
    print(dense_layer1)

    # dense 2
    wd2 = tf.Variable(tf.truncated_normal([4096, 1024], stddev=0.03), name='wd2')
    bd2 = tf.Variable(tf.truncated_normal([1024], stddev=0.01), name='bd2')
    dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
    dense_layer2 = tf.nn.relu(dense_layer2)
    # dropout here...
    print(dense_layer2)

    wd3 = tf.Variable(tf.truncated_normal([1024, output_size], stddev=0.03), name='wd3')
    bd3 = tf.Variable(tf.truncated_normal([output_size], stddev=0.01), name='bd3')
    dense_layer3 = tf.matmul(dense_layer2, wd3) + bd3

    return model_name, dense_layer3

def build_alexnet(x_shaped, input_dimension, output_size): 
    model_name = "alexnet"
    
    # Layer 1
    conv1 = tf.layers.conv2d(inputs=x_shaped,filters=96,kernel_size=11, strides=(4,4), padding="VALID", activation=tf.nn.relu ,name='conv1')
    conv1 = tf.layers.batch_normalization(conv1, training=True,  scale=True)
    pool1 = tf.layers.max_pooling2d(conv1, 3, 2, padding='VALID', name='pool1')

    # Layer 2
    conv2 = tf.layers.conv2d(inputs=pool1,filters=256,kernel_size=5, strides=(1,1), padding="SAME", activation=tf.nn.relu ,name='conv2')
    conv2 = tf.layers.batch_normalization(conv2, training=True,  scale=True)
    pool2 = tf.layers.max_pooling2d(conv2, 3, 2, padding='VALID', name='pool2')


    # Layer 3
    conv3 = tf.layers.conv2d(inputs=pool2,filters=384,kernel_size=3, strides=(1,1), padding="SAME", activation=tf.nn.relu ,name='conv3')
    conv4 = tf.layers.conv2d(inputs=conv3,filters=384,kernel_size=3, strides=(1,1), padding="SAME", activation=tf.nn.relu ,name='conv4')
    conv5 = tf.layers.conv2d(inputs=conv4,filters=256,kernel_size=3, strides=(1,1), padding="SAME", activation=tf.nn.relu ,name='conv5')
    pool3 = tf.layers.max_pooling2d(conv5, 3, 2, padding='VALID', name='pool3')

    

    print("pool3 shape", pool3.shape)
    print("pool3 shape", pool3.shape[1], pool3.shape[2], pool3.shape[3])
    flattened = tf.reshape(pool3, [-1, pool3.shape[1] * pool3.shape[2] * 256])
    
    # dense 1
    dense1 = tf.layers.dense(flattened, 4096, activation=tf.nn.relu, use_bias=True)
    dense1 = tf.layers.dropout(dense1, rate=0.5)
    print('dense1', dense1.shape)

    # dense 2
    dense2 = tf.layers.dense(dense1, 4096, activation=tf.nn.relu, use_bias=True)
    dense2 = tf.layers.dropout(dense2, rate=0.5)

    # dense 3
    dense3 = tf.layers.dense(dense2, 1024, activation=tf.nn.relu, use_bias=True)
    
    # output layer
    dense4 = tf.layers.dense(dense3, output_size, activation=None, use_bias=True)
    print('dense4', dense4.shape)
    return model_name, dense4