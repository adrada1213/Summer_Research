import os
# This is so dumb, python has problem importing sibling directory
# So we kinda have to do hacky stuff to 'include' the directory as the main running one..really dumb
import sys
sys.path.append(os.path.abspath("src"))
import packages.neural_network_builder as nn
import packages.mapper_functions as mf
import packages.dataset_mapper as dm

from random import randrange
import math
import tensorflow as tf
import datetime
import time
import h5py
import numpy as np
import tensorflow.contrib.slim as slim


# Read the file
# filepath = '/mnt/cube/edward-playground/ukb_tagging/data_sequence'
#filepath = 'E:\\MR-tagging\\dataset-RNNCNN\\data_sequence_new'
filepath = 'E:\\cine-machine-learning\\dataset'

model_name = "rnncnn_testdata"

use_random_split = True
test_ratio = 0.2 # test ratio is a fraction of the whole data
train_ratio = 0.8 # train ratio is a fraction of the (whole_data - test_data)

# Image and output size
img_size = 128
output_size = 2 * 168

# Hyperparameters optimisation variables
initial_learning_rate = 1e-4
epochs = 60
batch_size = 30
val_batch_size = 100
initial_magnitude = 5.
training_keep_prob = 0.8

# RNN parameters
# use_normalized_data = True
# rnn_nodes = [1024,1024]
rnn_nodes = 1024
state_size = 1024
time_steps = 2
trunc_time_steps = 2
#time_steps = 20
#trunc_time_steps = 20


alpha = 0.1
def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=alpha)

'''
    retrieve the number of variables used by the network and print it
'''
def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    

# --- Wrapper functions for the tf.dataset input pipeline ---
def _training_data_load_wrapper(data_path, idx, es_idx):
    return tf.py_func(func=mf._load_hd5_img_and_coords_sequence, 
        inp=[fpath, group, idx, es_idx], Tout=[tf.float32, tf.float32])
    
def _resize_frames(imgs, coords):

    print(imgs)
    img_seq = tf.reshape(imgs, [trunc_time_steps, img_size, img_size, 1])
    print(img_seq)
    coords_seq = tf.reshape(coords, [trunc_time_steps, output_size])

    return img_seq, coords_seq

def _pick_truncated_frames(imgs, coords):

    img_seq = tf.reshape(imgs, [time_steps,  img_size, img_size])
    coords_seq = tf.reshape(coords, [time_steps, output_size])

    return tf.py_func(func=_pick_random_sequence_frames, 
        inp=[img_seq, coords_seq], Tout=[tf.float32, tf.float32, tf.int64])

def _pick_random_sequence_frames(img_seq, coords_seq):
    # we gonna pick a random single img
    rnd_start = randrange(0,time_steps - trunc_time_steps + 1) 
    # print('picked time', rnd)

    img_frames = img_seq[rnd_start:rnd_start+trunc_time_steps]
    coords_frames = coords_seq[rnd_start:rnd_start+trunc_time_steps]
    # print(img.shape)
    
    return img_frames.astype('float32'), coords_frames.astype('float32'), rnd_start
    # return img_seq, coords_seq
    
# --- end of wrapper functions ---

def create_summary(model_name, name, value):
    tf.summary.scalar('{}/{}'.format(model_name,name), value)

def create_summary_value(model_name, name, value):
    summary.value.add(tag='{}/{}'.format(model_name, name), simple_value=value)


'''
    Input pipeline
    This function accepts a list of filenames with index to read
    The _training_data_load_wrapper will read the filename-index pair and load the data
'''
def initialize_dataset(filepaths, groups, indices, training=False):
    ds = tf.data.Dataset.from_tensor_slices((filepaths, groups, indices, es_indices))
    
    if training:
        ds = ds.shuffle(5000)

    # Run the mapping functions on each data
    ds = ds.map(_training_data_load_wrapper, num_parallel_calls=4)

    # TODO: We've tried to play around with 2,5,7, and 10 truncated frames before including all 20 frames
    # ds = ds.map(_pick_truncated_frames)
    ds = ds.map(_resize_frames)

    # TODO: add augmentation here (e.g. flip, translation, normalize, contrast, zoom, etc)

    if training:
        ds = ds.batch(batch_size=batch_size).prefetch(10).repeat()
    else:
        # ds = ds.batch(batch_size=len(indexes)).repeat()
        ds = ds.batch(batch_size=val_batch_size).repeat()
        
    it = ds.make_one_shot_iterator()
    next_element = it.get_next()
    return next_element

def initialize_dataset_from_map(dataset: dm.DataMappingSet, training=False):
    return initialize_dataset(dataset.filepaths, dataset.groups, dataset.indices, dataset.es_indices, training)


# --------------------------- CNN cells --------------------------- 
# Spatial feature extraction, shared weight using tf.AUTO_REUSE 
def build_cnn_cell(input_layer):
    with tf.variable_scope('cnn_cell', reuse = tf.AUTO_REUSE ):
        # Layer 1
        layer1 = tf.layers.conv2d(inputs=input_layer,filters=32,kernel_size=3, strides=(1,1), padding="VALID", activation=leaky_relu ,name='conv1')
        pool1 = tf.layers.max_pooling2d(layer1, 2, 2, padding='VALID', name='pool1')
        
        # Layer 2
        layer2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=3, strides=(1,1), padding="VALID", activation=leaky_relu ,name='conv2')
        pool2 = tf.layers.max_pooling2d(layer2, 2, 2, padding='VALID', name='pool2')
        
        # Layer 3
        layer3 = tf.layers.conv2d(inputs=pool2,filters=64,kernel_size=3, strides=(1,1), padding="VALID", activation=leaky_relu ,name='conv4')
        layer3 = tf.layers.max_pooling2d(layer3, 2, 2, padding='VALID', name='pool3')

        # Layer 5
        layer5 = tf.layers.conv2d(inputs=layer3,filters=128,kernel_size=3, strides=(1,1), padding="SAME", activation=leaky_relu ,name='conv8')
        # transpose flatten just like YOLO
        dim = layer5.shape[1] * layer5.shape[2] * layer5.shape[3]
        layer5_transposed = tf.transpose(layer5,(0,3,1,2))
        flattened = tf.reshape(layer5_transposed, [-1, 1, dim])

        # flattened = tf.reshape(layer5, [-1, 1, layer5.shape[1] * layer5.shape[2] * layer5.shape[3]])
        # print('flat',flattened.shape)
        
        dense1 = tf.layers.dense(flattened, 1024, activation=leaky_relu, use_bias=True)
        # TODO: maybe add dropout here, but for now don't, we want all the features to go to RNN    
        # dense1 = tf.layers.dropout(dense1, rate=1-keep_prob)

        return dense1

# --------------------------- Loss function --------------------------- 
def get_midwall_cc_strains(flattened_coords):
    epsilon = 0.0001
    # reshape to batch x time x 2 x 168 points (24 radials with 7 points)
    coords_batch = tf.reshape(flattened_coords, [-1, trunc_time_steps, 2, 168])
    midwall_points = coords_batch[:,:,:, 3::7]  # get point index 3 for every radial

    # we will have to calculate the strain between every points
    points_arr = tf.unstack(midwall_points, axis=3)

    # strain formula: ((l^2/L^2)-1) / 2  --> l^2 = x^2 + y^2
    # with x and y is the difference between x and y coords of 2 points
    ccs = []
    # the cc strain is circular, so we going through all of them and back to point 0
    for r in range(0,len(points_arr)):
        # for the last point, calculate between point_r and point_0
        if r+1 == len(points_arr):
            cc_diff = tf.square(points_arr[r] - points_arr[0])
        else:
            cc_diff = tf.square(points_arr[r] - points_arr[r+1])

        # do the sum: x^2 + y^2
        cc_sum = cc_diff[:,:,0] + cc_diff[:,:,1]

        # we need to unstack it first
        ccsum_arr = tf.unstack(cc_sum, axis=1)

        res_arr = []
        # we are going to do the l^2 / L^2 here for every frame
        for t in range(0,len(ccsum_arr)):
            # divide by time 0
            if t == 0:
                cc_divv = (ccsum_arr[t] + epsilon) / (cc_sum[:,0] + epsilon)                 
            else:
                cc_divv = ccsum_arr[t] / (cc_sum[:,0] + epsilon)

            # do the strain formula
            res = (cc_divv - 1) / 2
            res_arr.append(res)
        partial_cc = tf.stack(res_arr, axis=1)
        # put the partial_cc in every time frame back together
        ccs.append(partial_cc)
    # stack the partial_cc for every links together
    stacked_ccs = tf.stack(ccs, axis=2)

    # calculate the mean cc for every time frame
    mid_cc = tf.reduce_mean(stacked_ccs, axis=2)
    return mid_cc

def get_radial_strains(flattened_coords):
    epsilon = 0.0001
    # reshape to batch x time x 2 x 168 points (24 radials with 7 points)
    coords_batch = tf.reshape(flattened_coords, [-1, trunc_time_steps, 2, 168])
    endo_batch = coords_batch[:,:,:, ::7]
    epi_batch = coords_batch[:,:,:, 6::7]

    diff = tf.square(epi_batch - endo_batch)
    # print('diff', diff.shape)
    summ = diff[:,:,0,:] + diff[:,:,1,:] # x2 + y2
    # print('summ', summ.shape)

    # divv = tf.divide(summ, summ[:,0,:])
    summ_arr = tf.unstack(summ, axis=1)
    summ_ed = summ[:,0,:] + epsilon

    res = []
    for t in range(0,len(summ_arr)):
        if t == 0:
            summ_t = summ_arr[t] + epsilon  
        else:
            summ_t = summ_arr[t]
        # summ_t = summ_arr[t] + epsilon  
        res.append(summ_t / summ_ed)
        # res.append(s / summ_ed)

    divv = tf.stack(res, axis=1)
    # print('summ_arr', summ_arr.shape)
    
    # print('divv', divv.shape)
    res = (divv - 1) / 2
    res = tf.reduce_mean(res, axis=2)
    return res
# --------------------------- /end loss function --------------------------- 


# -- Data input prep --
print('\n-------------------------------------------')
# Cause we have multiple hd5 file, we need to create a mapping between filepath, and index, so the tf.Dataset can read it easily later
'''
if use_random_split:
    print('Use random split on whole dataset')
    dataset_dict, test_dict = dm.split_dataset_randomly(filepath, input_patterns, testdata_mapfile, test_ratio, train_ratio)
    train_set = dataset_dict['train']
    validation_set = dataset_dict['validate']

else:
    print('Use input and validation patterns')
    train_files, train_patients, train_indexes = dm.build_filemap(filepath, input_patterns)
    validate_files, validate_patients, validate_indexes = dm.build_filemap(filepath, val_patterns)

    train_set = dm.DataMappingSet(train_patients, train_files, train_indexes)
    validation_set = dm.DataMappingSet(validate_patients, validate_files, validate_indexes)
    dataset_dict = {'train': train_set, 'validate': validation_set}
'''

dataset_dict, test_dict = dm.load_all_datasets(filepath, uk_biobank_h5_file)
train_set = dataset_dict["train"]
validation_set = dataset_dict["validate"]

total_batch = int(train_set.count() / batch_size)
validation_batch = math.ceil(validation_set.count() / val_batch_size)

print('Training batch', total_batch)
print('Validation batch', validation_batch)
print('-------------------------------------------\n')

# ----------------- TensorFlow stuff -------------------
# Reset all the tensor variables
tf.reset_default_graph()  # We need to do this here before creating any tensor -> Yep, Dataset is a tensor object

# Dataset initialisation (Training data)
next_element = initialize_dataset_from_map(train_set, training=True)
next_validation = initialize_dataset_from_map(validation_set, training=False)


learning_rate = tf.Variable( initial_value = initial_learning_rate, trainable = False, name = 'learning_rate' ) 
magnitude = tf.Variable( initial_value = initial_magnitude, trainable = False, name = 'magnitude' ) 
adjust_learning_rate = tf.assign( learning_rate, learning_rate / np.sqrt( 2 ) )
adjust_magnitude = tf.assign( magnitude, 10. )

# -- Placeholders --
# declare the training data placeholders
# batch size, trunc_time_steps, img_width x img_size
x = tf.placeholder(tf.float32, [None, None, img_size, img_size], name='x')
x = tf.expand_dims(x,4)

y = tf.placeholder(tf.float32, [None, None, output_size], name='y')
init_state = tf.placeholder(tf.float32, [2, None, state_size], name='init_state')
keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')
# init_state = tf.placeholder(tf.float32, [2, 2, None, state_size], name='init_state')

# -- Model name --
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
model_dir = "./models/{}_{}".format(model_name, timestamp)
# Do not use .ckpt on the model_path
model_path = "{}/{}".format(model_dir, model_name)
training_model_path = "{}/train/{}".format(model_dir, model_name)

# ------- Main Network ------
print('\nBuilding network model...')
flat_stack = []
# CNN part
for t in range(0, trunc_time_steps):
    # this is why it is NOT dynamic, we have to loop through the frames
    # TODO: find a way to represent the feature extraction in a dynamic way
    img = x[:,t,:,:, :]
    features = build_cnn_cell(img)
    flat_stack.append(features)
cnn_outputs = tf.concat(flat_stack,axis=1)

# batches x time_steps x embedding
print('cnn_outputs',cnn_outputs.shape)

# RNN units
# basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=rnn_nodes)
basic_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_nodes)
basic_cell = tf.nn.rnn_cell.DropoutWrapper(basic_cell, output_keep_prob=keep_prob)
rnn_outputs, states = tf.nn.dynamic_rnn(cell=basic_cell, inputs=cnn_outputs, dtype=tf.float32, time_major=False)

# def make_lstm_cell(lstm_size, keep_prob):
#     lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
#     drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
#     return drop

# with tf.name_scope('lstm'):
#      multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([make_lstm_cell(node, 0.5) for node in rnn_nodes])
# rnn_outputs, states = tf.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=cnn_outputs, dtype=tf.float32, time_major=False)


rnn_outputs = tf.identity(rnn_outputs, name="rnn_outputs")
states = tf.identity(states, name="states")

# put some dense layers here ...
dense1 = tf.layers.dense(rnn_outputs, 1024, activation=tf.nn.relu, use_bias=True)
dense1 = tf.layers.dropout(dense1, rate=1-keep_prob)

y_ = tf.layers.dense(dense1,units=output_size,activation=None, use_bias=True)

print(output_size)
# Name output layer as tensor Variable so we can restore it easily
cnn_outputs = tf.identity(cnn_outputs, name="cnn_outputs")

y_ = tf.identity(y_, name="y_")
print(y_)
print(y)

# Loss and training function
reshaped_y = tf.reshape(y, [-1, trunc_time_steps * output_size])
reshaped_y_ = tf.reshape(y_, [-1, trunc_time_steps  * output_size])
print(reshaped_y.shape)

# we need to calculate the loss across all the predictions being flattened
loss_old = tf.losses.mean_squared_error(labels=reshaped_y, predictions=reshaped_y_)
loss_old = tf.identity(loss_old, name="loss_old")

print('Modelling radial strain error...')
rr_diff = get_radial_strains(y_) - get_radial_strains(y)
rr_diff = tf.abs(rr_diff) * magnitude

print('Modelling midwall circumferential strain error...')
cc_diff = get_midwall_cc_strains(y_) - get_midwall_cc_strains(y)
cc_diff = tf.abs(cc_diff) * magnitude

# print('reg term magnitude:', magnitude)

sqrdiff2 = tf.square(y_ - y)
sqrdiff2 = tf.reduce_mean(sqrdiff2, axis=2)
print('RR error', rr_diff.shape)
print('CC error', cc_diff.shape)
print('MSE', sqrdiff2.shape)

# Combined loss function
loss2 = sqrdiff2 + rr_diff + cc_diff
# loss2 = sqrdiff2
loss2 = tf.reduce_mean(loss2)
loss2 = tf.identity(loss2, name="loss")

# Average them so we can plot it in a tensorboard graph 
rr_strain_err = tf.reduce_mean(rr_diff)
cc_strain_err = tf.reduce_mean(cc_diff)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss2, name='optimizer')

# Tensorboard stuff
unique_model_name = '{}_{}'.format(model_name, timestamp)
create_summary(unique_model_name, 'learning_rate', learning_rate)

# print the parameters
model_summary()

# ------------ setup the initialisation operator -------------------
init_op = tf.global_variables_initializer()

previous_low = 1000 # just a random number for initial cost
prev_train_low = 1000

# ----- Run the training -----
with tf.Session() as sess:
    # initialise the variables
    print("Initializing session...")

    sess.run(init_op)
    saver = tf.train.Saver()

    # summary - Tensorboard stuff
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(model_dir+'/tensorboard/train', sess.graph)
    val_writer = tf.summary.FileWriter(model_dir+'/tensorboard/validation')
    '''
    # save data mapping
    if use_random_split:
        if not os.path.isfile(os.path.join(filepath, testdata_mapfile)):
            # If there was no test data mapping file yet, we definitely have to save this
            print('\nSaving test cases into:', filepath, testdata_mapfile)
            dm.save_dataset_mapping(filepath, test_dict, testdata_mapfile)

    # In any case, always save the whole splitted cases in the model directory
    print('\nSaving all case split into:', model_dir)
    dm.save_dataset_mapping(model_dir, dataset_dict)
    '''
    print("Starting the training for landmarks RNN at {}".format(time.ctime()))
    start_time = time.time()
    
    _current_state = np.zeros((2, batch_size, state_size))
    # _current_state = np.zeros((2, 2, batch_size, state_size))
    print('Setting init state to 0')

    for epoch in range(epochs):
        # ------------------------------- Training -------------------------------
        start_loop = time.time()
        print("\nEpoch", (epoch+1), time.ctime())

        # Reduce learning rate every few steps
        if epoch >= 10 and epoch % 10 == 0:
            adjust_learning_rate.eval()
            print('Learning rate adjusted to', sess.run(learning_rate))
        
        # if epoch == 30:
        #     adjust_magnitude.eval()
        #     print('Magnitude adjusted to', sess.run(magnitude))

        avg_cost = 0
        avg_s_err = 0
        avg_sc_err = 0
        avg_old = 0

        for i in range(total_batch):
            next_batch = sess.run(next_element)
            batch_x = next_batch[0]
            batch_y = next_batch[1]
            # batch_t = next_batch[2]

            _, summ, c, c_old, train_s_err, train_sc_err, cnn_features, _current_state, pred = sess.run([optimizer, merged, loss2, loss_old, rr_strain_err, cc_strain_err, cnn_outputs, states, y_], feed_dict={x: batch_x, y: batch_y, init_state: _current_state, keep_prob: training_keep_prob})
            # _, summ, c, cnn_features, pred = sess.run([optimizer, merged, loss2, cnn_outputs, y_], feed_dict={x: batch_x, y: batch_y})

            avg_cost += c / total_batch
            avg_old += c_old / total_batch
            avg_s_err += train_s_err / total_batch
            avg_sc_err += train_sc_err / total_batch

        print(batch_y.shape)
        print(pred.shape)

        # we would like to see the cnn outputs
        # print('CNN features', cnn_features.shape)

        # for p in range(0,2):
        #     for t in range(0,2):    
        #         # print('pat',p,' feature',t,batch_x[p][t][0:5])
        #         print('pat',p,' feature',t,cnn_features[p][t][0:5])
                
        for p in range(0,2):
            for t in range(0,2):
                # print('pat',p,'t',batch_t[p]+t,'y_ vs y:',pred[p][t][0:3], batch_y[p][t][0:3])
                print('pat',p,'t',t,'y_ vs y:',pred[p][t][0:3], batch_y[p][t][0:3])
        print('_current_state', _current_state[0][0][0:5])
        print('--\n')
        end_loop = time.time()
        # print("Time taken for epoch {}: {:.2f} seconds".format(epoch+1, end_loop-start_loop))
        
        # add additional summary for the average loss
        summary = tf.Summary()
        summary.value.add(tag='{}/Avg_loss'.format(unique_model_name), simple_value=avg_cost)
        summary.value.add(tag='{}/Lambda_RR_strain_error'.format(unique_model_name), simple_value=avg_s_err)
        summary.value.add(tag='{}/Lambda_CC_strain_error'.format(unique_model_name), simple_value=avg_sc_err)
        summary.value.add(tag='{}/RR_Strain_error'.format(unique_model_name), simple_value=avg_s_err/ sess.run(magnitude))
        summary.value.add(tag='{}/CC_Strain_error'.format(unique_model_name), simple_value=avg_sc_err/ sess.run(magnitude))
        train_writer.add_summary(summary, epoch)
        
        train_writer.add_summary(summ, epoch)

        train_msg = "{} Training   - Loss: {:.3f}, Loss (MSE): {:.3f}, RR strain_err: {:.3f}, CC strain_err: {:.3f}, training time : {:.2f} seconds".format(epoch+1, avg_cost, avg_old, avg_s_err, avg_sc_err, end_loop-start_loop)
        print(train_msg)


        # ------------------------------- Validation -------------------------------
        # Run on validation set
        # print('Validating...')
        val_time = time.time()
        validation_loss = 0
        vold =0
        val_s_err = 0
        val_sc_err = 0

        for j in range(validation_batch):
            next_val_batch = sess.run(next_validation)
            X_validate = next_val_batch[0]
            Y_validate = next_val_batch[1]
            
            # print(X_validate.shape)
            # print(Y_validate.shape)
            # feed = feed_dict={x: X_validate, y: Y_validate, init_state: _current_state, keep_prob: 1.0}
            feed = feed_dict={x: X_validate, y: Y_validate, init_state: _current_state }
            vloss, vloss_old, v_s_err, v_sc_err, summ = sess.run([loss2, loss_old, rr_strain_err, cc_strain_err, merged], feed_dict=feed)
            # print(vloss)
            # multiply with number of items in a batch, divide by total validation items
            validation_loss += vloss * len(X_validate) / validation_set.count()
            vold += vloss_old * len(X_validate) / validation_set.count()
            val_s_err += v_s_err * len(X_validate) / validation_set.count()
            val_sc_err += v_sc_err * len(X_validate) / validation_set.count()

        # Verbose message
        val_msg =   "{} Validation - Loss: {:.3f}, Loss (MSE): {:.3f}, RR strain_err: {:.3f}, CC strain_err: {:.3f},validation time: {:.2f} seconds".format(epoch+1, validation_loss, vold, val_s_err, val_sc_err, time.time()-val_time)
        print(val_msg)
        
        # add additional summary for the average loss
        summary = tf.Summary()
        summary.value.add(tag='{}/Avg_loss'.format(unique_model_name), simple_value=validation_loss)
        summary.value.add(tag='{}/Lambda_RR_strain_error'.format(unique_model_name), simple_value=val_s_err)
        summary.value.add(tag='{}/Lambda_CC_strain_error'.format(unique_model_name), simple_value=val_sc_err)
        summary.value.add(tag='{}/RR_Strain_error'.format(unique_model_name), simple_value=val_s_err/ sess.run(magnitude))
        summary.value.add(tag='{}/CC_Strain_error'.format(unique_model_name), simple_value=val_sc_err/ sess.run(magnitude))
        val_writer.add_summary(summary, epoch)

        val_writer.add_summary(summ, epoch)

        # ------------------------------- Save the weights -------------------------------
        if validation_loss < previous_low:
            # Save model weights to disk whenever the validation loss reaches a new low
            save_path = saver.save(sess, model_path)
            print("Model saved in file: %s" % save_path)
            # Update the cost for saving purposes
            previous_low = validation_loss
            
            # write a log
            with open(os.path.join(model_dir, 'savelog.txt'), 'a') as f:
                msg = 'Epoch {},  Cost {:.3f}, validation loss {:.3f}\n'.format((epoch+1), avg_cost, validation_loss)
                f.write(msg)
    # /END of epoch loop

    print("\nTraining RNN complete!")
    print("Total time taken for training: ", (time.time() - start_time), " seconds.")
    print("Finished at ", time.ctime())