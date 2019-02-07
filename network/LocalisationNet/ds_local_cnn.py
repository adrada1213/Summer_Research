import os
# This is so dumb, python has problem importing sibling directory
# So we kinda have to do hacky stuff to 'include' the directory as the main running one..really dumb
import sys
sys.path.append(os.path.abspath("src"))
# print(sys.path)
import packages.neural_network_builder as nn
import packages.mapper_functions as mf
import packages.dataset_mapper as dm

import tensorflow as tf
import tensorflow.contrib.slim as slim

import datetime
import math
import time
import h5py
import numpy as np

	
# Read the file
# filepath = '/mnt/cube/edward-playground/ukb_tagging/data_sequence_original'
filepath = 'E:\\cine-machine-learning\\dataset'

# training_file = 'dataset-train.noresize.h5'
# input_patterns = ['CIM_DATA_EL1','CIM_DATA_EL2']
# val_patterns = ['CIM_D']
# validation_file = 'dataset-train.noresize.h5'

uk_biobank_h5_file = "UK_Biobank.h5"
use_random_split = True

# Image and output size
img_size = 256
output_size = 3
bbox_adjustment = 0.3

# Hyperparameters optimisation variables
network_name = 'local'
initial_learning_rate = 1e-3
epochs = 100
batch_size = 50
val_batch_size = 100
training_keep_prob = 0.8

early_stop_threshold = 15 # stop if there is no loss decrease in a certain number of epoch
previous_acc = 0 # just a random number for initial cost

'''
    retrieve the number of variables used by the network and print it
'''
def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

# --- Wrapper functions for the tf.dataset input pipeline ---
def _training_data_load_wrapper(fpath, group, idx):
    # print('wrapper', idx)
    return tf.py_func(func=mf._load_hd5_img_and_centroid_from_sequence, 
        inp=[fpath, group, idx], Tout=[tf.float32, tf.float32])

def _augmentation_wrapper(img, coords):
    return tf.py_func(func=mf._rotate_img_and_centroid, 
        inp=[img, coords], Tout=[tf.float32, tf.float32])

def _resize_function(img, coords):
    img = tf.reshape(img, [256, 256])
    coords = tf.reshape(coords, [3])
    return img, coords
# --- end of wrapper functions ---

def create_summary(model_name, name, value):
    tf.summary.scalar('{}/{}'.format(model_name,name), value)

def bbox_iou_corner_xy(centroids1, centroids2):
    """
    Calculate Accuracy (IoU)

    Args:
        bboxes1: shape (total_bboxes1, 4)
            with x1, y1, x2, y2 point order.
        bboxes2: shape (total_bboxes2, 4)
            with x1, y1, x2, y2 point order.

        p1 *-----
           |     |
           |_____* p2

    Returns:
        Tensor with shape (total_bboxes1, total_bboxes2)
        with the IoU (intersection over union) of bboxes1[i] and bboxes2[j]
        in [i, j].
    """
    epsilon = 0.0001
    centre_x1, centre_y1, edge_lengths1 = tf.split(centroids1, 3, axis=1)
    centre_x2, centre_y2, edge_lengths2 = tf.split(centroids2, 3, axis=1)

    x11 = tf.add_n([centre_x1, -edge_lengths1])
    y11 = tf.add_n([centre_y1, -edge_lengths1])
    x12 = tf.add_n([centre_x1, edge_lengths1])
    y12 = tf.add_n([centre_y1, edge_lengths1])

    x21 = tf.add_n([centre_x2, -edge_lengths2])
    y21 = tf.add_n([centre_y2, -edge_lengths2])
    x22 = tf.add_n([centre_x2, edge_lengths2])
    y22 = tf.add_n([centre_y2, edge_lengths2])

    xI1 = tf.maximum(x11, x21)
    xI2 = tf.minimum(x12, x22)

    yI1 = tf.maximum(y11, y21)
    yI2 = tf.minimum(y12, y22)

    #inter_area = (xI1 - xI2) * (yI1 -yI2)

    inter_area = (xI2 - xI1) * (yI2 -yI1)

    bboxes1_area = (x12 - x11) * (y12 - y11)
    bboxes2_area = (x22 - x21) * (y22 - y21)

    union = (bboxes1_area + bboxes2_area) - inter_area

    iou = (inter_area+epsilon) / (union+epsilon)
    print('iou',iou)
    return tf.reduce_mean(iou)

'''
    Input pipeline
    This function accepts a list of filenames with index to read
    The _training_data_load_wrapper will read the filename-index pair and load the data
'''
def initialize_dataset(filepaths, groups, indices, training=False):
    
    ds = tf.data.Dataset.from_tensor_slices((filepaths, groups, indices))

    if training:
        ds = ds.shuffle(1000)

    # Run the mapping functions on each data
    ds = ds.map(_training_data_load_wrapper, num_parallel_calls=4)
    
    # ds = ds.map(_adjust_bbox_wrapper) # we don't do this anymore, the bbox in the dataset has been adjusted during prepare_data

    if training:
        ds = ds.map(_augmentation_wrapper)
        # TODO: add more augmentation here (e.g. flip, translation, normalize, contrast, zoom, etc)

    ds = ds.map(_resize_function)

    if training:
        ds = ds.batch(batch_size=batch_size).prefetch(10).repeat()
    else:
        # ds = ds.batch(batch_size=len(indexes)).repeat()
        ds = ds.batch(batch_size=val_batch_size).repeat()
        
    it = ds.make_one_shot_iterator()
    next_element = it.get_next()
    return next_element

def initialize_dataset_from_map(dataset: dm.DataMappingSet, training):
    return initialize_dataset(dataset.filepaths, dataset.groups, dataset.indices, training)

# -- Data input prep --
print('\n-------------------------------------------')


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
adjust_learning_rate = tf.assign( learning_rate, learning_rate / np.sqrt( 2 ) )

# -- Placeholders --
# raw input and label shape, we gonna reshape it later
x1 = tf.placeholder(tf.float32, [None, img_size, img_size], name='x1')
y = tf.placeholder(tf.float32, [None, output_size], name='y')
is_training = tf.placeholder(tf.bool, name='is_training')
keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')

# dynamically reshape the input (training sample, height, weight, channel)
x_shaped1 = tf.reshape(x1, [-1, img_size, img_size, 1])
print('Input shape:', x_shaped1)

# -- Main network --
model_name, y_ = nn.build_network(network_name, x_shaped1, img_size, output_size, is_training=True, keep_prob=training_keep_prob)
# Name output layer as tensor Variable so we can restore it easily
y_ = tf.identity(y_, name="y_")

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
model_dir = "./models/{}_{}".format(model_name, timestamp)
# Do not use .ckpt on the model_path
model_path = "{}/{}".format(model_dir, model_name)
training_model_path = "{}/train/{}".format(model_dir, model_name)


# Loss and training function
iou = bbox_iou_corner_xy(centroids1=y, centroids2=y_)
iou = tf.identity(iou, name="iou")

loss = tf.losses.mean_squared_error(labels=y, predictions=y_)
loss = tf.identity(loss, name="loss")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer')

# !!! By default the update ops are placed in tf.GraphKeys.UPDATE_OPS, so they need to be added as a dependency to the train_op
# refer to: https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss, name='train_op')

# Tensorboard stuff
unique_model_name = '{}_{}'.format(model_name, timestamp)
create_summary(unique_model_name, 'loss', loss)
create_summary(unique_model_name, 'IoU', iou)
create_summary(unique_model_name, 'learning_rate', learning_rate)

# print the parameters
model_summary()

# setup the initialisation operator
init_op = tf.global_variables_initializer()

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
    # total_batch = int(len(all_files) / batch_size)
    # print('Total batch', total_batch)
    print("Starting the training for bounding_box corners at {}".format(time.ctime()))
    start_time = time.time()

    last_saved_epoch = 0
    for epoch in range(epochs):
        # ------------------------------- Training -------------------------------
        start_loop = time.time()
        print("\nEpoch", (epoch+1), time.ctime())
        
        # Reduce learning rate every few steps
        if epoch >= 10 and epoch % 5 == 0:
            adjust_learning_rate.eval()
            print('Learning rate adjusted to', sess.run(learning_rate))

        avg_cost = 0
        avg_acc = 0
        # loop through the training set
        for i in range(total_batch):
            # Get input and label batch
            next_batch = sess.run(next_element)
            batch_x1 = next_batch[0]
            batch_y = next_batch[1]
            
            # Feed the network and optimize
            _, summ, c, acc = sess.run([train_op, merged, loss, iou], feed_dict={x1: batch_x1, y: batch_y, is_training: True, keep_prob: training_keep_prob})
            avg_cost += c / total_batch
            avg_acc += acc / total_batch

        end_loop = time.time()
        train_msg = "{} Training   - Loss: {:.3f}, IoU: {:.3f}, training time  : {:.2f} seconds".format(epoch+1, avg_cost, avg_acc, end_loop-start_loop)
        print(train_msg)

        # add additional summary for the average loss
        summary = tf.Summary()
        summary.value.add(tag='{}/Avg_loss'.format(unique_model_name), simple_value=avg_cost)
        summary.value.add(tag='{}/Accuracy'.format(unique_model_name), simple_value=avg_acc)
        train_writer.add_summary(summary, epoch)

        train_writer.add_summary(summ, epoch)
        
        # ------------------------------- Validation -------------------------------
        # Run on validation set
        val_time = time.time()
        validation_loss = 0
        validation_acc = 0
        # loop through the validation set
        for j in range(validation_batch):
            # Get input and label batch
            next_val_batch = sess.run(next_validation)
            X_validate = next_val_batch[0]
            Y_validate = next_val_batch[1]
            
            # Feed it to the network
            feed = feed_dict={x1: X_validate, y: Y_validate, is_training: False, keep_prob: 1}
            summ, v_loss, v_acc = sess.run([merged, loss, iou], feed_dict=feed)
            
            # multiply with number of items in a batch, divide by total validation items
            validation_loss += v_loss * len(X_validate) / validation_set.count()
            validation_acc += v_acc * len(X_validate) / validation_set.count()

        # Verbose message
        val_msg =   "{} Validation - Loss: {:.3f}, IoU: {:.3f}, validation time: {:.2f} seconds".format(epoch+1, validation_loss, validation_acc, time.time()-val_time)
        print(val_msg)

        summary = tf.Summary()
        summary.value.add(tag='{}/Avg_loss'.format(unique_model_name), simple_value=validation_loss)
        summary.value.add(tag='{}/Accuracy'.format(unique_model_name), simple_value=validation_acc)
        val_writer.add_summary(summary, epoch)

        val_writer.add_summary(summ, epoch)

        # ------------------------------- Save the weights -------------------------------
        if validation_acc > previous_acc:
            # Save model weights to disk whenever the validation acc reaches a new high
            save_path = saver.save(sess, model_path)
            print("Model saved in file: %s" % save_path)
            
            # Update the cost for saving purposes
            last_saved_epoch = epoch
            previous_acc = validation_acc
            
            # write to log
            with open(os.path.join(model_dir, 'savelog.txt'), 'a') as f:
                msg = 'Epoch {},  Loss (train/val): {:.3f} / {:.3f}, Acc (train/val): {:.3f} / {:.3f}\n'.format((epoch+1), avg_cost, validation_loss, avg_acc, validation_acc)
                f.write(msg)
        else:
            if last_saved_epoch > 0: # only do this if we ever saved before
                if ((epoch - last_saved_epoch) >= early_stop_threshold):
                    msg = '\nEpoch {} - Early stopping, no more validation loss decrease after {} epochs'.format(epoch+1, early_stop_threshold)
                    print(msg)
                    with open(os.path.join(model_dir, 'savelog.txt'), 'a') as f:
                        f.write('{}\n'.format(msg))
                    # stop the training
                    break
    # /END of epoch loop

    print("\nTraining for bounding box complete!")
    print("Total time taken for training: ", (time.time() - start_time), " seconds.")
    print("Finished at ", time.ctime())

    os.system("predict_localisation_bbox.py")

    os.system("shutdown -L")