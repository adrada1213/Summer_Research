from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse
import tensorflow as tf
import numpy as np
import time
import h5py
import os

class LocalisationNetwork:
    output_size = 3
    session = None
    bbox_adjustment = 0.0 # not needed anymore to be adjusted, ground truth was already adjusted

    def __del__(self):
        if self.session is not None:
            print('Closing tf.session...')
            self.session.close()

    def __init__(self, model_dir, model_name):
        self.network_name = model_name

        # ----- tensorflow stuff -------------
        tf.reset_default_graph() 

        if self.session is None:
            print('\nCreating new tf.session')
            self.session = tf.Session()

        print('Restoring model {}'.format(model_name))
        new_saver = tf.train.import_meta_graph('{}\\{}.meta'.format(model_dir, model_name))
        new_saver.restore(self.session, tf.train.latest_checkpoint(model_dir))
        
        # restore the tensors
        graph = tf.get_default_graph()
        self.placeholder_x = graph.get_tensor_by_name("x1:0")
        self.placeholder_y = graph.get_tensor_by_name("y:0")
        self.prediction = graph.get_tensor_by_name("y_:0")
        self.loss = graph.get_tensor_by_name("loss:0")

        self.iou = graph.get_tensor_by_name("iou:0")
        self.is_training = graph.get_tensor_by_name("is_training:0")
        self.keep_prob = graph.get_tensor_by_name("keep_prob:0")
        
        # we gonna print the memory usage of the model
        #if self.session is not None:
        #    self.print_memory_usage() #not needed for network, doesn't work with some versions of TF (guess)

    '''
        A way to print the actual usage of memory used by the model
        https://stackoverflow.com/questions/36123740/is-there-a-way-of-determining-how-much-gpu-memory-is-in-use-by-tensorflow
    '''
    def print_memory_usage(self):
        with tf.device('/device:GPU:0'):  # Replace with device you are interested in
            self.bytes_in_use = BytesInUse()
        
        mem_bytes = self.session.run(self.bytes_in_use)
        print('\nModel {} restored, memory usage {:.2f} MB'.format(self.network_name, mem_bytes / 2**20))

    def save_predictions(self, input_file, points, output_path):
        new_filename = input_file
        if (input_file.endswith('.h5')):
            new_filename = input_file[:-3] # strip the .h5

        output_filename = '{}.result.h5'.format(new_filename)
        print('Saving predictions in {}...'.format(output_filename))
        save_result(output_path, output_filename, points)
        print('Prediction saved!')
    
    def predict_and_calculate_loss(self, data_x, data_y):
        # make sure the data are in correct shape
        X_data = data_x
        Y_data = data_y
        #Y_data = adjust_bounding_box(Y_data, self.bbox_adjustment)
        reshaped_labels = np.reshape(Y_data, [-1, self.output_size])

        # --- do the prediction here ---
        start_time = time.time()
        
        #feed_dict ={x1: X_data1, y: reshaped_labels, is_training: training_mode, keep_prob: keep_prob }
        feed_dict = { self.placeholder_x: X_data , self.placeholder_y: reshaped_labels, self.keep_prob: 1, self.is_training: False}
        corners, loss, iou_val = self.session.run([self.prediction, self.loss, self.iou], feed_dict)
        
        time_taken = time.time() - start_time
        
        # --- end of prediction/reshaping ---
        
        print("Time taken for prediction of {} images: {:.2f} seconds".format(len(X_data), time_taken))
        print("{:.2f} Image/second".format(len(X_data) / time_taken))
        print("\nLoss (MSE)  = ", loss," IoU = ", iou_val)
        
        return corners

    def predict_corners(self, ed_frame):
        X_data = ed_frame

        # --- do the prediction here ---
        start_time = time.time()
        
        feed_dict = { self.placeholder_x: X_data, self.keep_prob: 1, self.is_training: False }
        corners = self.session.run(self.prediction, feed_dict)

        
        time_taken = time.time() - start_time
        
        # --- end of prediction ---
        fps = len(X_data) / time_taken
        print("Bbox prediction - {} images: {:.2f} seconds ({:.2f} fps)".format(len(X_data), time_taken, fps))
        
        return corners


# ---- non class functions ---
def save_result(output_path, filename, coords):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    if not os.path.isfile(os.path.join(output_path, filename)):
        with h5py.File(os.path.join(output_path, filename), 'w') as hf:
            hf.create_dataset("centroid_preds", data=coords, maxshape = (None, 3))
    
    else:
        with h5py.File(os.path.join(output_path, filename), 'a') as hf:
            hf["centroid_preds"].resize((hf["centroid_preds"].shape[0])+coords.shape[0], axis = 0)
            hf["centroid_preds"][-coords.shape[0]:] = coords

def adjust_bounding_box(arr, adjustment_fraction = 0):
    if adjustment_fraction == 0:
        return arr

    diff_x  = arr[:,2]-arr[:,0]
    diff_y = arr[:,3]-arr[:,1]

    # stack em vertically
    stack = np.vstack((-diff_x, -diff_y, diff_x, diff_y))
    stack = np.transpose(stack)
    # should look like this now [[-dx, -dy, +dx +dy]... [..]]
    stack = adjustment_fraction * stack

    # add the adjustment
    return arr + stack

def predict_data(model_dir, model_name, input_path, input_file, output_path):
    # -- load the network
    model = LocalisationNetwork(model_dir, model_name)

    # -- load the data
    with h5py.File(os.path.join(input_path,input_file), 'r') as hl:
        data_x = np.asarray(hl.get('ed_imgs')[:,0,:,:]) # frame 0 only, ED frame
        data_y = np.asarray(hl.get('bbox_corners'))

    # -- predict and save
    bbox_corners = model.predict_and_calculate_loss(data_x, data_y)
    model.save_predictions(input_file, bbox_corners, output_path)


if __name__ == "__main__":    
    # input file
    base_path = '/mnt/cube/edward-playground/ukb_tagging/rnncnn'
    
    # network file
    model_path  = '{}/models/ds_local_all'.format(base_path)
    model_name = 'localizer_2'
    
    # input file
    data_path = '/mnt/cube/edward-playground/ukb_tagging/data_sequence_original'
    output_path = '{}/ds_local_all_output'.format(base_path)

    input_file = 'CIM_DATA_EL1.seq.noresize.0.h5'

    training_mode = False
    keep_prob = 1

    predict_data(model_path, model_name, data_path, input_file, output_path)

    