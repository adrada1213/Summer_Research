from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse
import tensorflow as tf
import numpy as np
import time
import h5py
import os

class NetworkModel:
    time_steps = 20
    output_size = 2 * 168
    session = None
    
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
        new_saver = tf.train.import_meta_graph('{}/{}.meta'.format(model_dir, model_name))
        new_saver.restore(self.session, tf.train.latest_checkpoint(model_dir))
        
        # restore the tensors
        graph = tf.get_default_graph()
        self.placeholder_x = graph.get_tensor_by_name("x:0")
        self.placeholder_y = graph.get_tensor_by_name("y:0")
        self.prediction = graph.get_tensor_by_name("y_:0")
        self.loss = graph.get_tensor_by_name("loss:0")
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

    def predict_and_calculate_loss(self, data_x, data_y):
        # make sure the data are in correct shape
        X_data = data_x[:,0:self.time_steps]
        Y_data = data_y[:,0:self.time_steps]
        reshaped_labels = np.reshape(Y_data, [-1, self.time_steps, self.output_size])
        
        # --- do the prediction here ---
        start_time = time.time()
        
        feed_dict ={self.placeholder_x: X_data , self.placeholder_y: reshaped_labels, self.keep_prob: 1}
        res, loss_mse = self.session.run([self.prediction, self.loss], feed_dict)
        points = np.reshape(res, [-1, self.time_steps, 2,168])
        
        time_taken = time.time() - start_time
        # --- end of prediction/reshaping ---
        
        nr_sequence = reshaped_labels.shape[0]
        print("Time taken for prediction of {} image sequences: {:.2f} seconds".format(nr_sequence, time_taken))
        print("{:.2f} Frame/second".format(nr_sequence * self.time_steps / time_taken))
        print("\nLoss (mse) = ", loss_mse)
        
        return points

    def predict_landmark_sequences(self, image_sequences):
        # make sure the data are in correct shape
        X_data = image_sequences[:,0:self.time_steps]
        
        # --- do the prediction here ---
        start_time = time.time()
        
        feed_dict ={self.placeholder_x: X_data, self.keep_prob: 1}
        res = self.session.run(self.prediction, feed_dict)
        points = np.reshape(res, [-1, self.time_steps, 2,168])
        
        time_taken = time.time() - start_time
        # --- end of prediction/reshaping ---
        
        nr_sequence = X_data.shape[0]
        fps = nr_sequence * self.time_steps / time_taken
        print("Landmark tracking - {} image sequences: {:.2f} seconds ({:.2f} fps)".format(nr_sequence, time_taken, fps))
        
        return points

    def save_predictions(self, input_file, points, output_path):
        new_filename = input_file
        if (input_file.endswith('.h5')):
            new_filename = input_file[:-3] # strip the .h5

        output_filename = '{}.rnn_points.h5'.format(new_filename)
        print('Saving prediction as {}...'.format(output_filename))
        save_result(output_path, output_filename, points)
        print('Prediction saved!')


# --- non class functions ----
def save_result(output_path, filename, preds):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    with h5py.File(os.path.join(output_path, filename), 'w') as hf:
        hf.create_dataset("ed_coords_preds", data=preds)


def predict_data(model_dir, model_name, input_path, input_file, output_path):

    # -- load the network
    model = NetworkModel(model_dir, model_name)
    
    # -- load the data
    with h5py.File(os.path.join(input_path,input_file), 'r') as hl:
        data_x = np.asarray(hl.get('ed_imgs'))
        data_y = np.asarray(hl.get('ed_coords'))

    # -- predict and save
    points = model.predict_and_calculate_loss(data_x, data_y)
    model.save_predictions(input_file, points, output_path)


'''
    Predict model points (ed_coords) given ed_imgs
    This function loads an h5 file and save the predicted ed_coords in another .h5 file
    TODO: The model must be loaded once and allow this to actually loop through files
'''
def predict_model_points(model_dir, model_name, input_path, input_file, output_path, time_steps):

    with h5py.File(os.path.join(input_path,input_file), 'r') as hl:
        data_x = np.asarray(hl.get('ed_imgs'))
        data_y = np.asarray(hl.get('ed_coords'))

    output_size = 2 * 168
    
    data_x = data_x
    data_y = data_y
    
    print("Total dataset", len(data_x))
    print("Total target", len(data_y))

    # reshaping
    X_data = data_x[:,0:time_steps]
    Y_data = data_y[:,0:time_steps]

    # ----- tensorflow stuff -------------
    tf.reset_default_graph() 
    with tf.Session() as sess:
        print('Restoring model ... ')
        new_saver = tf.train.import_meta_graph('{}/{}.meta'.format(model_dir, model_name))
        new_saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        
        # restore the tensors
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        prediction = graph.get_tensor_by_name("y_:0")
        #cnn_outputs = graph.get_tensor_by_name("cnn_outputs:0")
        loss = graph.get_tensor_by_name("loss:0")
        
        test_data = X_data
        test_label = Y_data
        #print(test_data.shape)
        #print(test_label.shape)
        reshaped_labels = np.reshape(test_label, [-1, time_steps, output_size])
        
        print('Model restored ... Running prediction')
        
        start_time = time.time()
        
        feed_dict ={x:test_data , y: reshaped_labels}
        print(prediction)
        #print(loss)
        
        #feats, res, loss_mse = sess.run([cnn_outputs, prediction, loss], feed_dict)
        res, loss_mse = sess.run([prediction, loss], feed_dict)
        # res = sess.run([prediction], feed_dict)
        
        time_taken = time.time() - start_time
        
        nr_sequence = reshaped_labels.shape[0]
        print("Time taken for prediction of {} image sequences: {:.2f} seconds".format(nr_sequence, time_taken))
        print("{:.2f} Frame/second".format(nr_sequence * time_steps / time_taken))
        
        # we would like to see the cnn outputs
        #print('CNN features', feats.shape)
        #for idx in range(0,3):    
        #    print('feature',idx,feats[0][idx][0:10])

        points = np.reshape(res, [-1, time_steps, 2,168])
        
        print("\nLoss (mse) = ", loss_mse)
        
        new_filename = input_file
        if (input_file.endswith('.h5')):
            new_filename = input_file[:-3] # strip the .h5

        output_filename = '{}.rnn_points.h5'.format(new_filename)
        print('Saving prediction as {}...'.format(output_filename))
        save_result(output_path, output_filename, points)
        print('Prediction saved!')

if __name__ == "__main__":    
    # input file
    base_path = '/mnt/cube/edward-playground/ukb_tagging/rnncnn'
    
    # network file
    model_path  = '{}/models/rnncnn_reg5_20'.format(base_path)
    model_name = 'rnncnn_reg5_20'
    
    # input file
    data_path = '/mnt/cube/edward-playground/ukb_tagging/data_sequence'
    #data_path = base_path
    output_path = '{}/rnncnn_reg5_20_output'.format(base_path)

    input_file = 'CIM_DATA_EL1.seq.0.h5'
    
    #predict_model_points(model_path, model_name, data_path, input_file, output_path, time_steps = 10)
    #predict_model_points(model_path, model_name, data_path, input_file, output_path, time_steps = 20)
    predict_data(model_path, model_name, data_path, input_file, output_path)