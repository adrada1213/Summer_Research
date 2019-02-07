import load_coords_rnncnn as lr
import numpy as np
import h5py
import os

base_path = '/mnt/cube/edward-playground/ukb_tagging/rnncnn'

# input file
data_path = '/mnt/cube/edward-playground/ukb_tagging/data_sequence'
output_path = '{}/rnncnn_reg5_20_output'.format(base_path)

# network file
model_path  = '{}/models/rnncnn_reg5_20'.format(base_path)
model_name = 'rnncnn_reg5_20'

time_steps = 20
max_time_steps = 20


if __name__ == "__main__":
    # traverse the input folder, keep as array to combine later
    files  = os.listdir(data_path)
    # get all the .h5 input filenames
    input_files =  [f for f in files if "CIM_D" in f]

    print('{} files found!'.format(len(input_files)))
    print(input_files)

    # -- load the network
    model = lr.NetworkModel(model_path, model_name)

    # loop through all the input files, and run the network
    for num, input_file in enumerate(input_files):
        print("\n--------------------------")
        print('\nPredicting Coords seq: {} ({}/{})'.format(input_file, num+1, len(input_files)))

        # -- load the data
        with h5py.File(os.path.join(data_path,input_file), 'r') as hl:
            data_x = np.asarray(hl.get('ed_imgs'))
            data_y = np.asarray(hl.get('ed_coords'))

        # -- predict and save
        points = model.predict_and_calculate_loss(data_x, data_y)
        model.save_predictions(input_file, points, output_path)

    print("Done!")