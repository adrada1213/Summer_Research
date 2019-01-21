import load_localisation_network as ll
import numpy as np
import h5py
import os

base_path = '/mnt/cube/edward-playground/ukb_tagging'

# input file
data_path = '/mnt/cube/edward-playground/ukb_tagging/data_sequence_original'
output_path = '{}/rnncnn/ds_local_all_output'.format(base_path)

# network file
model_path  = '{}/rnncnn/models/ds_local_all'.format(base_path)
model_name = 'localizer_2'

if __name__ == "__main__":    
    # traverse the input folder, keep as array to combine later
    files  = os.listdir(data_path)
    # get all the .h5 input filenames
    input_files =  [f for f in files if "CIM_D" in f]

    print('{} files found!'.format(len(input_files)))
    print(input_files)

    # -- load the network
    model = ll.LocalisationNetwork(model_path, model_name)

    # loop through all the input files, and run the network
    for num, input_file in enumerate(input_files):
        print("\n--------------------------")
        print('\nPredicting Bounding Box Corners: {} ({}/{})'.format(input_file, num+1, len(input_files)))

        # -- load the data
        with h5py.File(os.path.join(data_path,input_file), 'r') as hl:
            data_x = np.asarray(hl.get('ed_imgs')[:,0,:,:]) # frame 0 only, ED frame
            data_y = np.asarray(hl.get('bbox_corners'))

        # -- predict and save
        points = model.predict_and_calculate_loss(data_x, data_y)
        model.save_predictions(input_file, points, output_path)

    print("Done!")