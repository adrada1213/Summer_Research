import load_localisation_network as ll
import numpy as np
import h5py
import os

base_path = 'E:\\cine-machine-learning\\network\\LocalisationNet'

# input file
data_path = 'E:\\cine-machine-learning\\dataset'
output_path = '{}\\ds_local_all_output'.format(base_path)

# network file
model_path  = '{}\\models\\localizer_2_20190125-1203'.format(base_path)
model_name = 'localizer_2'

if __name__ == "__main__":    
    # traverse the input folder, keep as array to combine later
    input_files  = ["UK_Biobank.h5"]

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
            grp_cine = hl["/test/cine"]
            patients = np.asarray(hl["test"].get('patients'))
            for i in range(0,len(patients),5):
                try:
                    data_x = np.asarray(grp_cine.get('images')[i:i+5,0,:,:]) # frame 0 only, ED frame
                    data_y = np.asarray(grp_cine.get('centroids')[i:i+5])
                    cnt = 5
                except:
                    data_x = np.asarray(grp_cine.get('images')[i:len(patients),0,:,:])
                    data_y = np.asarray(grp_cine.get('centroids')[i:len(patients)])
                    cnt = len(patients)-i
                # -- predict and save
                print("Predicting centroid for {} slices".format(cnt))
                points = model.predict_and_calculate_loss(data_x, data_y)
                model.save_predictions(input_file, points, output_path)

    print("Done!")
    #os.system("shutdown -L")