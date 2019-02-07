import h5py
import h5py_cache as h5c
import numpy as np
import os
import logging
import pydicom
from time import time
from datetime import datetime
from random import shuffle
from prepare_dicom_images import prepare_dicom_images
from prepare_data_functions import get_cim_path, get_cim_patients, log_and_print, log_error_and_print, calculate_time_elapsed, sendemail, calculate_centroid, translate_coordinates, calculate_edge_length
import glob
import fnmatch
from pointer_functions import load_ptr_content, get_slices
from cvi42_functions import read_mapping_file, get_cvi42_id, get_root, get_contour_points, get_indices
from dicom_functions import get_dicom_info, plot_images
from test import plot_contour_points

logger = logging.getLogger(__name__)

class DataSetModel:
    def __init__(self):
        self.patient_names = []
        self.slices = []

        self.cine_images = []
        self.cine_dicom_paths = []
        self.cine_centroids = []
        self.cine_es_indices = []
        self.cine_landmark_coords = []  #will only contain ed and es landmark points

        self.tagged_images = []
        self.tagged_dicom_paths = []
        self.tagged_centroids = []
        self.tagged_es_indices = []
        self.tagged_landmark_coords = []    #will contain all 20 frames of landmark points

def get_data_from_h5_file(h5_filepath, cim_path, patient_name, ptr_slices):
    # get the list of h5py files
    h5_files = [os.path.join(h5_filepath, f) for f in os.listdir(h5_filepath) if fnmatch.fnmatch(f, "*.seq.noresize.*.h5")]

    # get the cim patient name and model name from cim path
    cim_model_name, cim_pat_name = os.path.split(cim_path)

    # gets the paths of the h5 files of the specified observer
    paths = [f for f in h5_files if cim_model_name.lower() in f.lower()]

    patient_names = []
    slices = []
    tagged_centroids = []
    tagged_landmark_coords = []
    tagged_es_indices = []
    # loop through the paths and get info for current patient    
    for path in paths:
        with h5py.File(path, 'r') as hf:
            patients = np.array(hf.get("patients"))
            p_indices = np.array(np.where(patients==cim_pat_name))[0]
            if len(p_indices) != 0:
                tmp_slices = np.array(hf.get("slices")[p_indices[0]:p_indices[-1]+1])
                tmp_landmark_coords = np.array(hf.get("ed_coords")[p_indices[0]:p_indices[-1]+1,:,:,:])
                tmp_es_indices = np.array(hf.get("es_frame_idx")[p_indices[0]:p_indices[-1]+1])
                init = True
                for sl in ptr_slices:
                    p_index = np.array(np.where(tmp_slices=="series_1_slice_{}".format(sl+1)))[0]
                    if len(p_index) != 0:    #if we found the index of the slice with landmark coords
                        if len(p_index) > 1:
                            log_error_and_print("Duplicate slice! Patient: {}, CIM_Path: {}, h5 filename: {}, indices: {}".format(patient_name, cim_path, path, p_indices))
                            log_and_print("Adding only one slice...")
                            p_index = [p_index[0]]
                        slices.append(sl)
                        tagged_es_indices.extend(tmp_es_indices[p_index])
                        patient_names.append(patient_name.replace("_", " "))
                        tmp_tagged_centroid = calculate_centroid(tmp_landmark_coords[p_index,0,:,:][0]) #put in brackets so we can add it to numpy array with []
                        edge_length = calculate_edge_length(tmp_tagged_centroid, tmp_landmark_coords[p_index,0,:,:][0])
                        tmp_tagged_centroid.extend([edge_length])
                        tmp_tagged_centroid = [tmp_tagged_centroid]
                        if init:
                            tagged_landmark_coords = tmp_landmark_coords[p_index,:,:,:]
                            tagged_centroids = np.array(tmp_tagged_centroid)
                            init = False
                        else:
                            tagged_centroids = np.append(tagged_centroids, tmp_tagged_centroid, axis = 0)
                            tagged_landmark_coords = np.append(tagged_landmark_coords, tmp_landmark_coords[p_index,:,:,:], axis = 0)
                            
                if len(slices) != 0:    #add to dataset model
                    slices = np.array(slices)
                    tagged_es_indices = np.array(tagged_es_indices)

                    if slices.shape[0] != tagged_landmark_coords.shape[0]:
                        print(slices.shape[0], tagged_landmark_coords.shape[0])
                        log_error_and_print("Adding unequal number of data...\nPatient: {}, CIM_Path: {}, h5 filename: {}, indices: {}".format(patient_name, cim_path, path, p_indices))

                    return patient_names, slices, tagged_centroids, tagged_landmark_coords, tagged_es_indices
                
    return None, None, None, None, None

def get_all_data(dsm, filepaths, ptr_files_path, eds_h5_filepath, LVModel_path, cvi42_path, cvi42_ids, p_ids, cim_patients, ptr):
    # loop through the patients and obtain needed data for the dataset model
    if len(dsm.slices) == 0:
        init = True #need for initialisation of numpy arrays
    else:
        init = False
    #for ptr in ptr_files:
    patient_name = ptr.replace("_match.img_imageptr", "")   #get the patient name
    pat_id = patient_name.replace("_", "")[:8] #get the patient id to find the matching folder in the LVModel path using the csv file

    # get the cim path for the patient MODEL\PatientName
    cim_path = get_cim_path(patient_name, cim_patients)
    
    # get the path of the pointer
    ptr_path = os.path.join(ptr_files_path, ptr)

    # read the content of the image pointer
    ptr_content = load_ptr_content(ptr_path)

    # read the pointer and get the slices, if there are duplicate slices in the file, skip the current patient
    # TODO: fix prepare_image_pointers to account for the slices with the same name
    ptr_slices = get_slices(ptr_content)
    if len(ptr_slices) > 3:
        log_error_and_print("Patient {} has duplicate slice names".format(patient_name))
        return
    
    # get the folder id of the current patient from the mapping file
    # if patient doesn't match any folder, go to the next patient
    try:
        cvi42_id = get_cvi42_id(cvi42_ids, p_ids, pat_id)
    except ValueError:
        log_error_and_print("Folder ID not found in mapping file. PatID: {}".format(pat_id))
        return

    # check if the zip file for the current patient exists (if not, return and go to the next patient)
    zip_path = os.path.join(cvi42_path, cvi42_id + "_cvi42.zip")
    if not os.path.isfile(zip_path):
        log_error_and_print("Missing zip file. PatID: {} FolderID: {}".format(pat_id, cvi42_id))
        return

    # get the contour points for the current patient
    try:
        root = get_root(cvi42_path, cvi42_id)
    except FileNotFoundError:
        log_error_and_print("Missing .cvi42wsx file. PatID: {} FolderID: {}".format(pat_id, cvi42_id))
        return

    ed_contour_pts = get_contour_points(root, ptr_content, ptr_slices, [0]*len(ptr_slices))

    # check which ed slice don't have epi/endo contours (no contours means we can't calculate centroid. No centroid means we can't
    # translate landmark points properly so, we're going to ignore that slice
    deduct = 0
    for i in range(len(ptr_slices)):
        if ed_contour_pts[i-deduct] == [[-1],[-1]]:
            del ed_contour_pts[i-deduct]
            ptr_slices = np.delete(ptr_slices, i-deduct)
            deduct += 1

    if len(ptr_slices) != 0:
        patient_names, slices, tagged_centroids, tagged_landmark_coords, tagged_es_indices = get_data_from_h5_file(eds_h5_filepath, cim_path, patient_name, ptr_slices)
    else:
        log_and_print("No contours found. PatID: {} Folder ID: {}".format(pat_id, cvi42_id))
        return

    if slices is not None:
        print("Landmark coordinates and contours found for patient {}".format(patient_name))
        print("Image pointer path: {}".format(ptr_path))

        # if the slices that has contour points don't have landmark coordinates in the cim folders, that slice is useless so we remove it
        deduct = 0
        if len(ptr_slices) != len(slices):
            print("Removing slices")
            for i in range(len(ptr_slices)):
                if not ptr_slices[i] in slices:
                    del ed_contour_pts[i-deduct]
                    deduct += 1

        # loop through the slices.
        #   1. Get the ES indices
        #   2. Calculate the centroid
        #   3. Get the dicom images (resized and padded), paths, x and y differences (translation)
        cine_es_indices = get_indices(LVModel_path, cvi42_id, ptr_content, slices)
        cine_centroids = []
        cine_landmark_coords = []
        cine_images_all = []
        tagged_images_all = []
        for i, sl in enumerate(slices):
            # get the info from dicom files
            cine_dicom_paths, cine_images, tagged_dicom_paths, tagged_images, x_ratio, y_ratio, w_diff, h_diff = get_dicom_info(filepaths, ptr_content, sl)

            # get the ED centroid and translate it
            cine_ed_centroid = calculate_centroid(ed_contour_pts[i])
            trans_cine_ed_centroid = [(cine_ed_centroid[0]*x_ratio)+(w_diff//2), (cine_ed_centroid[1]*y_ratio)+(h_diff//2)]
            tagged_centroid = tagged_centroids[i]

            # calculate the translation
            translation = [cine_c - tagged_c for cine_c, tagged_c in zip(trans_cine_ed_centroid, tagged_centroid[:2])]

            # get the ed and es landmark coordinates and translate them
            ed_landmark_coords = tagged_landmark_coords[i,0,:,:]
            cine_ed_landmark_coords = translate_coordinates(ed_landmark_coords, translation)
            
            if cine_es_indices[i] != -1:
                es_landmark_coords = tagged_landmark_coords[i,tagged_es_indices[i],:,:]
                cine_es_landmark_coords = translate_coordinates(es_landmark_coords, translation)
            else:
                cine_es_landmark_coords = [[-1]*168, [-1]*168]
        

            #contour_pts = translate_coordinates(ed_contour_pts[i], translation)
            #plot_contour_points(contour_pts, cine_images[0], sl)
            #plot_images(patient_name, cine_images[0], tagged_images[0], cine_ed_landmark_coords, tagged_landmark_coords[i,0,:,:], save_image=False)
            #plot_images(patient_name, cine_images[cine_es_indices[i]], tagged_images[tagged_es_indices[i]], cine_es_landmark_coords, tagged_landmark_coords[i,tagged_es_indices[i],:,:], save_image=False)

            # append to list
            trans_cine_ed_centroid.append(tagged_centroid[2])   #should be the same width since the landmark coords weren't resized, just translated
            cine_centroids.append(trans_cine_ed_centroid)
            cine_landmark_coords.append([cine_ed_landmark_coords, cine_es_landmark_coords])
            cine_images_all.append(cine_images)
            tagged_images_all.append(tagged_images)

            dsm.cine_dicom_paths.append(cine_dicom_paths)
            dsm.tagged_dicom_paths.append(tagged_dicom_paths)

        cine_centroids = np.array(cine_centroids)
        cine_landmark_coords = np.array(cine_landmark_coords)
        cine_images_all = np.array(cine_images_all)
        tagged_images_all = np.array(tagged_images_all)
        # add needed data to the dataset model
        dsm.patient_names.extend(patient_names) #DONE
            
        if init:
            dsm.slices = slices
            # cine set
            dsm.cine_centroids = cine_centroids #DONE
            dsm.cine_landmark_coords = cine_landmark_coords #DONE
            dsm.cine_images = cine_images_all   #DONE   
            dsm.cine_es_indices = cine_es_indices   #DONE
            # tagged set
            dsm.tagged_centroids = tagged_centroids #DONE
            dsm.tagged_landmark_coords = tagged_landmark_coords #DONE
            dsm.tagged_images = tagged_images_all   #DONE
            dsm.tagged_es_indices = tagged_es_indices   #DONE

        else:
            dsm.slices = np.append(dsm.slices, slices, axis = 0)
            # cine set
            dsm.cine_centroids = np.append(dsm.cine_centroids, cine_centroids, axis = 0)
            dsm.cine_landmark_coords = np.append(dsm.cine_landmark_coords, cine_landmark_coords, axis = 0)
            dsm.cine_images = np.append(dsm.cine_images, cine_images_all, axis = 0)
            dsm.cine_es_indices = np.append(dsm.cine_es_indices, cine_es_indices, axis = 0)
            # tagged set
            dsm.tagged_centroids = np.append(dsm.tagged_centroids, tagged_centroids, axis = 0)
            dsm.tagged_landmark_coords = np.append(dsm.tagged_landmark_coords, tagged_landmark_coords, axis = 0)
            dsm.tagged_images = np.append(dsm.tagged_images, tagged_images_all, axis = 0)
            dsm.tagged_es_indices = np.append(dsm.tagged_es_indices, tagged_es_indices, axis = 0)
    
    else:
        log_and_print("No landmark coordinates for patient {}".format(patient_name))

def create_datasets(hf, key, dsm):
    # create group for the current set
    grp = hf.create_group(key)
    grp_cine = grp.create_group("cine")
    grp_tagged = grp.create_group("tagged")

    # converting data to numpy arrays
    patients = np.array(dsm.patient_names, dtype=object)
    cine_dicom_paths = np.array(dsm.cine_dicom_paths, dtype = object)
    tagged_dicom_paths = np.array(dsm.tagged_dicom_paths, dtype = object)

    grp.create_dataset("patients", data=patients, dtype=h5py.special_dtype(vlen=str), maxshape = (None, ))
    grp.create_dataset("slices", data=dsm.slices, maxshape = (None, ))
    
    # put all the cine data in the cine group
    grp_cine.create_dataset("dicom_paths", data=cine_dicom_paths, dtype=h5py.special_dtype(vlen=str), maxshape = (None, 50))
    grp_cine.create_dataset("centroids", data=dsm.cine_centroids, maxshape = (None, 3))
    grp_cine.create_dataset("landmark_coords", data=dsm.cine_landmark_coords, maxshape = (None, 2, 2, 168))
    grp_cine.create_dataset("images", data=dsm.cine_images, maxshape = (None, 50, 256, 256))
    grp_cine.create_dataset("es_indices", data=dsm.cine_es_indices, maxshape = (None, ))

    # put all the tagged data in tagged group
    grp_tagged.create_dataset("dicom_paths", data=tagged_dicom_paths, dtype=h5py.special_dtype(vlen=str), maxshape = (None, 20))
    grp_tagged.create_dataset("centroids", data=dsm.tagged_centroids, maxshape = (None, 3))
    grp_tagged.create_dataset("landmark_coords", data=dsm.tagged_landmark_coords, maxshape = (None, 20, 2, 168))
    grp_tagged.create_dataset("images", data=dsm.tagged_images, maxshape = (None, 20, 256, 256))
    grp_tagged.create_dataset("es_indices", data=dsm.tagged_es_indices, maxshape = (None, ))

    

def add_datasets(hf, key, dsm):
    grp = hf["//{}".format(key)]
    grp_cine = hf["//{}//cine".format(key)]
    grp_tagged = hf["//{}//tagged".format(key)]
                        
    # converting data to numpy arrays
    patients = np.array(dsm.patient_names, dtype=object)
    cine_dicom_paths = np.array(dsm.cine_dicom_paths, dtype = object)
    tagged_dicom_paths = np.array(dsm.tagged_dicom_paths, dtype = object)

    grp["patients"].resize((grp["patients"].shape[0])+patients.shape[0], axis = 0)
    grp["patients"][-patients.shape[0]:] = patients

    grp["slices"].resize((grp["slices"].shape[0])+dsm.slices.shape[0], axis = 0)
    grp["slices"][-dsm.slices.shape[0]:] = dsm.slices

    # cines
    grp_cine["dicom_paths"].resize((grp_cine["dicom_paths"].shape[0])+cine_dicom_paths.shape[0], axis = 0)
    grp_cine["dicom_paths"][-cine_dicom_paths.shape[0]:] = cine_dicom_paths

    grp_cine["centroids"].resize((grp_cine["centroids"].shape[0])+dsm.cine_centroids.shape[0], axis = 0)
    grp_cine["centroids"][-dsm.cine_centroids.shape[0]:] = dsm.cine_centroids

    grp_cine["landmark_coords"].resize((grp_cine["landmark_coords"].shape[0])+dsm.cine_landmark_coords.shape[0], axis = 0)
    grp_cine["landmark_coords"][-dsm.cine_landmark_coords.shape[0]:] = dsm.cine_landmark_coords

    grp_cine["images"].resize((grp_cine["images"].shape[0])+dsm.cine_images.shape[0], axis = 0)
    grp_cine["images"][-dsm.cine_images.shape[0]:] = dsm.cine_images

    grp_cine["es_indices"].resize((grp_cine["es_indices"].shape[0])+dsm.cine_es_indices.shape[0], axis = 0)
    grp_cine["es_indices"][-dsm.cine_es_indices.shape[0]:] = dsm.cine_es_indices

    # tagged
    grp_tagged["dicom_paths"].resize((grp_tagged["dicom_paths"].shape[0])+tagged_dicom_paths.shape[0], axis = 0)
    grp_tagged["dicom_paths"][-tagged_dicom_paths.shape[0]:] = tagged_dicom_paths

    grp_tagged["centroids"].resize((grp_tagged["centroids"].shape[0])+dsm.tagged_centroids.shape[0], axis = 0)
    grp_tagged["centroids"][-dsm.tagged_centroids.shape[0]:] = dsm.tagged_centroids

    grp_tagged["landmark_coords"].resize((grp_tagged["landmark_coords"].shape[0])+dsm.tagged_landmark_coords.shape[0], axis = 0)
    grp_tagged["landmark_coords"][-dsm.tagged_landmark_coords.shape[0]:] = dsm.tagged_landmark_coords

    grp_tagged["images"].resize((grp_tagged["images"].shape[0])+dsm.tagged_images.shape[0], axis = 0)
    grp_tagged["images"][-dsm.tagged_images.shape[0]:] = dsm.tagged_images

    grp_tagged["es_indices"].resize((grp_tagged["es_indices"].shape[0])+dsm.tagged_es_indices.shape[0], axis = 0)
    grp_tagged["es_indices"][-dsm.tagged_es_indices.shape[0]:] = dsm.tagged_es_indices


def create_h5_file(filepaths, ptr_files_path, eds_h5_filepath, LVModel_path, cvi42_path, cvi42_ids, p_ids, cim_patients, output_dir, output_filename, dataset_dict):
    '''
    I need to create three groups (train, val, test)
    Inside each group, I need to create these datasets:
        + patients (as in the header) - N number of rows

    '''

    # create the h5 file
    # we incrementally write to h5 file so that we don't store everything to memory(slows down after some time)
    dsm = DataSetModel()    #initialise datasetmodel
    print("Creating h5file...")
    for key, ptr_files in dataset_dict.items():
        log_and_print("Obtaining data for {} set".format(key))
        start_ = time()
        start = time()
        p_cnt = 0   #initialise number unique cases added to set
        s_cnt = 0   #initiliase number of slices added to set
        for i, ptr in enumerate(ptr_files): #loop through the pointers
            get_all_data(dsm, filepaths, ptr_files_path, eds_h5_filepath, LVModel_path, cvi42_path, cvi42_ids, p_ids, cim_patients, ptr)
            if len(set(dsm.patient_names)) == 5 or (i == len(ptr_files)-1 and len(set(dsm.patient_names)) != 0):   #if we have 5 unique patients added or if we have reached the end of the dictionary
                p_cnt += len(set(dsm.patient_names))
                s_cnt += len(dsm.slices)
                print("Looped through {}/{} patients for {} set".format(i+1, len(ptr_files), key))
                try:
                    if not os.path.isfile(os.path.join(output_dir, output_filename)):   #creating the h5 file if it doesn't exist
                        with h5py.File(os.path.join(output_dir, output_filename), 'w') as hf:
                            create_datasets(hf, key, dsm)
                                
                    else:   #if h5file exists
                        with h5py.File(os.path.join(output_dir, output_filename), 'a') as hf:
                            if "//{}".format(key) not in hf:
                                create_datasets(hf, key, dsm)
                            else:
                                add_datasets(hf, key, dsm)

                    hrs, mins, secs = calculate_time_elapsed(start_)
                    print("Added {} unique cases to {} set".format(len(set(dsm.patient_names)), key))
                    print("Added {} slices to {} set".format(len(dsm.slices), key))
                    print("Elapsed time: {} hours {} minutes {} seconds\n".format(hrs, mins, secs))
                    start_ = time()
                    dsm = DataSetModel()    #reset the datasetmodel

                except:
                    logger.error("Unexpected Error",exc_info = True)
                    log_error_and_print("{} Patients not added".format(len(set(dsm.patient_names))))
                    dsm = DataSetModel()
                    continue


        hrs, mins, secs = calculate_time_elapsed(start)
        log_and_print("Finished creating {} set".format(key))
        log_and_print("Total number of unique cases: {}".format(p_cnt))
        log_and_print("Total number of slices: {}".format(s_cnt))
        log_and_print("Elapsed time: {} hours {} minutes {} seconds\n".format(hrs, mins, secs))

def prepare_h5_files(filepaths, ptr_files_path, ed_h5_filepath, LVModel_path, cvi42_path, mapping_file, cim_patients, output_dir, output_filename, num_cases):
    # create the output directory if it is non-existent
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    log_and_print("h5 file will be stored in {}".format(output_dir))
    log_and_print("Output filename: {}".format(output_filename))

    # get all the pointer file paths
    ptr_files = [f for f in os.listdir(ptr_files_path) if f.endswith("_match.img_imageptr")]
    if num_cases is not None:
        ptr_files = ptr_files[:num_cases]
    log_and_print("Creating h5 file from {} pointer files\n".format(len(ptr_files)))

    # main function of the program to create
    shuffle_data = True

    if shuffle_data:
        log_and_print("Shuffling data...")
        shuffle(ptr_files)

    # Divide the data into 60% train, 20% validation, and 20% test
    # ref code: http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html
    train = ptr_files[0:int(0.6*len(ptr_files))]

    val = ptr_files[int(0.6*len(ptr_files)):int(0.8*len(ptr_files))]

    test = ptr_files[int(0.8*len(ptr_files)):]

    dataset_dict = {"train": train, "validation": val, "test": test}

    cvi42_ids, p_ids = read_mapping_file(mapping_file)

    create_h5_file(filepaths, ptr_files_path, eds_h5_filepath, LVModel_path, cvi42_path, cvi42_ids, p_ids, cim_patients, output_dir, output_filename, dataset_dict)


if __name__ == "__main__":
    # start logging
    start = time() # to keep time
    ts = datetime.fromtimestamp(start).strftime('%Y-%m-%d') #time stamp for the log file
    logname = "{}-prepare-h5-files-all.log".format(ts)
    logging.basicConfig(filename=logname, level=logging.DEBUG)
    output_messages = ["====================STARTING MAIN PROGRAM====================",
                        "Operation started at {}".format(datetime.now().time())]
    log_and_print(output_messages)

    # where the multipatient files are stored
    filepaths = ["E:\\Original Images\\2014", "E:\\Original Images\\2015"]

    # where the pointer files with matching series and cim files
    ptr_files_path = "C:\\Users\\arad572\\Documents\\Summer Research\\code\\prepare_data\\img_ptrs\\matches"

    # where edward's h5py files are located
    eds_h5_filepath = "C:\\Users\\arad572\\Documents\\MR-tagging\\dataset-localNet\\data_sequence_original"

    # specify CVI42 filepath
    cvi42_path = "E:\\ContourFiles\\CVI42"

    # specify the location of the modellers
    LVModel_path = "E:\\LVModellerFormatV2"
    
    # specify where the mapping file is
    mapping_file = "E:\\confidential_bridging_file_r4.csv"

    # where the cim models are
    cim_dir = "C:\\Users\\arad572\\Downloads\\all CIM"
    cim_models = ["CIM_DATA_AB", "CIM_DATA_EL1", "CIM_DATA_EL2", "CIM_DATA_EM", "CIM_DATA_KF", "CIM_Data_ze_1", "CIM_DATA_ze_2", "CIM_DATA_ze_3", "CIM_DATA_ze_4"]
    
    # list the cim patient folders
    cim_patients = get_cim_patients(cim_dir, cim_models)
    
    # specify the number of cases we want to loop through (replace with None if you want all unique cases)
    num_cases = None

    # where h5 files will be stored
    output_dir = os.path.join(os.getcwd(), "h5_files")

    # name of the h5file
    if num_cases is not None:
        output_filename = "UK_Biobank_{}cases.h5".format(num_cases)
    else:
        output_filename = "UK_Biobank.h5"

    try:
        prepare_h5_files(filepaths, ptr_files_path, eds_h5_filepath, LVModel_path, cvi42_path, mapping_file, cim_patients, output_dir, output_filename, num_cases)

        hrs, mins, secs = calculate_time_elapsed(start)
        output_messages = ["====================H5 FILE CREATION FINISHED!====================",
                        "h5 file: {}".format(os.path.join(output_dir, output_filename)),
                        "Operation finished at {}".format(str(datetime.now())),
                        "Total elapsed time: {} hours {} minutes {} seconds\n".format(hrs, mins, secs)]
        log_and_print(output_messages)

        #sendemail("adrada1213@gmail.com", "ad_rada@hotmail.com", "prepare_h5_files.py Program Finished", "Here's the log file:", os.path.join(os.getcwd(),logname))

        #os.system("shutdown -s -t 600")
    
    except:
        logger.error("UNEXPECTED ERROR", exc_info = True)
        #sendemail("adrada1213@gmail.com", "ad_rada@hotmail.com", "prepare_h5_files.py Program Interrupted", "Here's the log file:", os.path.join(os.getcwd(),logname))

        #os.system("shutdown -s -t 600")