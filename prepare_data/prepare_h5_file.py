'''
This python program creates an h5 file containing train, test, and validation set for the deep learning networks.
The sets are divided based on the number of patients (not the slices).

Note: After the creation of the h5 file, use the move_h5_data.py program to correctly divide train, test, and 
validation sets.
Each set will contain:  
    patients        -   the name of the patients    (N)
    slices          -   the slice number that was added for the matching patient    (N)
    Cine Group:
     +images            -   The dicom images/frames for the matching slice  (N X 50)
     +dicom_paths-      -   General image paths to the dicom images (e.g. IMAGEPATH\MultiPatient2015...\...)    (N x 50)
     +centroids         -   The x, y coords of the centre of the left ventricle for each slice, with half the length of the edge    (N x 3)
     +landmark_coords   -   converted global landmarks from cim (N x 2 x 2 x 168)
     +es_indices        -   indices of the frames showing end systole  (N)
    Tagged Group:
     +images            -   The dicom images/frames for the tagged slices   (N)
     +dicom_paths-      -   General image paths to the dicom images (e.g. IMAGEPATH\MultiPatient2015...\...)    (N x 20)
     +centroids         -   The x, y coords of the centre of the left ventricle for each slice, with half the length of the edge    (N x 3)
     +landmark_coords   -   converted global landmarks from cim (N x 20 x 2 x168)
     +es_indices        -   indices of the frames showing end systole  (N)
    (N = Number of slices)
Author: Amos Rada
Date:   25/02/2019
'''
# import needed libraries
import h5py
import numpy as np
import os
import logging
import pydicom
import sys
from time import time
from datetime import datetime
from random import shuffle
# importing from created functions
from general_functions import log_and_print, log_error_and_print, calculate_time_elapsed, sendemail
from image_pointer_functions import load_ptr_content, get_slices, get_image_path
from cim_functions import get_cim_path, get_cim_patients, get_global_landmarks, get_es_index
from LV_modeller_functions import read_mapping_file, get_folder_id, get_cine_es_index
from hdf5_functions import create_datasets, add_datasets
from dicom_functions import convert_3D_points_to_2D
from image_functions import pad_image, print_image_with_landmarks
from coordinates_functions import pad_coordinates, calculate_centroid, calculate_edge_length, calculate_distance, translate_coordinates
from cvi42_functions import get_contour_points

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

def get_cine_data(ptr_content, slice_num, cvi42_path, LVModel_path, folder_id, global_landmarks, es_index, filepaths):
    '''
    This function obtains data needed for the cine set.
    Inputs:
        ptr_content (numpy array) = content of the pointer
        slice_num (int) = slice number of the current patient
        cvi42_path (string) = path to the folders containing the zip files which contains the contour files
        LVModel_path (string) = path to the folders containing the ED and ES GP's and SliceInfoFiles
        folder_id (string) = folder id of the current patient based on the mapping file
        global_landmarks (3 x 168 array) = 3D landmarks from the cim folders
        es_index (int) = index of the end sytolic frame from the tagged slice
        filepaths (list) = directories containing the multipatient/singlepatient folders
    Outputs:
        cine_images (50x256x256 array) = list containing the pixel array for each image in the current slice
        cine_dicom_paths (50x1 list of strings) = paths to each dicom frame in that slice
        cine_centroid (1 x 3 list of float) = contains x coordinate, y coordinate, and half the edge length 
        cine_es_index (int) = index of the end systolic frame of the current cine slice
        cine_landmark_coords (2 x 2 x 168 array of float) = landmark coordinates for the ed and es frames of the current slice
    '''
    # initialise the data needed for the cine set
    cine_dicom_paths = []
    cine_images = []
    cine_landmark_coords = []
    cine_centroid = []
    
    # if folder in the LVModeller is found, try to get the ES index of the cine slice
    if folder_id != "doesn't exist":
        cine_es_index = get_cine_es_index(LVModel_path, folder_id, ptr_content, slice_num)
    else:   # if folder id is not found (i.e. not in the mapping file, or no folder in the LVModeller)
        cine_es_index = -1

    # get the cine frames for the current slice from the image pointer
    cine_frames = ptr_content[np.logical_and(ptr_content["series"]==0, ptr_content["slice"]==slice_num)]
    # loop through the frames
    for i, fr in enumerate(cine_frames):
        # condition to limit the number of frames to 50
        if i < 50:
            # get the general image path to the frame
            gen_path = fr["path"]
            # get the specific image path to the frame
            image_path = get_image_path(gen_path, filepaths)
            # add the general path to the set
            cine_dicom_paths.append(gen_path)

            # get the image/pixel array from the dicom header
            orig_image = pydicom.dcmread(image_path).pixel_array
            # pad the image to 256x256
            padded_image = pad_image(orig_image, 256)
            # add this image to the set
            cine_images.append(padded_image)

            # if the frame is the ED frame
            if fr["frame"] == 0:
                translate = False   #variable to indicate whether we're going to translate the landmark coords based on the centroid of the contour
                new_landmark_coords = [[],[]]   #initialise new landmarks
                # we're going to stack the landmarks (i.e. from [[x1,x2,x3,...],[y1,y2,y3,...],[z1,z2,z3,...]]), we're going to convert it to
                # [[x1,y1,z1],[x2,y2,z2],...])
                try:
                    stacked_landmarks = np.stack((global_landmarks[fr["frame"]][0], global_landmarks[fr["frame"]][1], global_landmarks[fr["frame"]][2]), axis=-1)
                except:
                    stacked_landmarks = np.stack((global_landmarks[i][0], global_landmarks[i][1], global_landmarks[i][2]), axis=-1)
                # convert the points to 2D using the plane of the cine image
                landmark_coords = convert_3D_points_to_2D(stacked_landmarks, image_path)
                # put the converted landmarks in unstacked format (i.e. [[x1,x2,x3,...],[y1,y2,y3,...]])
                for i in range(len(landmark_coords)):
                    new_landmark_coords[0].append(landmark_coords[i][0])
                    new_landmark_coords[1].append(landmark_coords[i][1])
                
                # check if there are epi contours for the current slice of the current patient
                try:
                    epi_contours = get_contour_points(cvi42_path, folder_id, ptr_content, slice_num)
                except:
                    epi_contours = -1
            
                # if there is, we translate the coordinates based on the centre of the contour
                if epi_contours != -1:
                    contour_centroid = calculate_centroid(epi_contours)
                    landmark_centroid = calculate_centroid(new_landmark_coords)
                    #distance = calculate_distance(contour_centroid, landmark_centroid)
                    #print(distance)
                    #if distance >= 1:
                    new_landmark_coords = translate_coordinates(new_landmark_coords, landmark_centroid, contour_centroid)
                    translate = True    #set translate to true (to be used in the es index)

                # convert the coordinates into coordinates that will fit the padded image
                new_landmark_coords = pad_coordinates(new_landmark_coords, orig_image.shape, padded_image.shape)

                # add the new landmark coords to the set
                cine_landmark_coords.append(new_landmark_coords)

                # calculate the centroid, and half the edge of the coordinates, and add to set
                cine_centroid = calculate_centroid(new_landmark_coords)
                edge_length = calculate_edge_length(cine_centroid, new_landmark_coords)
                cine_centroid.append(edge_length)

            # if the current frame is the es frame of the cine slice
            elif fr["frame"] == cine_es_index:
                # initialise landmarks, stack the global landmarks, convert to 3D
                new_landmark_coords = [[],[]]
                stacked_landmarks = np.stack((global_landmarks[es_index][0], global_landmarks[es_index][1], global_landmarks[es_index][2]), axis=-1)
                landmark_coords = convert_3D_points_to_2D(stacked_landmarks, image_path)
                # unstack the 2D landmarks
                for i in range(len(landmark_coords)):
                    new_landmark_coords[0].append(landmark_coords[i][0])
                    new_landmark_coords[1].append(landmark_coords[i][1])

                if translate:   #if we translated the ED landmarks, we translate the ES landmarks as well
                    new_landmark_coords = translate_coordinates(new_landmark_coords, landmark_centroid, contour_centroid)
                    translate = False #reset translate condition

                # convert the coordinates into coordinates that will fit the padded image
                new_landmark_coords = pad_coordinates(new_landmark_coords, orig_image.shape, padded_image.shape)

                # add the coordinates to the set
                cine_landmark_coords.append(new_landmark_coords)
    
    # if the es index is not found, we just add landmarks containing -1's. this will ensure that the shape of the landmarks that
    # we're adding to the dataset will be the uniform
    if cine_es_index == -1 and len(cine_landmark_coords) != 2:
        cine_landmark_coords.append([[-1]*168, [-1]*168])
    
    # if there are less than 50 frames found, we keep adding the last path and image to the set until it reaches 50
    if len(cine_images) < 50:
        for i in range(50-len(cine_frames)):
            cine_images.append(padded_image)
            cine_dicom_paths.append(gen_path)    

    return cine_images, cine_dicom_paths, cine_centroid, cine_es_index, cine_landmark_coords

def get_tagged_data(ptr_content, slice_num, global_landmarks, es_index, filepaths):
    '''
    This function obtains data needed for the tagged set
    Inputs:
        ptr_content (numpy array) = content of the pointer
        slice_num (int) = slice number of the current patient
        global_landmarks (3 x 168 array) = 3D landmarks from the cim folders
        es_index (int) = index of the end sytolic frame from the tagged slice
        filepaths (list) = directories containing the multipatient/singlepatient folders
    Outputs:
        tagged_images (20x256x256 array) = list containing the pixel array for each image in the current slice
        tagged_dicom_paths (20x1 list of strings) = paths to each dicom frame in that slice
        tagged_centroid (1 x 3 list of float) = contains x coordinate, y coordinate, and half the edge length 
        tagged_es_index (int) = index of the end systolic frame of the current cine slice
        tagged_landmark_coords (20 x 2 x 168 array of float) = landmark coordinates for the ed and es frames of the current slice
    '''
    # initiliase the data needed for the tagged set
    tagged_dicom_paths = []
    tagged_images = []
    tagged_landmark_coords = []
    tagged_centroid = []

    # get the tagged frames for the current slice from the image pointer
    tagged_frames = ptr_content[np.logical_and(ptr_content["series"]==1, ptr_content["slice"]==slice_num)]
    # loop through each frame in the slice
    for i, fr in enumerate(tagged_frames):
        # condition to limit the number of frames to 20
        if i < 20:
            new_landmark_coords = [[],[]]   #initilaising new landmark coords
            # get the general image path to the dicom file
            gen_path = fr["path"]  
            # get the specific image path to the dicom file
            image_path = get_image_path(gen_path, filepaths)
            # we're going to stack the landmarks (i.e. from [[x1,x2,x3,...],[y1,y2,y3,...],[z1,z2,z3,...]]), we're going to convert it to
            # [[x1,y1,z1],[x2,y2,z2],...])
            try:
                stacked_landmarks = np.stack((global_landmarks[fr["frame"]][0], global_landmarks[fr["frame"]][1], global_landmarks[fr["frame"]][2]), axis=-1)
            except:
                stacked_landmarks = np.stack((global_landmarks[i][0], global_landmarks[i][1], global_landmarks[i][2]), axis=-1)
            # convert the global landmarks into 2D using the plane of the tagged image
            landmark_coords = convert_3D_points_to_2D(stacked_landmarks, image_path)
            # unstack the 2D landmarks
            for i in range(len(landmark_coords)):
                new_landmark_coords[0].append(landmark_coords[i][0])
                new_landmark_coords[1].append(landmark_coords[i][1])
            # get the image from the dicom header
            orig_image = pydicom.dcmread(image_path).pixel_array
            # padd the image to 256 x 256
            padded_image = pad_image(orig_image, 256)
            # convert the coordinates into coordinates that will fit the padded image
            new_landmark_coords = pad_coordinates(new_landmark_coords, orig_image.shape, padded_image.shape)
            
            # add the landmark coords, dicom path, and images to the set
            tagged_landmark_coords.append(new_landmark_coords)
            tagged_dicom_paths.append(gen_path)
            tagged_images.append(padded_image)
    
    # if there are less than 20 frames, we keep adding the last dicom path, landmark coords, and image to the 
    # set until it reaches 20
    if len(tagged_images) < 20:
        for i in range(20-len(tagged_frames)):
            tagged_dicom_paths.append(gen_path)
            tagged_landmark_coords.append(new_landmark_coords)
            tagged_images.append(padded_image)

    # the tagged es index will be the same as the es index passed to the function
    tagged_es_index = es_index
    # calculate the centroid and half the edge length from the landmark coords
    tagged_centroid = calculate_centroid(tagged_landmark_coords[0])
    edge_length = calculate_edge_length(tagged_centroid, tagged_landmark_coords[0])
    # add to set
    tagged_centroid.append(edge_length)

    return tagged_images, tagged_dicom_paths, tagged_centroid, tagged_es_index, tagged_landmark_coords

def get_all_data(dsm, filepaths, ptr_files_path, cvi42_path, LVModel_path, f_ids, p_ids, cim_patients, ptr):
    '''
    This function obtains all the data needed for each slice of each patient.
    Inputs:
        dsm = DataSetModel
        filepaths (list) = directories containing the multipatient/singlepatient folders
        ptr_files_path (string) = where the image pointers CONTAINING THE MATCHING slices are stored
        cvi42_path (string) = where the zip files containing the contours are stored
        LVModel_path (string) = path to the folders containing the ED and ES GP's and SliceInfoFiles
        f_ids (list of strings) = folder ids of the patients based on the mapping file
        p_ids (list of strings) = patient ids based on the mapping file
        cim_patients (list of strings) = paths to all the patients in the cim models 
        ptr (list) = filename image pointer of the current patient
    Outputs:
        None. This functions adds data to the dataset model.
    '''
    # check if the dataset model has been reset
    if len(dsm.slices) == 0:
        init = True #need for initialisation of numpy arrays
    else:
        init = False

    patient_name = ptr.replace("_match.img_imageptr", "")   #get the patient name
    pat_id = patient_name.replace("_", "")[:8] #get the patient id to find the matching folder in the LVModel path using the csv file

    # get the patient cim path
    cim_path = get_cim_path(patient_name, cim_patients)
    
    # get the path of the pointer
    ptr_path = os.path.join(ptr_files_path, ptr)

    # read the content of the image pointer
    ptr_content = load_ptr_content(ptr_path)

    # read the pointer and get the slices
    ptr_slices = get_slices(ptr_content)
    
    # get the folder id of the current patient from the mapping file (this is needed to get the es index for the cine slices)
    try:
        folder_id = get_folder_id(f_ids, p_ids, pat_id)
    except ValueError:
        folder_id = "doesn't exist"
        log_error_and_print("Folder ID not found in mapping file. PatID: {}".format(pat_id))

    slices = [] #this will be used to determine whether there are landmarks for the current patient
    # Loop through each pointer slice
    for ptr_slice in ptr_slices:
        try:
            # get the global landmarks, if there are no global landmarks, skip the slice
            global_landmarks = get_global_landmarks(cim_path, ptr_slice)
            es_index = get_es_index(cim_path, ptr_slice)
        except:
            #print("No landmark coordinates for Patient {} Slice {}. CIM Path: {}".format(patient_name, ptr_slice, cim_path))
            continue
        #print("Landmark coordinates found for patient {} Slice {}".format(patient_name, ptr_slice))
        #print("Image pointer path: {}".format(ptr_path))
        slices.append(ptr_slice)
        
        # get the data needed for the cine set
        cine_images, cine_dicom_paths, cine_centroid, cine_es_index, cine_landmark_coords = get_cine_data(ptr_content, ptr_slice, cvi42_path, LVModel_path, folder_id, global_landmarks, es_index, filepaths)
        #print_image_with_landmarks(cine_images[cine_es_index], cine_landmark_coords[1])
        # get the data needed for the tagged set
        tagged_images, tagged_dicom_paths, tagged_centroid, tagged_es_index, tagged_landmark_coords = get_tagged_data(ptr_content, ptr_slice, global_landmarks, es_index, filepaths)
        #print_image_with_landmarks(tagged_images[0], tagged_landmark_coords[0])
        
        # add needed data to the dataset model
        dsm.patient_names.append(patient_name)
        dsm.cine_dicom_paths.append(cine_dicom_paths)
        dsm.tagged_dicom_paths.append(tagged_dicom_paths)
        if init:
            dsm.slices = np.array([ptr_slice])
            # cine set
            dsm.cine_centroids = np.array([cine_centroid]) #DONE
            dsm.cine_landmark_coords = np.array([cine_landmark_coords]) #DONE
            dsm.cine_images = np.array([cine_images])   #DONE   
            dsm.cine_es_indices = np.array([cine_es_index])   #DONE
            # tagged set
            dsm.tagged_centroids = np.array([tagged_centroid]) #DONE
            dsm.tagged_landmark_coords = np.array([tagged_landmark_coords]) #DONE
            dsm.tagged_images = np.array([tagged_images])   #DONE
            dsm.tagged_es_indices = np.array([tagged_es_index])   #DONE

            init = False    #to indicate that we're not initialising the data in model as numpy arrays anymore

        else:
            dsm.slices = np.append(dsm.slices, [ptr_slice], axis = 0)
            # cine set
            dsm.cine_centroids = np.append(dsm.cine_centroids, [cine_centroid], axis = 0)
            dsm.cine_landmark_coords = np.append(dsm.cine_landmark_coords, [cine_landmark_coords], axis = 0)
            dsm.cine_images = np.append(dsm.cine_images, [cine_images], axis = 0)
            dsm.cine_es_indices = np.append(dsm.cine_es_indices, [cine_es_index], axis = 0)
            # tagged set
            dsm.tagged_centroids = np.append(dsm.tagged_centroids, [tagged_centroid], axis = 0)
            dsm.tagged_landmark_coords = np.append(dsm.tagged_landmark_coords, [tagged_landmark_coords], axis = 0)
            dsm.tagged_images = np.append(dsm.tagged_images, [tagged_images], axis = 0)
            dsm.tagged_es_indices = np.append(dsm.tagged_es_indices, [tagged_es_index], axis = 0)

    if slices is None:
        log_error_and_print("No landmark coordinates for {}/{} slices. Patient: {} CIM Path: {}".format(len(ptr_slices), len(ptr_slices), patient_name, cim_path))
    else:
        if len(ptr_slices) > len(slices):
            log_error_and_print("No landmark coordinates for {}/{} slices. Patient: {} CIM Path: {}".format(len(ptr_slices)-len(slices), len(ptr_slices), patient_name, cim_path))

def create_h5_file(filepaths, ptr_files_path, cvi42_path, LVModel_path, f_ids, p_ids, cim_patients, output_dir, output_filename, dataset_dict):
    '''
    This function adds the data from the model to the h5 file.
    Inputs:
        dsm = DataSetModel
        filepaths (list) = directories containing the multipatient/singlepatient folders
        ptr_files_path (string) = where the image pointers CONTAINING THE MATCHING slices are stored
        cvi42_path (string) = where the zip files containing the contours are stored
        LVModel_path (string) = path to the folders containing the ED and ES GP's and SliceInfoFiles
        f_ids (list of strings) = folder ids of the patients based on the mapping file
        p_ids (list of strings) = patient ids based on the mapping file
        cim_patients (list of strings) = paths to all the patients in the cim models 
        output_dir (string) = where you want the h5 file to be stored
        output_filename (string) = name of the h5 file that is going to be created
        dataset_dict (dictionary) = contains the sets (train, validation, test) of patients
    Output:
        None. This function creates the h5 file
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
            get_all_data(dsm, filepaths, ptr_files_path, cvi42_path, LVModel_path, f_ids, p_ids, cim_patients, ptr)
            #if we have 5 unique patients added or if we have reached the end of the patient for that set
            if len(set(dsm.patient_names)) == 5 or (i == len(ptr_files)-1 and len(set(dsm.patient_names)) != 0):   
                p_cnt += len(set(dsm.patient_names))
                s_cnt += len(dsm.slices)
                print("Looped through {}/{} patients for {} set".format(i+1, len(ptr_files), key))
                try:
                    #creating the h5 file if it doesn't exist
                    if not os.path.isfile(os.path.join(output_dir, output_filename)):   
                        with h5py.File(os.path.join(output_dir, output_filename), 'w') as hf:
                            create_datasets(hf, key, dsm)
                                
                    else:   #if h5file exists, we just add to the data inside the h5 file
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

                except: #error handling
                    logger.error("Unexpected Error",exc_info = True)
                    log_error_and_print("{} Patients not added".format(len(set(dsm.patient_names))))
                    dsm = DataSetModel()
                    continue

        hrs, mins, secs = calculate_time_elapsed(start)
        log_and_print("Finished creating {} set".format(key))
        log_and_print("Total number of unique cases: {}".format(p_cnt))
        log_and_print("Total number of slices: {}".format(s_cnt))
        log_and_print("Elapsed time: {} hours {} minutes {} seconds\n".format(hrs, mins, secs))

def prepare_h5_files(filepaths, ptr_files_path, cvi42_path, LVModel_path, mapping_file, cim_patients, output_dir, output_filename, num_cases):
    '''
    This is the main function of the program.
    Inputs:
        dsm = DataSetModel
        filepaths (list) = directories containing the multipatient/singlepatient folders
        ptr_files_path (string) = where the image pointers CONTAINING THE MATCHING slices are stored
        cvi42_path (string) = where the zip files containing the contours are stored
        LVModel_path (string) = path to the folders containing the ED and ES GP's and SliceInfoFiles
        mapping_file (string) = directory to the confidential mapping file
        cim_patients (list of strings) = paths to all the patients in the cim models 
        output_dir (string) = where you want the h5 file to be stored
        output_filename (string) = name of the h5 file that is going to be created
        num_cases (int or None) = number of cases user wants to add to the h5 file
    Output:
        None. This function creates an h5 file.
    '''
    # create the output directory if it is non-existent
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    log_and_print("h5 file will be stored in {}".format(output_dir))
    log_and_print("Output filename: {}".format(output_filename))

    # get all the pointer file paths
    ptr_files = [f for f in os.listdir(ptr_files_path) if f.endswith("_match.img_imageptr")]
    
    log_and_print("Creating h5 file from {} pointer files\n".format(len(ptr_files)))

    shuffle_data = True

    # shuffle the image pointers
    if shuffle_data:
        log_and_print("Shuffling data...")
        shuffle(ptr_files)

    # if the user has set a limit to the number of cases he/she wants to include
    if num_cases is not None:
        ptr_files = ptr_files[:num_cases]

    # Divide the data into 60% train, 20% validation, and 20% test
    # ref code: http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html
    train = ptr_files[0:int(0.6*len(ptr_files))]

    val = ptr_files[int(0.6*len(ptr_files)):int(0.8*len(ptr_files))]

    test = ptr_files[int(0.8*len(ptr_files)):]

    dataset_dict = {"train": train, "validation": val, "test": test}

    # get the folder ids and the patient ids from the confidential mapping file
    f_ids, p_ids = read_mapping_file(mapping_file)

    # create the h5 file
    create_h5_file(filepaths, ptr_files_path, cvi42_path, LVModel_path, f_ids, p_ids, cim_patients, output_dir, output_filename, dataset_dict)

if __name__ == "__main__":
    '''
    Calls the main function (prepare_h5_file)
    To be modified by user: 
        logname (string) = the name of your log file
        filepaths (list of strings) = where the multipatient folders are stored (even if there's only one path, put it in a list (i.e. []))
        ptr_files_path (string) = where the image pointers CONTAINING THE MATCHING slices are stored
        cvi42_path (string) = where the zip files containing the contours are stored
        LVModel_path (string) = where the LVModeller folders are stored (contains the ED and ES GP and SliceInfoFiles)
        mapping_file (string) = directory to the confidential mapping file (maps folder id and patient id)
        cim_dir (string) = where the cim models are stored
        cim_models (list of strings) = names of the cim models (as a list)
        num_cases (int or None) = number of cases you want to add to the h5 file (put None if you want to add all)
        output_dir (string) = where you want the h5 file to be stored
    '''
    # start logging
    start = time() # to keep time
    ts = datetime.fromtimestamp(start).strftime('%Y-%m-%d') #time stamp for the log file
    logname = "{}-prepare-h5-files.log".format(ts)
    logging.basicConfig(filename=logname, level=logging.DEBUG)
    output_messages = ["====================STARTING MAIN PROGRAM====================",
                        "Operation started at {}".format(datetime.now().time())]
    log_and_print(output_messages)

    # where the multipatient files are stored
    filepaths = ["E:\\Original Images\\2014", "E:\\Original Images\\2015"]

    # where the pointer files with matching series and cim files
    ptr_files_path = "C:\\Users\\arad572\\Documents\\Summer Research\\Summer Research Data\\img_ptrs\\matches"

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
    output_dir = "C:\\Users\\arad572\\Documents\\Summer Research\\Summer Research Data\\h5_files"

    # name of the h5file
    if num_cases is not None:
        output_filename = "UK_Biobank_{}cases.h5".format(num_cases)
    else:
        output_filename = "UK_Biobank.h5"

    # if h5 file already exists, prompt the user if they want to overwrite the h5 file
    if os.path.isfile(os.path.join(output_dir, output_filename)):
        overwrite = input("{} already exists. Do you want to overwrite the file? (Y or N): ".format(output_filename))
        if overwrite.lower() == "y":
            os.remove(os.path.join(output_dir, output_filename))
        else:
            sys.exit()

    try:
        prepare_h5_files(filepaths, ptr_files_path, cvi42_path, LVModel_path, mapping_file, cim_patients, output_dir, output_filename, num_cases)

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