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
from prepare_data_functions import get_cim_path, get_slices, get_cim_patients, log_and_print, log_error_and_print, calculate_time_elapsed, sendemail
import glob
import fnmatch

logger = logging.getLogger(__name__)

class DataSetModel:
    def __init__(self):
        self.patient_names = []
        self.slices = []
        self.cine_images = []
        self.tagged_images = []
        self.landmark_coords = []
        self.bbox_corners = []
        self.cine_px_spaces = []
        self.tagged_px_spaces = []
        self.es_frame_idx = []
        self.cim_paths = []
        self.cine_dicom_paths = []
        self.tagged_dicom_paths = []
        self.cine_img_pos = []
        self.tagged_img_pos = []
        self.cine_img_orient = []
        self.tagged_img_orient = []

def get_data_from_h5_file(h5_filepath, cim_path, patient_name, ptr_slices):
    # get the list of h5py files
    h5_files = [os.path.join(h5_filepath, f) for f in os.listdir(h5_filepath) if fnmatch.fnmatch(f, "*.seq.noresize.*.h5")]

    # get the cim patient name and model name from cim path
    cim_model_name, cim_pat_name = os.path.split(cim_path)

    # gets the paths of the h5 files of the specified observer
    paths = [f for f in h5_files if cim_model_name.lower() in f.lower()]

    patient_names = []
    cim_paths = []
    slices = []
    bbox_corners = []
    landmark_coords = []
    # loop through the paths and get info for current patient    
    for path in paths:
        with h5py.File(path, 'r') as hf:
            patients = np.array(hf.get("patients"))
            p_indices = np.array(np.where(patients==cim_pat_name))[0]
            if len(p_indices) != 0:
                tmp_slices = np.array(hf.get("slices"))[p_indices[0]:p_indices[-1]+1]
                tmp_bbox_corners = np.array(hf.get("bbox_corners"))[p_indices[0]:p_indices[-1]+1,:]
                tmp_landmark_coords = np.array(hf.get("ed_coords"))[p_indices[0]:p_indices[-1]+1,:,:,:]
                init = True
                for sl in ptr_slices:
                    p_index = np.array(np.where(tmp_slices=="series_1_slice_{}".format(sl+1)))[0]
                    if len(p_index) != 0:    #if we found the index of the slice with bbox corners
                        if len(p_index) > 1:
                            log_error_and_print("Duplicate slice! Patient: {}, CIM_Path: {}, h5 filename: {}, indices: {}".format(patient_name, cim_path, path, p_indices))
                            log_and_print("Adding only one slice...")
                            p_index = [p_index[0]]
                        slices.append(sl)
                        cim_paths.append(cim_path)
                        patient_names.append(patient_name.replace("_", " "))
                        if init:
                            bbox_corners = tmp_bbox_corners[p_index,:]
                            landmark_coords = tmp_landmark_coords[p_index,:,:,:]
                            init = False
                        else:
                            bbox_corners = np.append(bbox_corners, tmp_bbox_corners[p_index, :], axis = 0)
                            landmark_coords = np.append(landmark_coords, tmp_landmark_coords[p_index,:,:,:], axis = 0)
                            
                if len(slices) != 0:    #add to dataset model
                    slices = np.array(slices)

                    if slices.shape[0] != bbox_corners.shape[0]:
                        print(slices.shape[0], bbox_corners.shape[0])
                        log_error_and_print("Adding unequal number of data...\nPatient: {}, CIM_Path: {}, h5 filename: {}, indices: {}".format(patient_name, cim_path, path, p_indices))

                    return patient_names, cim_paths, slices, bbox_corners, landmark_coords
                
    return None, None, None, None, None

def get_all_data(dsm, filepaths, ptr_files_path, eds_h5_filepath, cim_patients, ptr):
    # loop through the patients and obtain needed data for the dataset model
    if len(dsm.slices) == 0:
        init = True #need for initialisation of numpy arrays
    else:
        init = False
    #for ptr in ptr_files:
    patient_name = ptr.replace("_match.img_imageptr", "")   #get the patient name

    # get the cim path for the patient MODEL\PatientName
    cim_path = get_cim_path(patient_name, cim_patients)
    
    # get the path of the pointer
    ptr_path = os.path.join(ptr_files_path, ptr)

    # read the content of the image pointer
    datatype = [('series', '<i4'), ('slice', '<i4'), ('index', '<i4'), ('path', 'U255')]
    ptr_content = np.genfromtxt(ptr_path, delimiter='\t', names='series, slice, index, path', skip_header=1, dtype=datatype)

    ptr_slices = get_slices(ptr_content)

    patient_names, cim_paths, slices, bbox_corners, landmark_coords = get_data_from_h5_file(eds_h5_filepath, cim_path, patient_name, ptr_slices)

    if cim_paths is not None:
        print("Landmark coordinates found for patient {}".format(patient_name))
        print("Image pointer path: {}".format(ptr_path))
        # preprocess the dicom images 
        cine_dicom_paths, tagged_dicom_paths, cine_images, tagged_images, cine_px_spaces, tagged_px_spaces, cine_img_pos, tagged_img_pos, cine_img_orient, tagged_img_orient = prepare_dicom_images(filepaths, ptr_content, slices, view=False)
            
        # add needed data to the dataset model
        dsm.patient_names.extend(patient_names)
        dsm.cim_paths.extend(cim_paths)
        dsm.cine_dicom_paths.extend(cine_dicom_paths)
        dsm.tagged_dicom_paths.extend(tagged_dicom_paths)
        if init:
            dsm.slices = slices
            dsm.bbox_corners = bbox_corners
            dsm.landmark_coords = landmark_coords
            dsm.cine_images = cine_images
            dsm.tagged_images = tagged_images
            dsm.cine_px_spaces = cine_px_spaces
            dsm.tagged_px_spaces = tagged_px_spaces
            dsm.cine_img_pos = cine_img_pos
            dsm.tagged_img_pos = tagged_img_pos
            dsm.cine_img_orient = cine_img_orient
            dsm.tagged_img_orient = tagged_img_orient
        else:
            dsm.slices = np.append(dsm.slices, slices, axis = 0)
            dsm.bbox_corners = np.append(dsm.bbox_corners, bbox_corners, axis = 0)
            dsm.landmark_coords = np.append(dsm.landmark_coords, landmark_coords, axis = 0)
            dsm.cine_images = np.append(dsm.cine_images, cine_images, axis = 0)
            dsm.tagged_images = np.append(dsm.tagged_images, tagged_images, axis = 0)
            dsm.cine_px_spaces = np.append(dsm.cine_px_spaces, cine_px_spaces)
            dsm.tagged_px_spaces = np.append(dsm.tagged_px_spaces, tagged_px_spaces)
            dsm.cine_img_pos = np.append(dsm.cine_img_pos, cine_img_pos, axis = 0)
            dsm.tagged_img_pos = np.append(dsm.tagged_img_pos, tagged_img_pos, axis = 0)
            dsm.cine_img_orient = np.append(dsm.cine_img_orient, cine_img_orient, axis = 0)
            dsm.tagged_img_orient = np.append(dsm.tagged_img_orient, tagged_img_orient, axis = 0)
    
    else:
        log_and_print("No landmark coordinates for patient {}".format(patient_name))

def create_datasets(hf, key, dsm):
    # create group for the current set
    grp = hf.create_group(key)
    grp_cine = grp.create_group("cine")
    grp_tagged = grp.create_group("tagged")

    # converting data to numpy arrays
    patients = np.array(dsm.patient_names, dtype=object)
    cim_paths = np.array(dsm.cim_paths, dtype = object)
    cine_dicom_paths = np.array(dsm.cine_dicom_paths, dtype = object)
    tagged_dicom_paths = np.array(dsm.tagged_dicom_paths, dtype = object)

    grp.create_dataset("patients", data=patients, dtype=h5py.special_dtype(vlen=str), maxshape = (None, ))
    grp.create_dataset("cim_paths", data=cim_paths, dtype=h5py.special_dtype(vlen=str), maxshape = (None, )) 
    grp.create_dataset("slices", data=dsm.slices, maxshape = (None, ))
    grp.create_dataset("landmark_coords", data=dsm.landmark_coords, maxshape = (None, 20, 2, 168))
    grp.create_dataset("bbox_corners", data=dsm.bbox_corners, maxshape = (None, 4) )
    
    # put all the cine data in the cine group
    grp_cine.create_dataset("cine_dicom_paths", data=cine_dicom_paths, dtype=h5py.special_dtype(vlen=str), maxshape = (None, 50))
    grp_cine.create_dataset("cine_images", data=dsm.cine_images, maxshape = (None, 50, 256, 256))
    grp_cine.create_dataset("cine_px_spaces", data=dsm.cine_px_spaces, maxshape = (None, ))
    grp_cine.create_dataset("cine_image_orientations", data=dsm.cine_img_orient, maxshape = (None, 6))
    grp_cine.create_dataset("cine_image_positions", data=dsm.cine_img_pos, maxshape = (None, 3))

    # put all the tagged data in tagged group
    grp_tagged.create_dataset("tagged_dicom_paths", data=tagged_dicom_paths, dtype=h5py.special_dtype(vlen=str), maxshape = (None, 50))
    grp_tagged.create_dataset("tagged_images", data=dsm.tagged_images, maxshape = (None, 50, 256, 256))
    grp_tagged.create_dataset("tagged_px_spaces", data=dsm.tagged_px_spaces, maxshape = (None, ))
    grp_tagged.create_dataset("tagged_image_orientations", data=dsm.tagged_img_orient, maxshape = (None, 6))
    grp_tagged.create_dataset("tagged_image_positions", data=dsm.tagged_img_pos, maxshape = (None, 3))
    

def add_datasets(hf, key, dsm):
    grp = hf["//{}".format(key)]
    grp_cine = hf["//{}//cine".format(key)]
    grp_tagged = hf["//{}//tagged".format(key)]
                        
    # converting data to numpy arrays
    patients = np.array(dsm.patient_names, dtype=object)
    cim_paths = np.array(dsm.cim_paths, dtype = object)
    cine_dicom_paths = np.array(dsm.cine_dicom_paths, dtype = object)
    tagged_dicom_paths = np.array(dsm.tagged_dicom_paths, dtype = object)

    grp["patients"].resize((grp["patients"].shape[0])+patients.shape[0], axis = 0)
    grp["patients"][-patients.shape[0]:] = patients

    grp["cim_paths"].resize((grp["cim_paths"].shape[0])+cim_paths.shape[0], axis = 0)
    grp["cim_paths"][-cim_paths.shape[0]:] = cim_paths

    grp["slices"].resize((grp["slices"].shape[0])+dsm.slices.shape[0], axis = 0)
    grp["slices"][-dsm.slices.shape[0]:] = dsm.slices

    grp["landmark_coords"].resize((grp["landmark_coords"].shape[0])+dsm.landmark_coords.shape[0], axis = 0)
    grp["landmark_coords"][-dsm.landmark_coords.shape[0]:] = dsm.landmark_coords

    grp["bbox_corners"].resize((grp["bbox_corners"].shape[0])+dsm.bbox_corners.shape[0], axis = 0)
    grp["bbox_corners"][-dsm.bbox_corners.shape[0]:] = dsm.bbox_corners

    # cines
    grp_cine["cine_dicom_paths"].resize((grp_cine["cine_dicom_paths"].shape[0])+cine_dicom_paths.shape[0], axis = 0)
    grp_cine["cine_dicom_paths"][-cine_dicom_paths.shape[0]:] = cine_dicom_paths

    grp_cine["cine_images"].resize((grp_cine["cine_images"].shape[0])+dsm.cine_images.shape[0], axis = 0)
    grp_cine["cine_images"][-dsm.cine_images.shape[0]:] = dsm.cine_images

    grp_cine["cine_px_spaces"].resize((grp_cine["cine_px_spaces"].shape[0])+dsm.cine_px_spaces.shape[0], axis = 0)
    grp_cine["cine_px_spaces"][-dsm.cine_px_spaces.shape[0]:] = dsm.cine_px_spaces

    grp_cine["cine_image_orientations"].resize((grp_cine["cine_image_orientations"].shape[0])+dsm.cine_img_orient.shape[0], axis = 0)
    grp_cine["cine_image_orientations"][-dsm.cine_img_orient.shape[0]:] = dsm.cine_img_orient

    grp_cine["cine_image_positions"].resize((grp_cine["cine_image_positions"].shape[0])+dsm.cine_img_pos.shape[0], axis = 0)
    grp_cine["cine_image_positions"][-dsm.cine_img_pos.shape[0]:] = dsm.cine_img_pos

    # tagged
    grp_tagged["tagged_dicom_paths"].resize((grp_tagged["tagged_dicom_paths"].shape[0])+tagged_dicom_paths.shape[0], axis = 0)
    grp_tagged["tagged_dicom_paths"][-tagged_dicom_paths.shape[0]:] = tagged_dicom_paths

    grp_tagged["tagged_images"].resize((grp_tagged["tagged_images"].shape[0])+dsm.tagged_images.shape[0], axis = 0)
    grp_tagged["tagged_images"][-dsm.tagged_images.shape[0]:] = dsm.tagged_images

    grp_tagged["tagged_px_spaces"].resize((grp_tagged["tagged_px_spaces"].shape[0])+dsm.tagged_px_spaces.shape[0], axis = 0)
    grp_tagged["tagged_px_spaces"][-dsm.tagged_px_spaces.shape[0]:] = dsm.tagged_px_spaces

    grp_tagged["tagged_image_orientations"].resize((grp_tagged["tagged_image_orientations"].shape[0])+dsm.tagged_img_orient.shape[0], axis = 0)
    grp_tagged["tagged_image_orientations"][-dsm.tagged_img_orient.shape[0]:] = dsm.tagged_img_orient

    grp_tagged["tagged_image_positions"].resize((grp_tagged["tagged_image_positions"].shape[0])+dsm.tagged_img_pos.shape[0], axis = 0)
    grp_tagged["tagged_image_positions"][-dsm.tagged_img_pos.shape[0]:] = dsm.tagged_img_pos

def create_h5_file(filepaths, ptr_files_path, eds_h5_filepath, cim_patients, output_dir, output_filename, dataset_dict):
    '''
    I need to create three groups (train, val, test)
    Inside each group, I need to create these datasets:
        + patients (as in the header) - N number of rows
        + cim_paths (includes model and patient name in the path separated by \\) - N number of rows
        + slices (only includes the ones with landmark_coords) - N x slices
        + dicom_path (path from the cim image pointer)  - N x 50
        + cine_px_spaces
        + tagged_px_spaces
        + cine_images (pixel arrays of the cine images - all frames)    - N x  T x 256 x 256 (resized)
        + tagged_images (pixel arrays of the tagged images - all frames)    - N x T x 256 x 256 (resized)
        + landmark_coords (coordinates of the points)   - N x 20 x 2 x 168 
        + region (idk the need)
        + es_frame_idx (where the end systolic frame is) N
        + bbox corners N x 4
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
            get_all_data(dsm, filepaths, ptr_files_path, eds_h5_filepath, cim_patients, ptr)
            if len(set(dsm.patient_names)) == 5 or i == len(ptr_files)-1:   #if we have 20 unique patients added or if we have reached the end of the dictionary
                p_cnt += len(set(dsm.patient_names))
                s_cnt += len(dsm.slices)
                print("Looped through {}/{} patients for {} set".format(i+1, len(ptr_files), key))
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

                if i == len(ptr_files)-1:
                    hrs, mins, secs = calculate_time_elapsed(start)
                    log_and_print("Finished creating {} set".format(key))
                    log_and_print("Total number of unique cases: {}".format(p_cnt))
                    log_and_print("Total number of slices: {}".format(s_cnt))
                    log_and_print("Elapsed time: {} hours {} minutes {} seconds\n".format(hrs, mins, secs))

def prepare_h5_files(filepaths, ptr_files_path, ed_h5_filepath, cim_patients, output_dir, output_filename, num_cases):
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

    create_h5_file(filepaths, ptr_files_path, eds_h5_filepath, cim_patients, output_dir, output_filename, dataset_dict)


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
    ptr_files_path = "C:\\Users\\arad572\\Documents\\Summer Research\\code\\prepare_data\\img_ptrs\\new_matches_final"

    # where edward's h5py files are located
    eds_h5_filepath = "C:\\Users\\arad572\\Documents\\MR-tagging\\dataset-localNet\\data_sequence_original"

    # where the cim models are
    cim_dir = "C:\\Users\\arad572\\Downloads\\all CIM"
    cim_models = ["CIM_DATA_AB", "CIM_DATA_EL1", "CIM_DATA_EL2", "CIM_DATA_EM", "CIM_DATA_KF", "CIM_Data_ze_1", "CIM_DATA_ze_2", "CIM_DATA_ze_3", "CIM_DATA_ze_4"]
    
    # list the cim patient folders
    cim_patients = get_cim_patients(cim_dir, cim_models)
    
    # specify the number of cases we want to loop through (replace with None if you want all unique cases)
    num_cases = 50

    # where h5 files will be stored
    output_dir = os.path.join(os.getcwd(), "h5_files")

    # name of the h5file
    if num_cases is not None:
        output_filename = "UK_Biobank_{}cases.h5".format(num_cases)
    else:
        output_filename = "UK_Biobank.h5"

    try:
        prepare_h5_files(filepaths, ptr_files_path, eds_h5_filepath, cim_patients, output_dir, output_filename, num_cases)

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