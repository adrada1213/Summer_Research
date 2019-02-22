"""
This script contains functions needed to handle the files in the cim folders
Author: Amos Rada
Date:   22/02/2019
"""
# import needed libraries
import os
import numpy as np
# import functions from created scripts
from general_functions import log_error_and_print

def get_cim_patients(cim_dir, cim_models):
    '''
    This functions compiles all the paths of the cim patients in a list
    Inputs: 
        cim_dir (string) = directory to the cim models
        cim_models (list of strings) = name of the folders of the cim models
    Output: 
        cim_patients (list of strings) = list of all cim patient paths 
    '''
    # put the paths of the models in a list
    cim_models_paths = [os.path.join(cim_dir, d) for d in os.listdir(cim_dir) if d in cim_models]
    cim_patients = []
    # obtain the cim path of each patient
    for cim_model in cim_models_paths:
        cim_patients += [os.path.join(cim_model, d) for d in os.listdir(cim_model) if os.path.isdir(os.path.join(cim_model, d))]
    
    return cim_patients

def get_cim_path(patient_name, cim_patients):
    '''
    This function extracts the cim path of the patient
    Inputs: 
        patient_name (string) = name of patient from dicom header (with underscore)
        cim_patients (list of strings) = paths to the cim patients folder
    Output: 
        cim_path (string) = returns the path to the cim folder for the patient
    '''
    if patient_name != "4J_Y5_B5__XN":  #unique case where this patient has two underscores
        cim_ptr_path = [p for p in cim_patients if patient_name.replace("_Bio", "").lower() in p.lower()][0] #get the cim path of current patient
    else:    
        cim_ptr_path = [p for p in cim_patients if "4J_Y5_B5_XN".replace("_Bio", "").lower() in p.lower()][0]

    return cim_ptr_path

def get_pointer_paths(cim_dir):
    '''
    This function gets the image pointer paths of the patients from the cim directory
    Input:  
        cim_dir (string) = directory of the cim model/observer we want to get paths to the image pointers for
    Output: 
        ptr_files (list of strings) = paths to the image pointer files
    '''
    # list the patient folders in the current observer
    patient_folders = os.listdir(cim_dir)
    # initialise list containing paths to the image pointer files
    ptr_files = []
    # loop through the patients
    for patient_name in patient_folders:
        # image pointer files are in the system folder of the patient
        system_dir = os.path.join(cim_dir, patient_name, "system")
        try:
            files = [f for f in os.listdir(system_dir) if f.endswith(".img_imageptr")]
        except FileNotFoundError:
            log_error_and_print("The system cannot find the path specified {}".format(system_dir))
            continue
        try:
            # add to list
            ptr_files.append(os.path.join(system_dir, files[0]))
        except IndexError:
            log_error_and_print("No image pointer file for {} | CIM dir: {}".format(patient_name, os.path.join(cim_dir, patient_name)))
            continue
        files = []
    
    return ptr_files

def get_global_landmarks(cim_path, slice_num):
    '''
    This function extracts the global landmarks (3D/coordinate system of the MRI) from the model folders of CIM
    Inputs:
        cim_path (string) = path to the cim folder of the patient
        slice_num (int) = number of the slice we're interested in (start from 0)
    Output:
        global_landmarks (3x168 list) = 3D coordinates of the landmarks 
    '''
    # get the path containing the strain.dat files
    model_path = os.path.join(cim_path, "model_{}".format(os.path.basename(cim_path)), "series_1_slice_{}".format(slice_num+1))

    #way check if there is a strain file (contains landmarks), if there's none, there will be an error so make sure to catch it using try, except
    _ = [d for d in os.listdir(model_path) if d.endswith("_strain.dat")][0] 

    # initialise the list that will contain the 3D landmarks
    global_landmarks = []
    # loop from 1 to 20 (usual number of tagged frames)
    for i in range(1,21):
        try:
            # get the strain.dat file
            landmark_file = [d for d in os.listdir(model_path) if d.endswith("_{}_samplePt_strain.dat".format(i))][0]
        except: #to handle cases where there are less than 20 frames
            log_error_and_print("No landmark coordinates for slice {} frame {} CIM Path {}".format(slice_num, i, cim_path))
            continue
        filepath = os.path.join(model_path, landmark_file)
        # read the strain.dat file and extract the global coordinates
        data = np.genfromtxt(filepath, delimiter=' ', names=True)
        points_3D = (data['patientX'], data['patientY'], data['patientZ'])
        # add the landmarks to the list
        global_landmarks.append(points_3D)
    
    # if there are less than 20 frames, add the last set of global landmarks until we have 20 sets of global landmarks
    # (This is to ensure we have a uniform nmber of global landmarks because h5 files can't contain an array with 
    # non-uniform shape of arrays)
    if len(global_landmarks) != 20:
        for i in range(20-len(global_landmarks)):
            global_landmarks.append(points_3D)

    return global_landmarks

def get_es_index(cim_path, slice_num):
    '''
    This functions extracts the index of the end systolic frame of tagged slice from the strain.txt file in the patient model folder
    Inputs:
        cim_path (string) = path to the cim folder of the patient
        slice_num (int) = number of the slice we're interested in (start from 0)
    Output:
        es_index = index of the end sytolic frame
    '''
    # get the path containing the strain.txt file
    model_path = os.path.join(cim_path, "model_{}".format(os.path.basename(cim_path)), "series_1_slice_{}".format(slice_num+1))
    es_file = os.path.join(model_path, "{}_strain.txt".format(os.path.basename(cim_path)))

    # open the file
    with open(es_file) as f:    
        for line in f:
            if "ES Frame" in line:
                # get the index of the end systole frame
                es_index =  int(line.split(":")[1].strip()) - 1 #frame number in our image pointer starts from 0 but, frame number in the txt file starts from 1
                break
    
    return es_index

# ========== testing the functions ==========
if __name__ == "__main__":
    # where the cim models are
    cim_dir = "C:\\Users\\arad572\\Downloads\\all CIM"
    cim_models = ["CIM_DATA_AB", "CIM_DATA_EL1", "CIM_DATA_EL2", "CIM_DATA_EM", "CIM_DATA_KF", "CIM_Data_ze_1", "CIM_DATA_ze_2", "CIM_DATA_ze_3", "CIM_DATA_ze_4"]
    
    # list the cim patient folders
    cim_patients = get_cim_patients(cim_dir, cim_models)

    #print(cim_patients[0])

    # get the cim path for the patient MODEL\PatientName
    cim_path = get_cim_path("8M_TP_2A_82_Bio", cim_patients)

    print(cim_path)

    global_landmarks = get_global_landmarks(cim_path, 0)
    #es_index = get_es_index(cim_path, 1)
    #print(es_index)
    print(len(global_landmarks))
    #print(global_landmarks[1])