"""
This script contains functions to handle files in the LVModellerFormatV2 folder.
(ref code: based on the code kat gilbert gave me)
Author: Amos Rada
Date:   25/02/2019
"""
# import libraries
import os
import numpy as np
import csv
# import from created scripts
from general_functions import log_error_and_print

def get_uids(slice_info):
    '''
    This function extracts the uids from the SliceInfoFile
    Input:
        slice_info (string) = path to the SliceInfoFile.txt file
    Output:
        uids (list) = list of uids (id to identify dicom images)
    '''
    # open the slice info text file
    with open(slice_info) as f:
        uids = []   #initialise list
        for line in f:
            if "SOPInstanceUID" in line:
                # get the uid
                uid =  line.split(":")[1].strip()
                line = f.readline()

                # append to list
                uids.append(uid)
    
    return uids

def get_cine_es_index(LVModel_path, f_id, ptr_content, slice_num):
    '''
    This function extracts the indices of the end systolic frame of the cine slices
    of a patient.
    Inputs:
        LVModel_path (string) = path to the LVModellerFormatV2 folder
        f_id (string) = folder id of the patient (based on the confidential mapping file)
        ptr_content (array) = content of the image pointer of the patient
        slice_num (int) = slice we're interested in
    Output:
        es_index (int) = index of the end sytolic frame of the patient slice
    '''
    # get the folder path
    f_path = os.path.join(LVModel_path, f_id)

    # get the filepaths for the file containing  the slice info
    es_slice_info = os.path.join(f_path, "ES", "SliceInfoFile.txt")

    # read the ed and es slice info file and get the uids and slice ids
    try:
        es_uids = get_uids(es_slice_info)
    except FileNotFoundError:
        log_error_and_print("SliceInfoFile not found. Pat ID: {}".format(f_id))
        return -1

    # get the all cine frames
    cine_frames = ptr_content[ptr_content["series"] == 0]
    # get the cine frames for the slice of interest
    frames = cine_frames[cine_frames["slice"]==slice_num]

    es_index = -1   #we set es_index to -1 in case es index is not found
    for i, frame in enumerate(frames):
        uid = os.path.basename(frame["path"].replace(".dcm", ""))
        #if the uid of the current frame matches a uid from the es slice info file, then that frame is an es frame
        if uid in es_uids and i != 0:
            es_index = frame["frame"]
            break

    return es_index

def read_mapping_file(mapping_file):
    '''
    This function extracts the folder ids and the patient ids from the the confidential mapping file.
    Input:
        mapping_file (string) = path to the mapping file
    Outputs:
        f_ids (list of strings) = list of the folder ids
        p_ids (list of strings) = list of the patient ids
    '''
    with open(mapping_file, mode="r") as csv_file:
        f_ids = []
        p_ids = []
        csv_reader = csv.reader(csv_file, delimiter=',')

        for i, row in enumerate(csv_reader):
            if i != 0:
                f_ids.append(row[0][1:])
                p_ids.append(row[1][:8])

    return f_ids, p_ids

def get_folder_id(f_ids, p_ids, pat_id):
    '''
    This function gets the folder id of a patient based on their patient id
    Inputs:
        f_ids (list of strings) = list of folder ids from the mapping file
        p_ids (list of strings) = list of patient ids from the mapping file
        pat_id (list) = id of patient we're interested in (usually the patient name without the _ and the "bio" at the end)
    Output:
        f_id (string) = folder id of the patient
    '''
    # get the matching folder of current patient
    p_index = p_ids.index(pat_id)
    f_id = f_ids[p_index]

    return f_id

