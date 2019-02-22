
# import libraries
import os
import numpy as np
from general_functions import log_error_and_print
import csv

def get_uids(slice_info):
    with open(slice_info) as f:
        uids = []
        for line in f:
            if "SOPInstanceUID" in line:
                # get the uid and corresponding slice id
                uid =  line.split(":")[1].strip()
                line = f.readline()

                # append to list
                uids.append(uid)
    
    return uids

def get_cine_es_indices(LVModel_path, f_id, ptr_content, slices):
    # initialise the variables we need to return
    es_indices = []

    # get the folder path
    f_path = os.path.join(LVModel_path, f_id)

    # get the filepaths for the file containing  the slice info
    es_slice_info = os.path.join(f_path, "ES", "SliceInfoFile.txt")

    # read the ed and es slice info file and get the uids and slice ids
    try:
        es_uids = get_uids(es_slice_info)
    except FileNotFoundError:
        log_error_and_print("SliceInfoFile not found. Pat ID: {}".format(f_id))
        return np.array([-1]*len(slices))

    # get the cine frames
    cine_frames = ptr_content[ptr_content["series"] == 0]

    # loop through each slice in the image pointer
    for sl in slices:
        # loop through the frames in that slice
        frames = cine_frames[cine_frames["slice"]==sl]
        for j, frame in enumerate(frames):
            uid = os.path.basename(frame["path"].replace(".dcm", ""))
            if uid in es_uids and j != 0:
                #print(uid, es_uids)
                # add to list
                es_indices.append(frame["index"])
                break
    
            if j == len(frames)-1:
                es_indices.append(-1)
    
    return np.array(es_indices)

def get_cine_es_index(LVModel_path, f_id, ptr_content, slice_num):
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

    # get the cine frames
    cine_frames = ptr_content[ptr_content["series"] == 0]
    frames = cine_frames[cine_frames["slice"]==slice_num]

    # loop through the frames
    es_index = -1
    for i, frame in enumerate(frames):
        uid = os.path.basename(frame["path"].replace(".dcm", ""))
        if uid in es_uids and i != 0:
            # add to list
            es_index = frame["frame"]
            break

    return es_index

def read_mapping_file(mapping_file):
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
    # get the matching folder of current patient
    
    p_index = p_ids.index(pat_id)
    f_id = f_ids[p_index]

    return f_id

