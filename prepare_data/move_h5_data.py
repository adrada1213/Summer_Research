import h5py
import os
import logging
import numpy as np
from prepare_data_functions import calculate_time_elapsed, log_and_print
from prepare_h5_files import add_datasets, create_datasets, DataSetModel
from datetime import datetime
from time import time


def get_curent_ratio(h5_file):
    total = 0
    with h5py.File(h5_file, "r") as hf:
        curr_test = np.array(hf["/test/patients"]).size
        curr_train = np.array(hf["/train/patients"]).size
        curr_val = np.array(hf["/validation/patients"]).size
        
    total = curr_test + curr_train + curr_val

    return curr_test, curr_train, curr_val, total

def calculate_targets(total, ratio):
    test_target = int(total*ratio[0])
    train_target = int(total*ratio[1])
    val_target = total - (test_target+train_target)

    return test_target, train_target, val_target

# if the difference is negative, this means that the current number of slices in that data is more than the desired number of slices for that data
def calculate_differences(curr_test, curr_train, curr_val, trgt_test, trgt_train, trgt_val):
    test_diff = trgt_test-curr_test
    train_diff = trgt_train-curr_train
    val_diff = trgt_val-curr_val

    return test_diff, train_diff, val_diff

def get_slice_count(h5_file, group):
    with h5py.File(h5_file, "r") as hf:
        slice_count = np.array(hf["/{}/patients".format(group)]).shape[0]
    
    return slice_count

def get_data_from_group(dsm, h5_file, group, index):
    with h5py.File(h5_file, "r") as hf:
        # get the groups needed
        grp = hf.get(group)
        cine_grp = grp.get("cine")
        tagged_grp = grp.get("tagged")

        # get the list of patients
        patients = np.array(grp.get("patients"))
        patient = patients[-index] #take only one patient
        p_indices = np.array(np.where(patients==patient))[0]    #get the indices of this patient in the h5 file
        start, end = p_indices[0], p_indices[-1]+1  #start and end indices
        dsm.patient_names = patients[start:end]
        dsm.slices = np.array(grp.get("slices"))[start:end] 
        dsm.landmark_coords = np.array(grp.get("landmark_coords"))[start:end,:,:,:]
        dsm.bbox_corners = np.array(grp.get("bbox_corners"))[start:end,:]
        dsm.cim_paths = np.array(grp.get("cim_paths"))[start:end]

        # cines
        dsm.cine_dicom_paths = np.array(cine_grp.get("cine_dicom_paths"))[start:end, :]
        dsm.cine_img_orient = np.array(cine_grp.get("cine_image_orientations"))[start:end,:]
        dsm.cine_img_pos = np.array(cine_grp.get("cine_image_positions"))[start:end,:]
        dsm.cine_images = np.array(cine_grp.get("cine_images"))[start:end,:,:,:]
        dsm.cine_px_spaces = np.array(cine_grp.get("cine_px_spaces"))[start:end] 

        # tagged
        dsm.tagged_dicom_paths = np.array(tagged_grp.get("tagged_dicom_paths"))[start:end, :]
        dsm.tagged_img_orient = np.array(tagged_grp.get("tagged_image_orientations"))[start:end,:]
        dsm.tagged_img_pos = np.array(tagged_grp.get("tagged_image_positions"))[start:end,:]
        dsm.tagged_images = np.array(tagged_grp.get("tagged_images"))[start:end,:,:,:]
        dsm.tagged_px_spaces = np.array(tagged_grp.get("tagged_px_spaces"))[start:end] 
    
    return index+len(p_indices)

def create_group(dsm, new_h5_file, group):
    with h5py.File(new_h5_file, "a") as hf:
        create_datasets(hf, group, dsm)

def add_data_to_group(dsm, h5_file, group):
    with h5py.File(h5_file, "a") as hf:
        add_datasets(hf, group, dsm)

def get_data_for_group(h5_file, group, start, end):
    dsm = DataSetModel()
    with h5py.File(h5_file, "r") as hf:
        # get the groups needed
        grp = hf.get(group)
        cine_grp = grp.get("cine")
        tagged_grp = grp.get("tagged")

        # get the list of patients
        dsm.patient_names = np.array(grp.get("patients"))[start:end]
        dsm.slices = np.array(grp.get("slices"))[start:end] 
        dsm.landmark_coords = np.array(grp.get("landmark_coords"))[start:end,:,:,:]
        dsm.bbox_corners = np.array(grp.get("bbox_corners"))[start:end,:]
        dsm.cim_paths = np.array(grp.get("cim_paths"))[start:end]

        # cines
        dsm.cine_dicom_paths = np.array(cine_grp.get("cine_dicom_paths"))[start:end, :]
        dsm.cine_img_orient = np.array(cine_grp.get("cine_image_orientations"))[start:end,:]
        dsm.cine_img_pos = np.array(cine_grp.get("cine_image_positions"))[start:end,:]
        dsm.cine_images = np.array(cine_grp.get("cine_images"))[start:end,:,:,:]
        dsm.cine_px_spaces = np.array(cine_grp.get("cine_px_spaces"))[start:end] 

        # tagged
        dsm.tagged_dicom_paths = np.array(tagged_grp.get("tagged_dicom_paths"))[start:end, :]
        dsm.tagged_img_orient = np.array(tagged_grp.get("tagged_image_orientations"))[start:end,:]
        dsm.tagged_img_pos = np.array(tagged_grp.get("tagged_image_positions"))[start:end,:]
        dsm.tagged_images = np.array(tagged_grp.get("tagged_images"))[start:end,:,:,:]
        dsm.tagged_px_spaces = np.array(tagged_grp.get("tagged_px_spaces"))[start:end] 
    
    return dsm

def copy_h5_data(h5_file, new_h5_file, group, init, limit):
    if init:
        hf = h5py.File(new_h5_file, "w")
        hf.close()

    with h5py.File(new_h5_file, "r") as hf:
        if "/{}".format(group) not in hf:
            init_group = True
        else:
            init_group = False
    
    slice_cnt = get_slice_count(h5_file, group)
    if limit is not None:
        slice_cnt = slice_cnt - (limit-1)
    for i in range(0,slice_cnt,5):    #loop through the data in small increments to not overload the memory(if we have huge h5 files)
        if (slice_cnt - i) < 5:   #if we've reached the end of the data
            dsm = get_data_for_group(h5_file, group, i, slice_cnt)
        else:
            dsm = get_data_for_group(h5_file, group, i, i+5)
        if init_group:
            create_group(dsm, new_h5_file, group)
            init_group = False
        else:
            add_data_to_group(dsm, new_h5_file, group)

    return

'''
This is the main function
Inputs: h5_file = path to the h5 file we want to modify
        new_h5_file = path to the new h5 file we want to create
        ratio = test:train:validate ratio as a list 

Overview:
    1. Calculates the difference between the target number of slices and the current number of slices for each group
    2. Moves data from groups (old h5 file) that have number of slices that are more than desired number of slices to 
    groups (new h5 file) that have number of slices less than the desired number of slices.

How data is moved/copied:
    There are three cases:
    1. Taking data from 2 groups and adding to 1 group
    2. Adding data to 2 groups from 1 group
    3. Adding data from 1 group to 1 group (one group unchanged)

    First case(Taking data from 2 groups and adding to 1 group):
    1. Copy the group we're adding to from the old h5 file to the new h5 file
    2. Loop through the groups we're taking data from
        -Get all the data for all slices from a single patient(starting from the end)
        -Track the index of the first slice of that patient (this becomes our new index for the group we're taking data from)
        -Add the data to the group we're adding to in the new h5 file
        -Update the new number of slices for this group
        -If current group reaches the target number of slices, copy the data from this group (only up until the end index) to
        to the new h5 file
        -Move on to the next group

    Second case(Adding data to 2 groups from 1 group):
    Difference from the first case: We're copying the group we're getting 
    

'''
def move_h5_data(h5_file, new_h5_file, ratio):
    # groups in the h5 file
    groups = ["train", "test", "validation"]

    # get the current number of slices in each group and the total as well
    curr_test, curr_train, curr_val, total = get_curent_ratio(h5_file)
    print("Current data distribution -> Test: {} Train: {} Validation: {}".format(curr_test, curr_train, curr_val))
    print("Total: ", total)

    # calculate the target number of data we want for each group using the total and the ratio
    trgt_test, trgt_train, trgt_val = calculate_targets(total, ratio)
    trgt_dict = {"test": trgt_test, "train": trgt_train, "validation": trgt_val}    #create a dictionary for the values calculated

    # calculate the difference between the current number of slices in each group and the target number of slices for each group
    test_diff, train_diff, val_diff = calculate_differences(curr_test, curr_train, curr_val, trgt_test, trgt_train, trgt_val)
    diff_dict = {"test": test_diff, "train": train_diff, "validation": val_diff}    #create a dictionary for the values calculated

    # determine the group(s) we're taking data from, and the group(s) receiving the data using the differences
    grp_from = []
    grp_to = []
    # loop through the difference calculated from before
    for grp, diff in diff_dict.items():
        if diff > 0: #if target is larger, we need to add data to this group, therefore we;re gonna add the group that receives data
            grp_to.append(grp)
        elif diff < 0:  #vice versa
            grp_from.append(grp)

    # start moving data
    if len(grp_from) > len(grp_to): #if we're moving data from 2 groups to 1 group
        print("Moving data from {} & {} to {}".format(grp_from[0], grp_from[1], grp_to[0]))
        init = True #true means the h5 file hasn't been created yet
        # copy the data in the receiving group from the old h5 file to the new h5 file
        copy_h5_data(h5_file, new_h5_file, grp_to[0], init, None)   
        init = False
        # loop through the groups that we're taking data from
        for grp in grp_from:
            #index keeps track of the number of data we're copying from the from group in the old h5 file to the group in the new h5 file
            index = 1   
            slice_cnt = get_slice_count(h5_file, grp)   #get the current number of slices
            # loop until the number of slices is equal to or less than the target number of slices for that group
            while slice_cnt > trgt_dict[grp]:
                dsm = DataSetModel()    #initiate the model
                index = get_data_from_group(dsm, h5_file, grp, index)   #get data from group from old h5 file
                add_data_to_group(dsm, new_h5_file, grp_to[0])  #add data to new h5 file
                slice_cnt -= len(dsm.slices)    #update the number of slices of the group we're taking data from
            copy_h5_data(h5_file, new_h5_file, grp, init, index)    #if target number of slices is reached, we copy the data from the current group to the new h5 file
    
    elif len(grp_to) > len(grp_from):   #if we're moving data to 2 groups from 1 group
        print("Moving data to {} & {} from {}".format(grp_to[0], grp_to[1], grp_from[0]))
        init = True #true means the h5 file hasn't been created yet
        # copy the data in the receiving groups from the old h5 file to the new h5 file
        copy_h5_data(h5_file, new_h5_file, grp_to[0], init, None)
        init = False
        copy_h5_data(h5_file, new_h5_file, grp_to[1], init, None)
        # loop through the groups we're adding data to
        for grp in grp_to:
            #index keeps track of the number of data we're copying from the from group in the old h5 file to the group in the new h5 file
            index = 1
            slice_cnt = get_slice_count(h5_file, grp)   #get the current number of slices
            # loop until we get the desired number of slices
            while slice_cnt < trgt_dict[grp]:   #if the number of slices for the current group reaches the target, we move on to the next group
                dsm = DataSetModel()     #initiate the model
                index = get_data_from_group(dsm, h5_file, grp_from[0], index)  
                add_data_to_group(dsm, new_h5_file, grp)    #add data to new h5 file
                slice_cnt += len(dsm.slices)    #update the number of slices
        #if target number of slices is reached, we copy the data from the current group to the new h5 file
        copy_h5_data(h5_file, new_h5_file, grp_from[0], init, index)    

    else:   #if one of the groups contains the target amount of data
        print("Moving data from {} to {}".format(grp_from[0], grp_to[0]))
        # get the group that already reached its target number of slices
        unchanged_grp = [g for g in groups if g not in grp_to and g not in grp_from]
        print("{} group unchanged".format(unchanged_grp[0]))
        init = True #true measn the h5 file hasn't been created yet
        copy_h5_data(h5_file, new_h5_file, unchanged_grp[0], init, None)    #just copy the unchanged group from old h5 file to new h5 file
        init = False
        copy_h5_data(h5_file, new_h5_file, grp_to[0], init, None)   #copy the group we're adding data to
        index = 1
        slice_cnt = get_slice_count(h5_file, grp_from[0])
        while slice_cnt > trgt_dict[grp_from[0]]:
            dsm = DataSetModel()
            index = get_data_from_group(dsm, h5_file, grp_from[0], index)
            add_data_to_group(dsm, new_h5_file, grp_to[0])
            slice_cnt -= len(dsm.slices)
        copy_h5_data(h5_file, new_h5_file, grp_from[0], init, index)

    curr_test, curr_train, curr_val, total = get_curent_ratio(new_h5_file)
    print("New data distribution -> Test: {} Train: {} Validation: {}".format(curr_test, curr_train, curr_val))
    print("Total: ", total)

if __name__ == "__main__":
    # specify where the h5 file is located
    input_dir = "C:\\Users\\arad572\\Documents\\Summer Research\\code\\prepare_data\\h5_files"
    h5_filename = "UK_Biobank_50cases.h5"   #specify the filename
    h5_file = os.path.join(input_dir, h5_filename) 

    # specify where you want to put the new h5 file
    output_dir = "C:\\Users\\arad572\\Documents\\Summer Research\\code\\prepare_data\\h5_files"
    new_h5_filename = "UK_Biobank_50cases_new.h5"   #specify the new filename
    new_h5_file = os.path.join(output_dir, new_h5_filename)

    # specify how you want to divide the data [test, train, validation]
    ratio = [0.2, 0.6, 0.2]

    # change to False if you don't want to keep the old h5 file
    del_old = True
    print("Moving h5 data from {} to {}".format(os.path.basename(h5_file), os.path.basename(new_h5_file)))
    print("h5 files in {}".format(os.path.dirname(h5_file)))

    move_h5_data(h5_file, new_h5_file, ratio)   #move data

    if del_old: #if we don't want to keep the old file(saves space)
        print("Deleting {}".format(h5_filename))
        os.remove(h5_file)
        print("Renaming {} to {}".format(new_h5_filename, h5_filename))
        os.rename(os.path.join(output_dir, new_h5_filename), os.path.join(output_dir, h5_filename))
        print("New h5 file stored in: {}".format(os.path.join(output_dir, h5_file)))
    
    print("Finished")