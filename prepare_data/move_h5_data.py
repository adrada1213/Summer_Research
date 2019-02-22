import h5py
import os
import logging
import numpy as np
from general_functions import calculate_time_elapsed, log_and_print
from hdf5_functions import add_datasets, create_datasets
from prepare_h5_file import DataSetModel
from datetime import datetime
from time import time

'''
This function calculates the current division of data (test, train, validate) and the total
number of slices
Input:  
    h5_file = path to the h5 file
Output: 
    curr_test = current number of slices for the test group
    curr_train = current number of slices for the train group
    curr_val = current number of slices for the validation group
    total = total number of slices

TODO: Need to change the way I read the groups for efficiency - DONE
'''
def get_curent_ratio(h5_file):
    total = 0
    # read the h5 file and get the number of slices
    with h5py.File(h5_file, "r") as hf:
        curr_test = np.array(hf["/test/patients"]).size
        curr_train = np.array(hf["/train/patients"]).size
        curr_val = np.array(hf["/validation/patients"]).size
        
    total = curr_test + curr_train + curr_val   #add slices from each group to get total

    return curr_test, curr_train, curr_val, total

def calculate_targets(total, ratio):
    '''
    This function calculates the desired division of data (test, train, validate)
    Input:  Total = total number of slices
            ratio = expected ratio

    Output: test_target = target number of slices for the test group
            train_target = target number of slices for the train group
            val_target = target number of slices for the validation group
    '''
    test_target = int(total*ratio[0])
    train_target = int(total*ratio[1])
    val_target = total - (test_target+train_target)

    return test_target, train_target, val_target

def calculate_differences(curr_test, curr_train, curr_val, trgt_test, trgt_train, trgt_val):
    '''
    This function calculates the difference between the desired number of slices and the current number of slices for each group.
    (If the difference is negative, this means that the current number of slices in that data is more than the desired number of 
    slices for that data)
    Input:  curr_test = current number of slices for the test group
            curr_train = current number of slices for the train group
            curr_val = current number of slices for the validation group
            test_target = target number of slices for the test group
            train_target = target number of slices for the train group
            val_target = target number of slices for the validation group

    Output: test_diff = difference for test group
            train_diff = difference for train group
            val_diff = difference for validation group
    '''

    test_diff = trgt_test-curr_test
    train_diff = trgt_train-curr_train
    val_diff = trgt_val-curr_val

    return test_diff, train_diff, val_diff

def get_slice_count(h5_file, group):
    '''
    This function calculates the number of slices for each group
    Input:  h5_file = path to h5 file
            group = current group

    Output: slice_count = slices remaining
    '''
    with h5py.File(h5_file, "r") as hf:
        slice_count = np.array(hf["/{}/patients".format(group)]).shape[0]
    
    return slice_count

def add_data_to_model(dsm, grp, cine_grp, tagged_grp, start, end):
    '''
    This function adds the data needed to the model
    Input:  dsm = DataSetModel
            cine_grp = the cine group from h5 file
            tagged_grp = the tagged group from h5 file
            start = index of first slice
            end = index of last slice
    '''
    dsm.patient_names = np.array(grp.get("patients")[start:end])
    dsm.slices = np.array(grp.get("slices")[start:end]) 

    # cines
    dsm.cine_dicom_paths = np.array(cine_grp.get("dicom_paths")[start:end])
    dsm.cine_centroids = np.array(cine_grp.get("centroids")[start:end])
    dsm.cine_landmark_coords = np.array(cine_grp.get("landmark_coords")[start:end])
    dsm.cine_images = np.array(cine_grp.get("images")[start:end])
    dsm.cine_es_indices = np.array(cine_grp.get("es_indices")[start:end])

    # tagged set
    dsm.tagged_dicom_paths = np.array(tagged_grp.get("dicom_paths")[start:end])
    dsm.tagged_centroids = np.array(tagged_grp.get("centroids")[start:end])
    dsm.tagged_landmark_coords = np.array(tagged_grp.get("landmark_coords")[start:end])
    dsm.tagged_images = np.array(tagged_grp.get("images")[start:end])
    dsm.tagged_es_indices = np.array(tagged_grp.get("es_indices")[start:end])

def get_data_from_group(dsm, h5_file, group, index, slice_count):
    '''
    This function takes data from the current group. It gets the slices from the patient at the end
    and returns the index number of the first slice of that patient (from the right, as a positive number).
    Inputs: dsm = datasetmodel
            h5_file = path to the h5 file we;re taking data from
            group = group we're taking data from
            index = index of the first slice of the last patient taken (from the right)
            slice_count = current number of slices in that group

    Output: index = index of the first slice of the patient taken (from the right)
    '''
    with h5py.File(h5_file, "r") as hf:
        # get the groups needed
        grp = hf.get(group)
        cine_grp = grp.get("cine")
        tagged_grp = grp.get("tagged")

        # get the list of patients
        try:
            #so we don't have to read the entire patient array, slices range from 1-3 so taking 5 patients is good
            patients = np.array(grp.get("patients")[slice_count-5:slice_count])   
        except IndexError:
            patients = np.array(grp.get("patients")[:slice_count])
        patient = patients[-1] #take only one patient
        p_indices = np.array(np.where(patients==patient))[0]    #get the indices of this patient in the h5 file
        start, end = slice_count-len(p_indices), slice_count  #start and end indices
        add_data_to_model(dsm, grp, cine_grp, tagged_grp, start, end)

    return index+len(p_indices)

def create_group(dsm, new_h5_file, group):
    '''
    This function creates the group in the new h5 file
    Input:  dsm = DataSetModel containing data to be added
            new_h5_file = path to the new h5 file
            group = group we're adding
    '''
    with h5py.File(new_h5_file, "a") as hf:
        create_datasets(hf, group, dsm)

def add_data_to_group(dsm, new_h5_file, group):
    '''
    This adds data to the new h5 file
    Input:  dsm = DataSetModel containing data to be added
            new_h5_file = path to the new h5 file
            group = group to be added
    '''
    with h5py.File(new_h5_file, "a") as hf:
        add_datasets(hf, group, dsm)

def get_data_for_group(dsm, h5_file, group, start, end):
    '''
    This reads the old h5 file and gets the data for all the slices for the current patient
    Input:  h5_file = old h5 file path
            group = group that we're copying 
            start = index of the first slice for the current patient
            end = index of the last slice for the current patient
    '''
    with h5py.File(h5_file, "r") as hf:
        # get the groups needed
        grp = hf.get(group)
        cine_grp = grp.get("cine")
        tagged_grp = grp.get("tagged")

        add_data_to_model(dsm, grp, cine_grp, tagged_grp, start, end)
    
def copy_h5_data(h5_file, new_h5_file, group, init, limit):
    '''
    This function copies h5 data from old h5 file to the new h5 file
    Input:  h5_file = path to the old h5 file
            new_h5_file = path to the new h5 file
            group = group we're copying
            init = whether the h5 file is already created or not
            limit = index of the last slice we want to add (from the right)
    '''
    if init:    #to create the file
        hf = h5py.File(new_h5_file, "w")
        hf.close()

    # read the new h5 file and determine whether the group exists
    with h5py.File(new_h5_file, "r") as hf:
        if "/{}".format(group) not in hf:
            init_group = True
        else:
            init_group = False
    
    slice_cnt = get_slice_count(h5_file, group) #get the slice count
    if limit is not None:   #if the last index(from the right) is specified
        slice_cnt = slice_cnt - (limit-1) 
    for i in range(0,slice_cnt,5):    #loop through the data in small increments to not overload the memory(if we have huge h5 files)
        dsm = DataSetModel()
        if (slice_cnt - i) < 5:   #if we've reached the end of the data
            get_data_for_group(dsm, h5_file, group, i, slice_cnt)  #getting data for the group
        else:
            get_data_for_group(dsm, h5_file, group, i, i+5)
        if init_group:  #create group if it hasn't been created
            create_group(dsm, new_h5_file, group)
            init_group = False
        else:   #otherwise just append data to the group
            add_data_to_group(dsm, new_h5_file, group)

    return

def move_h5_data(h5_file, new_h5_file, ratio):
    '''
    This is the main function
    Inputs: h5_file = path to the h5 file we want to modify
            new_h5_file = path to the new h5 file we want to create
            ratio = test:train:validation ratio as a list 

    Overview:
        1. Calculates the difference between the target number of slices and the current number of slices for each group
        2. Moves data from groups (old h5 file) that have number of slices that are more than desired number of slices to 
        groups (new h5 file) that have number of slices less than the desired number of slices.
        -The movement of slices per iteration is patient based. (i.e. we move all slices for one patient from the old h5 file
        to the new h5 file)

    How data is moved/copied:
        There are three cases:
        1. Taking data from 2 groups and adding to 1 group
        2. Adding data to 2 groups from 1 group
        3. Adding data from 1 group to 1 group (one group unchanged)

        1. Copy the group/s we're adding to from the old h5 file to the new h5 file
        2. Loop through the groups we're taking data from
            -Get all the data for all slices from a single patient(starting from the end)
            -Track the index of the first slice of that patient (this becomes our new index for the group we're taking data from)
            -Add the data to the group we're adding to in the new h5 file
            -Update the new number of slices for this group
            -If current group reaches the target number of slices, copy the data from this group (only up until the end index) to
            to the new h5 file
            -Move on to the next group

        Second case(Adding data to 2 groups from 1 group):
        Difference from the first case: 
        -copies from both of the groups we're adding data to to the new h5 file
        -loops through the groups we're adding data to instead of group we're taking data from
        -Once all target number of slices met, we copy data from the group we're getting data from (only up until the end index)

        Third case (Adding data to 1 group from one group):
        Difference from the first case:
        -copies the data from unchanged group, and group we're adding to to the new h5 file
        -no loops
    '''

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
    # First case -> if we're moving data from 2 groups to 1 group
    if len(grp_from) > len(grp_to): 
        print("Moving data from {} & {} to {}".format(grp_from[0], grp_from[1], grp_to[0]))
        init = True #true means the h5 file hasn't been created yet
        # copy the data in the receiving group from the old h5 file to the new h5 file
        copy_h5_data(h5_file, new_h5_file, grp_to[0], init, None)   
        init = False
        # loop through the groups that we're taking data from
        for grp in grp_from:
            print("From {} to {}".format(grp, grp_to[0]))
            # index keeps track of the last index of slice we're copying from the group we're taking data from 
            # in the old h5 file to the group in the new h5 file
            index = 1   
            slice_cnt = get_slice_count(h5_file, grp)   #get the current number of slices
            # loop until the number of slices is equal to or less than the target number of slices for that group
            while slice_cnt > trgt_dict[grp]:
                dsm = DataSetModel()    #initiate the model
                index = get_data_from_group(dsm, h5_file, grp, index, slice_cnt)   #get data from group from old h5 file
                print("Moving {} slices for patient {}".format(len(dsm.slices), dsm.patient_names[0]))
                add_data_to_group(dsm, new_h5_file, grp_to[0])  #add data to new h5 file
                slice_cnt -= len(dsm.slices)    #update the number of slices of the group we're taking data from
            copy_h5_data(h5_file, new_h5_file, grp, init, index)    #if target number of slices is reached, we copy the data from the current group to the new h5 file
    
    # Second case ->  if we're moving data to 2 groups from 1 group
    elif len(grp_to) > len(grp_from):  
        print("Moving data to {} & {} from {}".format(grp_to[0], grp_to[1], grp_from[0]))
        init = True #true means the h5 file hasn't been created yet

        # copy the data in the receiving groups from the old h5 file to the new h5 file
        copy_h5_data(h5_file, new_h5_file, grp_to[0], init, None)
        init = False
        copy_h5_data(h5_file, new_h5_file, grp_to[1], init, None)

        # index keeps track of the last index of slice we're copying from the group we're taking data from 
        # in the old h5 file to the group in the new h5 file
        index = 1
        slice_cnt_from = get_slice_count(h5_file, grp_from[0])   #get the current number of slices of group we're taking data from
        # loop through the groups we're adding data to
        for grp in grp_to:
            print("From {} to {}".format(grp_from[0], grp))
            slice_cnt = get_slice_count(h5_file, grp)   #get the current number of slices of the group we're adding to
            # loop until we get the desired number of slices for the receiving group
            while slice_cnt < trgt_dict[grp]:   #if the number of slices for the current group reaches the target, we move on to the next group
                dsm = DataSetModel()     #initiate the model
                index = get_data_from_group(dsm, h5_file, grp_from[0], index, slice_cnt_from)   #get data from group we're getting data from
                print("Moving {} slices for patient {}".format(len(dsm.slices), dsm.patient_names[0]))  
                add_data_to_group(dsm, new_h5_file, grp)    #add data to new h5 file
                # update the number of slices for both receiving, and giving groups
                slice_cnt += len(dsm.slices)
                slice_cnt_from -= len(dsm.slices)
        #if target number of slices is reached, we copy the data from the current group to the new h5 file
        copy_h5_data(h5_file, new_h5_file, grp_from[0], init, index)    

    # Third case -> if we're moving data from 1 group to another
    else:   #if one of the groups contains the target amount of data
        print("Moving data from {} to {}".format(grp_from[0], grp_to[0]))
        # get the group that already reached its target number of slices
        unchanged_grp = [g for g in groups if g not in grp_to and g not in grp_from]
        print("{} group unchanged".format(unchanged_grp[0]))

        # copying data
        init = True #true means the h5 file hasn't been created yet
        copy_h5_data(h5_file, new_h5_file, unchanged_grp[0], init, None)    #just copy the unchanged group from old h5 file to new h5 file
        init = False
        copy_h5_data(h5_file, new_h5_file, grp_to[0], init, None)   #copy the group we're adding data to

        # index keeps track of the last index of slice we're copying from the group we're taking data from 
        # in the old h5 file to the group in the new h5 file
        index = 1
        slice_cnt = get_slice_count(h5_file, grp_from[0])   #get the current number of slices in the giving group
        while slice_cnt > trgt_dict[grp_from[0]]:   #move slices until we reach the target number of slices
            dsm = DataSetModel()    #initiate the model
            index = get_data_from_group(dsm, h5_file, grp_from[0], index, slice_cnt)
            print("Moving {} slices for patient {}".format(len(dsm.slices), dsm.patient_names[0]))
            add_data_to_group(dsm, new_h5_file, grp_to[0])  #add data to new h5 file
            slice_cnt -= len(dsm.slices)    #update number of slices
        copy_h5_data(h5_file, new_h5_file, grp_from[0], init, index)


    # get new ratio/division
    curr_test, curr_train, curr_val, total = get_curent_ratio(new_h5_file)
    print("New data distribution -> Test: {} Train: {} Validation: {}".format(curr_test, curr_train, curr_val))
    print("Total: ", total)

if __name__ == "__main__":
    '''
    Things to modify:
    input_dir = where the old h5 file you want to modify is located
    h5_filename = filename of you h5 file

    output_dir = where you want the new h5 file to be stored
    new_h5_filename = what you want the new h5 file to be called (must be a different name from the old one)

    ratio = how you want your data to be divided (must be equal to 1) [test, train, validation]

    del_old = if you want to delete the old h5 file
    '''
    # specify where the h5 file is located
    input_dir = "C:\\Users\\arad572\\Documents\\Summer Research\\Summer Research Code\\prepare_data\\h5_files"
    h5_filename = "UK_Biobank.h5"   #specify the filename
    h5_file = os.path.join(input_dir, h5_filename) 

    # specify where you want to put the new h5 file
    output_dir = "F:\\cine-machine-learning\\dataset"
    #output_dir = "C:\\Users\\arad572\\Documents\\Summer Research\\Summer Research Code\\prepare_data\\h5_files"
    new_h5_filename = "{}_new.h5".format(h5_filename.replace(".h5", ""))   #specify the new filename
    new_h5_file = os.path.join(output_dir, new_h5_filename)
        
    # specify how you want to divide the data [test, train, validation], total must equal to 1
    ratio = [0.2, 0.6, 0.2]

    # change to False if you don't want to keep the old h5 file
    del_old = False

    if h5_file == new_h5_file:
        print("Please specify a different output filename/path")
    elif sum(ratio) != 1:
        print("Sum of ratios must equal 1")
    else:
        print("Moving h5 data from {} to {}".format(os.path.basename(h5_file), os.path.basename(new_h5_file)))
        print("h5 files in {}".format(os.path.dirname(h5_file)))

        move_h5_data(h5_file, new_h5_file, ratio)   #move data

        if del_old: #if we don't want to keep the old file(saves space)
            print("Deleting {}".format(h5_filename))
            os.remove(h5_file)
            print("Renaming {} to {}".format(new_h5_filename, h5_filename))
            os.rename(os.path.join(output_dir, new_h5_filename), os.path.join(output_dir, h5_filename)) #renaming to old filename
            print("New h5 file stored in: {}".format(os.path.join(output_dir, h5_file)))
        
        print("Finished")
