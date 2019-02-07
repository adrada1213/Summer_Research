"""
This python file contains functions needed to handle image pointers
Author: Amos Rada
Date:   16/01/2019
"""
# import needed libraries
import pydicom
import numpy as np
import os
from time import time
from prepare_data_functions import log_and_print, log_error_and_print, calculate_time_elapsed


def load_ptr_content(ptr_path):
    '''
    This functions loads the content of the image pointer
    Input:  
        ptr_path = path to the image pointer we want to load

    Output: 
        ptr_content = content of the image pointer as a numpy array
    '''
    # read the content of the image pointer
    datatype = [('series', '<i4'), ('slice', '<i4'), ('index', '<i4'), ('path', 'U255')]
    ptr_content = np.genfromtxt(ptr_path, delimiter='\t', names='series, slice, index, path', skip_header=1, dtype=datatype)

    return ptr_content

def get_slices(ptr_content):
    '''
    This function gets the slices (e.g. 0 1 2) from an image pointer
    Input:  
        ptr_content = content of the current pointer (can read using np.genfromtxt)

    Output: 
        slices = the slices as an np.array 
    '''
    slice_condition = np.logical_and(ptr_content["series"] == 0, ptr_content["index"] == 0) #condition to get only one frame for each slice
    slices = ptr_content[slice_condition]["slice"]

    return slices

def create_new_image_pointer(filepath, pat_name, image_path, gen_image_path, sax_cine_prefix, tagged_cine_prefix, output_dir, suffix):
    '''
    TODO: Instead of taking the slice number from the series description, get the slice number from the series number

    This function is used to create image pointers from dicom files.
    The image pointer file will have a header that describes each column in pointer file
    Column 1 - Series Number (0-cine, 1-tagged), Column 2 - Slice Number, Column 3 - Index (which frame the image is in the current slice), Column 4 - general image path
    Inputs:
        filepath(string) = where the multipatient folders are stored
        pat_name(string) = name of the patient (from dicom header)
        image_path(string) = path to the .dcm files
        gen_image_path(string) = general path to the .dcm files (the filepath in the string is replace by "IMAGEPATH")
        sax_cine_prefix(string) = what the normal cine series name starts with
        tagged_cine_prefix(string) = what the tagged series name starts with
        output_dir(string) = where the image pointers will be saved
        suffix(string) = text and file extension after the patient name
    
    Output:
        None, creates the list needed to be stored in the image pointer file
    '''
    start = time()  #timekeeping
    files = [f for f in os.listdir(image_path) if f.endswith(".dcm")]   #list the image files in the patient folder
    cine_and_tagged = [] #cine and tagged image files
    i = 0 #iteration count for while loop
    log_and_print(["Making image pointer file for {} from {}".format(pat_name, image_path)])
    # iterate through the patient files
    while i < len(files):
        try:
            # read the dicom file and get the series description and the instance number
            ds = pydicom.dcmread(os.path.join(image_path,files[i]), specific_tags=["SeriesDescription", "InstanceNumber"])
        except pydicom.errors.InvalidDicomError:
            log_error_and_print("Forcing read...")
            ds = pydicom.dcmread(os.path.join(image_path,files[i]), force = True, specific_tags=["SeriesDescription", "InstanceNumber"])

        try:    #if pydicom file does not have a series description in the header info, skip that file
            curr_series = ds.SeriesDescription
        except AttributeError:
            log_error_and_print("Skipping file in {}".format(os.path.join(image_path,files[i])))
            i += 1
            continue

        if sax_cine_prefix in curr_series: #normal cines
            # we'll put the series number as 0 for the normal cines
            series_n = 0
            try:
                slice_n = int(curr_series[len(sax_cine_prefix):])-1 #file name is "CINE_segmented_SAX_b#", where # is the slice number
            except ValueError:  #in some rare occassions, the series name does not match the prefix. So we try a different approach.
                log_error_and_print("Special name case found | Patient dir: {}".format(image_path))
                try:
                    log_error_and_print("Trying different approach")
                    slice_n = int(curr_series[len(sax_cine_prefix):len(curr_series)-1])-1 #file name is "CINE_segmented_SAX_b#", where # is the slice number
                except ValueError:
                    log_error_and_print("Issue unresolved\n")
                    return
                log_and_print("Issue resolved!\n")

            # index will start from 0
            index = ds.InstanceNumber-1
            # add info to list
            cine_and_tagged.append("{:>2}\t{:>2}\t{:>2}\t{}\\{}".format(series_n, slice_n, index, gen_image_path, files[i]))

        elif tagged_cine_prefix in curr_series: #tagged cines
            # we'll put the series number as 1 for the normal cines
            series_n = 1
            try:
                slice_n = int(curr_series[len(tagged_cine_prefix):len(curr_series)-1])-1 #file name is "cine_tagging_3sl_SAX_b#s", where # is the slice number
            except ValueError:  #in some rare occassions, the series name does not match the prefix. So we try a different approach.
                log_error_and_print("Special name case found | Patient dir: {}".format(image_path))
                try:
                    log_error_and_print("Trying different approach...")
                    slice_n = int(curr_series[len(tagged_cine_prefix):])-1
                except ValueError:
                    log_error_and_print("Issue unresolved\n")
                    return
                log_and_print("Issue resolved!\n")

            # index will start from 0
            index = ds.InstanceNumber-1
            # add info to list
            cine_and_tagged.append("{:>2}\t{:>2}\t{:>2}\t{}\\{}".format(series_n, slice_n, index, gen_image_path, files[i]))

        i += 1  #increment number of files we've looped through

    log_and_print(["Looped through {} of {} files in {}".format(i, len(files), pat_name)])
    print("Sorting...")
    cine_and_tagged.sort()  #arrange the contents of the list in ascending error

    # saving the image pointer as a file
    save_image_pointer(filepath, pat_name, output_dir, cine_and_tagged, suffix)

    # time keeping
    _, mins, secs = calculate_time_elapsed(start)
    log_and_print(["Elapsed time: {} minutes {} seconds".format(mins, secs)])

    return

def save_image_pointer(filepath, pat_name, output_dir, image_ptr_list, suffix):
    '''
    This function will save the contents of the image pointer
    Inputs:
        filepath(string) = path to the multipatient folders
        pat_name(string) = name of the patient(from dicom header)
        output_dir(string) = where the image pointers will be saved
        image_path = 
        image_ptr_list(list) = contains all the ordered slices from both series of interest

    Output:
        None, creates image pointer
    '''
    output_file = os.path.join(output_dir, "{}{}".format(pat_name.replace(" ", "_"), suffix))
    # create the image pointer file
    with open(output_file, "w") as f:
        # header of the image pointer
        f.write("### SeriesNum(0=normal cine, 1=tagged cine) SliceNum IndexNum Path delimiter = \"\\t\" ({}) ###\n".format(filepath))
        for item in image_ptr_list:
            f.write("{}\n".format(item))
    # display info
    log_and_print(["Created image pointer file location: {}".format(output_file)])

    return

