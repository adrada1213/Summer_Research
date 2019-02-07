"""
This python file prepares the image pointers. In the patient folder, the dicom files are not in order that's why we 
need to create the image pointers. The image pointers will serve as a mapping file the the dicom images inside the 
patient folders. Only the image paths for the cine and tagged series will be included in the image pointers. Each 
series has different number of slices, each slices have multiple frames; usually 50 for cine series, and 20 for 
tagged series. 

The preparation of image pointers is divided into three steps:
    1. Creation of image pointer for all patients
    2. Moving the image pointers that have missing tagged series
    3. Finding the matching slices between the cine and tagged series
Author: Amos Rada
Date(Last edit): 22/01/2019   
"""

# import needed libraries
import os
import shutil
import logging
import numpy as np
import pydicom
from datetime import datetime
from time import time
from prepare_data_functions import sendemail, get_pointer_paths, calculate_time_elapsed, log_and_print
from dicom_functions import get_patient_name
from pointer_functions import create_new_image_pointer, load_ptr_content

# initialise logger
logger = logging.getLogger(__name__)

def prompt_user():
    '''
    Prompts user to enter which functions they want to use
    Output:
        steps(list) = the number of the function that the user wants to use
    '''
    print("Function 1: Create image pointers")
    print("Function 2: Move image pointers with missing data")
    print("Function 3: Find matches between series of interest")
    try:
        steps = [int(s) for s in input("Enter the function(s) you want to use (separated by space): ").split()]
        steps.sort()
        print()
    except ValueError:
        print()
        print("Invalid values entered")
        print("Please enter values from 1 to 3 separate by a space")
        steps = prompt_user()
    
    return steps

def create_img_ptrs(filepath, output_dir, suffix):
    '''
    Step 1: Prepare all the image pointers
    Inputs: 
        filepath(list) = the filepaths containing the multipatient folders
        output_dir(string) = where the image pointers will be stored
        suffix(string) = the text and file extension after the patient name
    
    Outputs:
        None, it creates image pointers
    '''
    start = time() #start time for this function
    output_messages = ["====================CREATING IMAGE POINTERS====================",
                        "Operation started at {}".format(datetime.now().time())]
    log_and_print(output_messages)

    # series identifiers (series of interest)
    sax_cine_prefix = "CINE_segmented_SAX_b"
    tagged_cine_prefix = "cine_tagging_3sl_SAX_b"

    # initialise other variables
    file_c = len([f for f in os.listdir(output_dir) if f.endswith(".img_imageptr")]) # variable to keep file count
    initial_file_c = file_c #number of files at the start of the program

    # if the creation of image pointer function was interrupted before, you can start from the last index shown in the log file
    start_i = int(input("Enter the patient number you want to start with (beginning = 0): "))
    print()

    # creating image pointers for tagged and cines based on the file path found in the image pointer
    log_and_print(["Creating image pointer files in {}\n".format(output_dir)])
    try:
        i = -1
        for fp in filepath:
            for root, _, files in os.walk(fp):
                new_fp = fp
                images = []
                # check if folder contains .dcm files. If yes, the current folder is a patient folder
                images = [f for f in files if f.endswith(".dcm")]
                if len(images) != 0:
                    i += 1  #keep count of the patient number
                    if i >= start_i:
                        # list the current image pointers
                        image_pointer_list = [imgptr for imgptr in os.listdir(output_dir) if imgptr.endswith(".img_imageptr")]
                        patient_path = root
                        # get the patient name from dicom header
                        patient_name = get_patient_name(patient_path)
                        # check if image pointer is not yet created
                        if "{}{}".format(patient_name.replace(" ", "_"), suffix) not in image_pointer_list:
                            # convert the patient path to a general one so it can be used in the future
                            gen_patient_path = patient_path.replace(new_fp, "IMAGEPATH")    
                            try:
                                # create the image pointer
                                create_new_image_pointer(new_fp, patient_name, patient_path, gen_patient_path, sax_cine_prefix, tagged_cine_prefix, output_dir, suffix)
                            except FileNotFoundError:
                                # error handling
                                logger.error("\nThe system cannot find the path specified {}\n".format(patient_path), exc_info=True)
                                continue
                            file_c+= 1  # increase number of image pointer files created
                            # info for user
                            output_messages = ["Total # of image pointers created: {}".format(file_c),
                                                "Iteration info: Patient#{} {}\n".format(i, patient_name)]
                            log_and_print(output_messages)
                        # info for user if image pointer already exists
                        else:       
                            print("{}{} already exists".format(patient_name.replace(" ","_"), suffix))
                            print("Iteration info: Patient#{} {}\n".format(i, patient_name))
            
            # display when function is finished running                    
            hrs, mins, secs = calculate_time_elapsed(start)
            output_messages = ["=========================IMAGE POINTERS CREATED!=========================",
                                "Image pointers created during operation: {}".format(file_c-initial_file_c),
                                "Total # of image pointers created: {}".format(file_c),
                                "Image pointer files stored in {}".format(output_dir),
                                "Operation finished at {}".format(str(datetime.now())),
                                "Total elapsed time: {} hours {} minutes {} seconds\n".format(hrs, mins, secs)]
            log_and_print(output_messages)

    except KeyboardInterrupt:   #display when code is interrupted by user
        hrs, mins, secs = calculate_time_elapsed(start)
        output_messages = ["=========================CREATION OF IMAGE POINTERS CANCELLED!=========================",
                            "Operation cancelled at {}".format(str(datetime.now())),
                            "Image pointers created during operation: {}".format(file_c-initial_file_c),
                            "Total # of image pointers created: {}".format(file_c),
                            "Image pointer files stored in {}".format(output_dir),
                            "Last iteration info: Patient#{} {}".format(i, patient_name),
                            "Total elapsed time: {} hours {} minutes {} seconds\n".format(hrs, mins, secs)]
        log_and_print(output_messages)

def move_img_ptrs(ptr_path, output_dir_missing, suffix):
    '''
    Step 2: Move image pointers with missing data (no tagged series) to a separate folder
    Inputs:
        ptr_path(string) = path to all the image pointers
        output_dir_missing(string) = where the image pointers with missing data will be moved to
        suffix(string) = text and file extension after the patient name
    Output:
        None, moves image pointers
    '''
    start = time() #timekeeping
    output_messages = ["====================MOVING IMAGE POINTERS====================",
                        "Operation started at {}".format(datetime.now().time())]
    log_and_print(output_messages)

    # get the path to each pointer file
    ptr_files = [os.path.join(ptr_path, ptr) for ptr in os.listdir(ptr_path) if ptr.endswith(".img_imageptr")] 

    # initialise other variables
    moved_c = len([f for f in os.listdir(output_dir_missing) if f.endswith(".img_imageptr")]) #keeps track of number of image pointers moved
    initial_moved_c = moved_c   #keeps track of number of image poiners moved in current run

    # looping through each pointer file
    log_and_print(["Moving image pointer files to {}\n".format(output_dir_missing)])
    try:
        for i, ptr in enumerate(ptr_files):
            # reading the content of the pointer file
            ptr_content = load_ptr_content(ptr_path)

            # only extract the first frame of each slice
            tagged_con = ptr_content["series"] == 1   #condition for tagged series
            first_frames_tagged = ptr_content[tagged_con]   #extract the first frames from each slice in tagged cine series

            patient_name = os.path.basename(ptr).replace(suffix, "")   #get the patient name
            print("Checking patient {} of {}: {}".format(i+1,len(ptr_files),patient_name))

            # if tagged series is missing
            if len(first_frames_tagged) == 0:
                log_and_print(["Moving image pointer {}".format(os.path.basename(ptr))])
                shutil.move(ptr, output_dir_missing)    #move the image pointer
                moved_c += 1    #increment number of files moved

        # display info when function is finished running
        hrs, mins, secs = calculate_time_elapsed(start)
        output_messages = ["=========================IMAGE POINTERS MOVED!=========================",
                            "Image pointers moved to {}".format(output_dir_missing),
                            "Looped through {} of {} patients".format(i+1,len(ptr_files)),
                            "Number of pointer files moved: {}".format(moved_c),
                            "Operation finished at {}".format(str(datetime.now().time())),
                            "Total elapsed time: {} hours {} minutes {} seconds\n".format(hrs, mins, secs)]
        log_and_print(output_messages)

    except KeyboardInterrupt:   #when the function is interrupted by the user
        hrs, mins, secs = calculate_time_elapsed(start)
        output_messages = ["=========================TRANSFER OF IMAGE POINTERS CANCELLED!=========================",
                            "Operation cancelled at {}".format(str(datetime.now())),
                            "Looped through {} of {} patients".format(i+1,len(ptr_files)),
                            "Number of pointer files moved during operation: {}".format(initial_moved_c-moved_c),
                            "Total elapsed time: {} hours {} minutes {} seconds".format(hrs, mins, secs)]
        log_and_print(output_messages)
        logger.error("Unexpected error occured at {}\n".format(str(datetime.now())), exc_info=True)

def find_matches(filepath, ptr_files_path, cim_models, cim_dir, output_dir_match, suffix):
    '''
    Step 3: Find the slices in the cine series that match the slices in the tagged series
    Inputs:
        filepath(list) = paths to the multipatient folders
        ptr_files_path(list) = path to the image pointers
        cim_models(list) = list of the names of the cim models
        cim_dir(string) = folder containing the cim models
        output_dir_match(string) = where the image pointers with matching slice will be saved
        suffix(string) = text and file extension that will be added after the patient name (e.g. _cine_and_tagged.img_imageptr)
    Output:
        None, creates image pointers with matching slices
    '''
    start = time() #timekeeping
    output_messages = ["====================FINDING MATCHES====================",
                        "Operation started at {}".format(datetime.now().time()),
                        "Finding matches for the image pointers stored in {}\n".format(ptr_files_path)]
    log_and_print(output_messages)

    # list each pointer file
    ptr_files = [ptr for ptr in os.listdir(ptr_files_path) if ptr.endswith("img_imageptr")] 

    # get all the paths to the cim ptr files
    cim_dirs = [os.path.join(cim_dir, d) for d in cim_models]
    cim_ptr_files = []
    for cim_dir in cim_dirs:
        cim_ptr_files += get_pointer_paths(cim_dir)

    # initialise other variables
    file_c = len([f for f in os.listdir(output_dir_match) if f.endswith(".img_imageptr")]) # variable to keep file count
    initial_file_c = file_c #number of files at the start of the program
    tol = 1e-5 #tolerance to check if two slices match
    match = False   #initialise match state
    # if match finding was interrupted before, you can start from the last index shown in the log file
    start_i = int(input("Enter the patient number you want to start with (beginning=0): "))

    # looping through each pointer file
    log_and_print(["Creating new image pointers in {}\n".format(output_dir_match)])
    try:
        for i, ptr in enumerate(ptr_files):
            if i >= start_i:    #condition to start at the specified index
                ptr_path = os.path.join(ptr_files_path, ptr)    #get the path to the image pointer
                match_ip = [f for f in os.listdir(output_dir_match) if f.endswith(".img_imageptr")]   #list all image pointers created by this program
                patient_name = ptr.replace(suffix, "") #get the patient name from the image pointer file name
                fp = filepath[0]   #default filepath is filepath1
                print("Checking patient {} of {}: {}".format(i+1,len(ptr_files),patient_name))
                output_filename = "{}_match.img_imageptr".format(patient_name)
                #if the image pointer with the matching slices doesn't exist for the patient yet
                if output_filename not in match_ip:    
                    try:
                        #get the cim pointer path of current patient
                        cim_ptr_path = [cim_ptr for cim_ptr in cim_ptr_files if patient_name.lower() in cim_ptr.lower()][0] 
                        logger.info("Patient#{}: {} - CIM image pointer found: {}".format(i+1, patient_name, cim_ptr_path))
                    except IndexError: 
                        try:
                            #get the cim pointer path of current patient
                            cim_ptr_path = [cim_ptr for cim_ptr in cim_ptr_files if patient_name.lower().replace("_bio", "") in cim_ptr.lower()][0] 
                            logger.info("Patient#{}: {} - CIM image pointer found: {}".format(i+1, patient_name, cim_ptr_path))
                        except IndexError:    
                            logger.error("Patient#{}: {} - No image pointer file found in the CIM folders\n".format(i+1, patient_name))
                            continue
                    
                    # reading the content of the pointer file (cine_and_tagged, and cim)
                    ptr_content = load_ptr_content(ptr_path)
                    cim_ptr_content = load_ptr_content(cim_ptr_path)
                    with open(ptr_path, "r") as f:  #obtain header to be written later to the output file together with the matching slices
                        header=f.readline().strip()

                    # only extract the first frame of each slice (cine_and_tagged, and cim)
                    condition_1 = np.logical_and(ptr_content["series"] == 0, ptr_content["index"] == 0) #condition to find the first frames of each slice in normal cine series
                    condition_2 = np.logical_and(ptr_content["series"] == 1, ptr_content["index"] == 0)   #condition to find the first frames of each slice in tagged cine series
                    condition_3 = np.logical_and(cim_ptr_content["series"] == 0, cim_ptr_content["index"] == 0) #condition to find the first frames of each slice in tagged cine series (cim)
                    first_fr_cine = ptr_content[condition_1]   #extract the first frames from each slice in normal cine series
                    first_fr_tagged = ptr_content[condition_2]   #extract the first frames from each slice in tagged cine series
                    cim_first_fr_tagged = cim_ptr_content[condition_3]  #extract the first frames from each slice in tagged cine series of the cim ptr file
                    
                    tagged_array = []   #reset tagged array 
                    cine_array = [] #reset cine array 
                    # loop through the slices in cim tagged cine
                    for cim_frame_t in cim_first_fr_tagged:
                        cim_file_name = os.path.basename(cim_frame_t["path"])    #get the filename of the first frame of the first slice in the cine imgpointer
                        try:
                            tagged_slice_i = [curr_i for curr_i, curr_slice in enumerate(first_fr_tagged) if cim_file_name in curr_slice["path"]][0]  #get the slice index from the created image pointer that matches the current slice in the cim image pointer
                        except IndexError:
                            match = False
                            logger.error("Error Occurred\n", exc_info = True)
                            break
                        tagged_slice = first_fr_tagged[tagged_slice_i]  #take tagged slice from our image pointer
                        tagged_series = ptr_content[np.logical_and(ptr_content["series"] == 1, ptr_content["slice"] == tagged_slice["slice"])] #get the whole series of that slice from the created image pointer
                        tagged_series[:]["slice"] = cim_frame_t["slice"] #replace the slice number of the tagged series with the slice number of the tagged series in the cim image pointer file
                        '''
                        if j == 0:  #to initialise the array to be saved later
                            tagged_array = tagged_series.flatten()
                        else:
                            tagged_array = np.append(tagged_array, tagged_series.flatten())
                        '''
                        first_fr_tagged = np.delete(first_fr_tagged, tagged_slice_i)    #delete added slice to reduce computational time
                        
                        tagged_image_file = tagged_slice["path"].replace("IMAGEPATH", fp)
                        
                        if not os.path.exists(tagged_image_file):   #if file is in the 2014 folder
                            tagged_image_file = tagged_slice["path"].replace("IMAGEPATH", filepath[1])
                            fp = filepath[1]
                        
                        ds_tag = pydicom.dcmread(tagged_image_file, specific_tags=["SliceLocation", "PatientPosition"])    #get the slice location from metaheader

                        # Loop through each slice in the cine series
                        for k, frame_c in enumerate(first_fr_cine):
                            cine_image_file = frame_c["path"].replace("IMAGEPATH", fp)    #get the file path
                            ds_cine = pydicom.dcmread(cine_image_file, specific_tags=["SliceLocation", "PatientPosition"]) #read to get slice location
                            if ds_tag.PatientPosition == ds_cine.PatientPosition:
                                # slice location determines whether the slices match
                                if abs(abs(ds_tag.SliceLocation)-abs(ds_cine.SliceLocation)) <= tol:   
                                    cine_series = ptr_content[np.logical_and(ptr_content["series"] == 0, ptr_content["slice"]==frame_c["slice"])]   #get the entire cine series that matches the current tagged slice
                                    cine_series[:]["slice"] = cim_frame_t["slice"]  #replace the slice number of the current cine series with the slice number from the tagged series of the cim image pointer file
                                    if not match:  #to initialise the array to be saved later
                                        match = True
                                        cine_array = cine_series.flatten()
                                        tagged_array = tagged_series.flatten()
                                    else:
                                        cine_array = np.append(cine_array, cine_series.flatten())
                                        tagged_array = np.append(tagged_array, tagged_series.flatten())
                                    first_fr_cine = np.delete(first_fr_cine, k) #delete added slice to reduce computational time
                            else:
                                log_and_print(["Patient position not the same"])

                    if match:
                        output_array = np.append(cine_array, tagged_array)  #combine two arrays
                        # create the image pointer file containing the slices in the cine and tagged series that match
                        logger.info("New image pointer file: {}\n".format(os.path.join(output_dir_match, output_filename)))
                        np.savetxt(os.path.join(output_dir_match, output_filename), output_array, fmt="%2d\t%2d\t%2d\t%s", delimiter = "\t", header = header, comments="")
                        file_c += 1
                        match = False   #reset match status
                    else:
                        logger.error("No match found! Patient directory: {}\n".format(os.path.dirname(cine_image_file)))
                        continue
                    
        hrs, mins, secs = calculate_time_elapsed(start)
        output_messages = ["=========================MATCHES FOUND!=========================",
                            "Looped through {} of {} patients".format(i+1-start_i,len(ptr_files)),
                            "Image pointers created during operation: {}".format(file_c-initial_file_c),
                            "Total # of image pointers created: {}".format(file_c),
                            "Image pointer files stored in {}".format(output_dir_match),
                            "Operation finished at {}".format(str(datetime.now())),
                            "Total elapsed time: {} hours {} minutes {} seconds\n".format(hrs, mins, secs)]
        log_and_print(output_messages)

        #sendemail("adrada1213@gmail.com", "ad_rada@hotmail.com", "prepare_img_ptrs.py Program Finished", "Here's the log file:", os.path.join(os.getcwd(),logname))

        #os.system("shutdown -s -t 0")
    
    except KeyboardInterrupt:
        hrs, mins, secs = calculate_time_elapsed(start)
        output_messages = ["=========================CREATION OF IMAGE POINTERS CANCELLED(MATCH)!=========================",
                            "Operation cancelled at {}".format(str(datetime.now())),
                            "Looped through {} of {} patients".format(i+1-start_i,len(ptr_files)),
                            "Image pointers created during operation: {}".format(file_c-initial_file_c),
                            "Total # of image pointers created: {}".format(file_c),
                            "Image pointer files stored in {}".format(output_dir_match),
                            "Last iteration info: Patient#{} {}".format(i, patient_name),
                            "Total elapsed time: {} hours {} minutes {} seconds\n".format(hrs, mins, secs)]
        log_and_print(output_messages)

    except:
        logger.error("UNEXPECTED ERROR", exc_info = True)
        #sendemail("adrada1213@gmail.com", "ad_rada@hotmail.com", "prepare_img_ptrs.py Program Interrupted", "Here's the log file:", os.path.join(os.getcwd(),logname))


def prepare_img_ptrs(basepath, filepath, suffix, cim_dir, cim_models, output_dir, output_dir_missing, output_dir_match):
    '''
    There are three steps to find the matching image pointers
    1. Create all the image pointers (create_img_ptrs)
    2. Moving the image pointers with missing data (No tagged series) (move_img_ptrs)
    3. Finding the matching slices between the cine and tagged series (find_matches)
    '''
    # create the output directories if they don't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir_missing):
        os.makedirs(output_dir_missing)
    if not os.path.exists(output_dir_match):
        os.makedirs(output_dir_match)

    # prompt the user which functions they want to use
    steps = prompt_user()

    try:
        for s in steps:
            if s == 1:
                try:
                    create_img_ptrs(filepath, output_dir, suffix)
                except KeyboardInterrupt:
                    break
            if s == 2:
                try:
                    move_img_ptrs(output_dir, output_dir_missing, suffix)
                except KeyboardInterrupt:
                    break
            if s == 3:
                try:
                    find_matches(filepath, output_dir, cim_models, cim_dir, output_dir_match, suffix)
                except KeyboardInterrupt:
                    break
        
        # Timekeeping
        hrs, mins, secs = calculate_time_elapsed(start_)
        output_messages = ["====================MAIN PROGRAM FINISHED!====================",
                            "Total elapsed time: {} hours {} minutes {} seconds\n".format(hrs, mins, secs)]
        log_and_print(output_messages)

    except KeyboardInterrupt:
        print("Operation Interrupted")
        print("Log file stored in: {}".format(os.path.join(basepath,logname)))

if __name__ == "__main__":
    '''
    Calls the main function (prepare_image_ptrs)
    To be modified by user: 
        basepath = where you want the image pointers to be stored
        filepath = where the multipatient folders are stored (even if there's only one path, put it in a list)
        suffix = text and file extension that will be added after the patient name (e.g. _cine_and_tagged.img_imageptr)
        cim_dir = where the cim folders are located
        cim_models = names of the cim models (as a list)        
    '''

    # logging info
    start_ = time() # to keep time
    ts = datetime.fromtimestamp(start_).strftime('%Y-%m-%d') #time stamp for the log file
    logname = "{}-prepare-img-ptrs.log".format(ts)
    logging.basicConfig(filename=logname, level=logging.DEBUG)

    # start of main program
    output_messages = ["====================STARTING MAIN PROGRAM====================",
                        "Operation started at {}\n".format(datetime.now().time())]
    log_and_print(output_messages)
    
    # where you want the outputs to be stored
    basepath = os.path.join(os.getcwd())
    #basepath = "" #uncomment and specify if you want the basepath to be different from the working directory

    # where the multipatient folders are stored
    filepath = ["E:\\Original Images\\2014", "E:\\Original\\Images\\2015"]

    # output image pointer file suffix
    suffix = "_cine_and_tagged.img_imageptr"

    # dirnames and paths of the cim models
    cim_dir = "C:\\Users\\arad572\\Downloads\\all CIM"
    cim_models = ["CIM_DATA_AB", "CIM_DATA_EL1", "CIM_DATA_EL2", "CIM_DATA_EM", "CIM_DATA_KF", "CIM_Data_ze_1", "CIM_DATA_ze_2", "CIM_DATA_ze_3", "CIM_DATA_ze_4"]

    # output directory of the image pointers
    output_dir = os.path.join(basepath, "test")

    # output directory of the image pointers with missing data
    output_dir_missing = os.path.join(output_dir, "missing")

    # output directory of the image pointers with matching cine and tagged slices
    output_dir_match = os.path.join(output_dir, "matches")

    # main function to prepare image pointers
    prepare_img_ptrs(basepath, filepath, suffix, cim_dir, cim_models, output_dir, output_dir_missing, output_dir_match)