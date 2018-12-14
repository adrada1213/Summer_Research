import os
import shutil
import logging
import numpy as np
from datetime import datetime
from time import time
import pydicom
import random
from prepare_data_functions import sendemail, get_pointer_paths, calculate_time_elapsed, log_and_print

logger = logging.getLogger(__name__)

'''
Functions to prepare image pointers
'''
def create_new_image_pointer(filepath, pat_name, image_path, gen_image_path, sax_cine_prefix, tagged_cine_prefix, output_dir, suffix):
    start = time()
    files = [f for f in os.listdir(image_path) if f.endswith(".dcm")]
    cine_and_tagged = [] #cine and tagged image files
    i = 0 #iteration count for while loop
    log_and_print(["Making image pointer file for {} from {}".format(pat_name, image_path)])
    while i < len(files):
        try:
            ds = pydicom.dcmread(os.path.join(image_path,files[i]), specific_tags=["SeriesDescription", "InstanceNumber"])
        except pydicom.errors.InvalidDicomError:
            print("Forcing read...")
            logger.error("Forcing read...")
            ds = pydicom.dcmread(os.path.join(image_path,files[i]), force = True, specific_tags=["SeriesDescription", "InstanceNumber"])
        try:
            curr_series = ds.SeriesDescription
        except AttributeError:
            print("Skipping file in {}".format(os.path.join(image_path,files[i])))
            logger.error("Skipping file in {}".format(os.path.join(image_path,files[i])))
            i += 1
            continue
        if sax_cine_prefix in curr_series: #normal cines
            series_n = 0
            try:
                slice_n = int(curr_series[len(sax_cine_prefix):])-1 #file name is "CINE_segmented_SAX_b#", where # is the slice number
            except ValueError:
                logger.error("Special name case found | Patient dir: {}".format(image_path))
                try:
                    logger.error("Trying different approach...")
                    slice_n = int(curr_series[len(sax_cine_prefix):len(curr_series)-1])-1 #file name is "CINE_segmented_SAX_b#", where # is the slice number
                except ValueError:
                    logger.error("Issue unresolved\n")
                    return
                logger.info("Issue resolved!\n")
            index = ds.InstanceNumber-1
            cine_and_tagged.append("{:>2}\t{:>2}\t{:>2}\t{}\\{}".format(series_n, slice_n, index, gen_image_path, files[i]))
        elif tagged_cine_prefix in curr_series: #tagged cines
            series_n = 1
            try:
                slice_n = int(curr_series[len(tagged_cine_prefix):len(curr_series)-1])-1 #file name is "cine_tagging_3sl_SAX_b#s", where # is the slice number
            except ValueError:
                logger.error("Special name case found | Patient dir: {}".format(image_path))
                try:
                    logger.error("Trying different approach...")
                    slice_n = int(curr_series[len(tagged_cine_prefix):])-1
                except ValueError:
                    logger.error("Issue unresolved\n")
                    return
                logger.info("Issue resolved!\n")
            index = ds.InstanceNumber-1
            cine_and_tagged.append("{:>2}\t{:>2}\t{:>2}\t{}\\{}".format(series_n, slice_n, index, gen_image_path, files[i]))
        i += 1

    log_and_print(["Looped through {} of {} files in {}".format(i, len(files), pat_name)])
    print("Sorting...")
    cine_and_tagged.sort()

    # saving the image pointer as a file
    save_image_pointer(filepath, pat_name, output_dir, image_path, cine_and_tagged, suffix)

    # time keeping
    hrs, mins, secs = calculate_time_elapsed(start)
    log_and_print(["Elapsed time: {} minutes {} seconds".format(mins, secs)])

    return

def save_image_pointer(filepath, pat_name, output_dir, image_path, image_ptr_list, suffix):
    output_file = os.path.join(output_dir, "{}{}".format(pat_name.replace(" ", "_"), suffix))
    with open(output_file, "w") as f:
        f.write("### SeriesNum(0=normal cine, 1=tagged cine) SliceNum IndexNum Path delimiter = \"\\t\" ({}) ###\n".format(filepath))
        for item in image_ptr_list:
            f.write("{}\n".format(item))
    log_and_print(["Created image pointer file location: {}".format(output_file)])

    return

def get_patient_name(image_path):
    files = [f for f in os.listdir(image_path) if f.endswith(".dcm")]
    try:
        random_index = random.randint(0,len(files))
    except ValueError:
        random_index = 0
    try:
        pfile = files[random_index]   #get one patient file
    except IndexError:
        pfile = files[random_index//2]
    ds = pydicom.dcmread(os.path.join(image_path, pfile), specific_tags=["PatientName"])    #read the patient name
    try:
        patient_name = str(ds.PatientName).replace("^", "_")
    except AttributeError:
        patient_name = get_patient_name(image_path)
        
    return patient_name

def prompt_user():
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

'''
Step 1 of preparing the data
Creating the image pointer files for the series of interest
'''
def create_img_ptrs(filepath, output_dir, suffix):
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
    start_i = int(input("Enter the patient number you want to start with (beginning = 0): "))
    print()

    # creating image pointers for tagged and cines based on the file path found in the image pointer
    log_and_print(["Creating image pointer files in {}\n".format(output_dir)])
    try:
        i = -1
        for root, dirs, files in os.walk(filepath):
            '''
            if os.path.dirname(root) == filepath:
                new_fp = root
            '''
            new_fp = filepath
            images = []
            images = [f for f in files if f.endswith(".dcm")]
            if len(images) != 0:
                i += 1  #keep count of the patient number
                if i >= start_i:
                    image_pointer_list = [imgptr for imgptr in os.listdir(output_dir) if imgptr.endswith(".img_imageptr")]
                    patient_path = root
                    patient_name = get_patient_name(patient_path)
                    if "{}{}".format(patient_name.replace(" ", "_"), suffix) not in image_pointer_list:
                        gen_patient_path = patient_path.replace(new_fp, "IMAGEPATH")
                        try:
                            create_new_image_pointer(new_fp, patient_name, patient_path, gen_patient_path, sax_cine_prefix, tagged_cine_prefix, output_dir, suffix)
                        except FileNotFoundError:
                            logger.error("\nThe system cannot find the path specified {}\n".format(patient_path), exc_info=True)
                            continue
                        file_c+= 1
                        # info for user
                        output_messages = ["Total # of image pointers created: {}".format(file_c),
                                            "Iteration info: Patient#{} {}\n".format(i, patient_name)]
                        log_and_print(output_messages)
                    else:
                        # info for user if image pointer already exists
                        #output_messages = ["{}{} already exists".format(patient_name.replace(" ","_"), suffix),
                        #                    "Iteration info: Patient#{} {}\n".format(i, patient_name)]
                        #log_and_print(output_messages, logger)
                        print("{}{} already exists".format(patient_name.replace(" ","_"), suffix))
                        print("Iteration info: Patient#{} {}\n".format(i, patient_name))
        
        # display when code is finished running                    
        hrs, mins, secs = calculate_time_elapsed(start)
        output_messages = ["=========================IMAGE POINTERS CREATED!=========================",
                            "Image pointers created during operation: {}".format(file_c-initial_file_c),
                            "Total # of image pointers created: {}".format(file_c),
                            "Image pointer files stored in {}".format(output_dir),
                            "Operation finished at {}".format(str(datetime.now())),
                            "Total elapsed time: {} hours {} minutes {} seconds\n".format(hrs, mins, secs)]
        log_and_print(output_messages)

    except KeyboardInterrupt:
        hrs, mins, secs = calculate_time_elapsed(start)
        output_messages = ["=========================CREATION OF IMAGE POINTERS CANCELLED!=========================",
                            "Operation cancelled at {}".format(str(datetime.now())),
                            "Image pointers created during operation: {}".format(file_c-initial_file_c),
                            "Total # of image pointers created: {}".format(file_c),
                            "Image pointer files stored in {}".format(output_dir),
                            "Last iteration info: Patient#{} {}".format(i, patient_name),
                            "Total elapsed time: {} hours {} minutes {} seconds\n".format(hrs, mins, secs)]
        log_and_print(output_messages)

'''
This is step 2
Move image pointers with missing data to a separate folder
'''
def move_img_ptrs(filepath, ptr_path, output_dir_missing, suffix):
    start = time() #start time for this function
    output_messages = ["====================MOVING IMAGE POINTERS====================",
                        "Operation started at {}".format(datetime.now().time())]
    log_and_print(output_messages)

    # get the path to each pointer file
    ptr_files = [os.path.join(ptr_path, ptr) for ptr in os.listdir(ptr_path) if ptr.endswith(".img_imageptr")] 

    # initialise other variables
    moved_c = len([f for f in os.listdir(output_dir_missing) if f.endswith(".img_imageptr")]) #moved image pointers count
    initial_moved_c = moved_c

    # looping through each pointer file
    log_and_print(["Moving image pointer files to {}\n".format(output_dir_missing)])
    try:
        for i, ptr in enumerate(ptr_files):
            # reading the content of the pointer file
            datatype = [('series', '<i4'), ('slice', '<i4'), ('index', '<i4'), ('path', 'U255')]
            ptr_content = np.genfromtxt(ptr, delimiter='\t', names='series, slice, index, path', skip_header=1, dtype=datatype)

            # only extract the first frame of each slice
            tagged_con = ptr_content["series"] == 1   #condition for tagged series
            first_frames_tagged = ptr_content[tagged_con]   #extract the first frames from each slice in tagged cine series

            patient_name = os.path.basename(ptr).replace(suffix, "")   #get the patient name
            print("Checking patient {} of {}: {}".format(i+1,len(ptr_files),patient_name))
            if len(first_frames_tagged) == 0:
                moved_c += 1
                log_and_print(["Moving image pointer {}".format(os.path.basename(ptr))])
                shutil.move(ptr, output_dir_missing)

        hrs, mins, secs = calculate_time_elapsed(start)
        output_messages = ["=========================IMAGE POINTERS MOVED!=========================",
                            "Image pointers moved to {}".format(output_dir_missing),
                            "Looped through {} of {} patients".format(i+1,len(ptr_files)),
                            "Number of pointer files moved: {}".format(moved_c),
                            "Operation finished at {}".format(str(datetime.now().time())),
                            "Total elapsed time: {} hours {} minutes {} seconds\n".format(hrs, mins, secs)]
        log_and_print(output_messages)

    except KeyboardInterrupt:
        hrs, mins, secs = calculate_time_elapsed(start)
        output_messages = ["=========================TRANSFER OF IMAGE POINTERS CANCELLED!=========================",
                            "Operation cancelled at {}".format(str(datetime.now())),
                            "Looped through {} of {} patients".format(i+1,len(ptr_files)),
                            "Number of pointer files moved during operation: {}".format(initial_moved_c-moved_c),
                            "Total elapsed time: {} hours {} minutes {} seconds".format(hrs, mins, secs)]
        log_and_print(output_messages)
        logger.error("Unexpected error occured at {}\n".format(str(datetime.now())), exc_info=True)

        #sendemail("adrada1213@gmail.com", "ad_rada@hotmail.com", "move_ptr_files.py Program Interrupted", "Here's the log file:", os.path.join(os.getcwd(),logname))

        #os.system("shutdown -s -t 0")

'''
This is step 3
Finding the matching slices between series of interest
'''
def find_match(filepath, ptr_files_path, cim_models, cim_dir, output_dir_match, suffix):
    start = time() #start time for this function
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
    start_i = int(input("Enter the patient number you want to start with: "))
    print()

    # looping through each pointer file
    log_and_print(["Creating new image pointers in {}\n".format(output_dir_match)])
    try:
        for i, ptr in enumerate(ptr_files):
            if i >= start_i:
                ptr_path = os.path.join(ptr_files_path, ptr)    #get the path to the image pointer
                match_ip = [f for f in os.listdir(output_dir_match) if f.endswith(".img_imageptr")]   #list all image pointers created by this program
                patient_name = ptr.replace(suffix, "") #get the patient name from the image pointer file name
                filepath = "E:\\Original Images\\2015"    #default filepath is filepath1
                print("Checking patient {} of {}: {}".format(i+1,len(ptr_files),patient_name))
                output_filename = "{}_match.img_imageptr".format(patient_name)
                if output_filename not in match_ip:    #if the image pointer with the matching slices doesn't exist for the patient yet
                    try:
                        cim_ptr_path = [cim_ptr for cim_ptr in cim_ptr_files if patient_name.lower() in cim_ptr.lower()][0] #get the cim pointer path of current patient
                        logger.info("Patient#{}: {} - CIM image pointer found: {}".format(i+1, patient_name, cim_ptr_path))
                    except IndexError: 
                        try:
                            cim_ptr_path = [cim_ptr for cim_ptr in cim_ptr_files if patient_name.lower().replace("_bio", "") in cim_ptr.lower()][0] #get the cim pointer path of current patient
                            logger.info("Patient#{}: {} - CIM image pointer found: {}".format(i+1, patient_name, cim_ptr_path))
                        except IndexError:    
                            logger.error("Patient#{}: {} - No image pointer file found in the CIM folders\n".format(i+1, patient_name))
                            continue
                    
                    # reading the content of the pointer file (cine_and_tagged, and cim)
                    datatype = [('series', '<i4'), ('slice', '<i4'), ('index', '<i4'), ('path', 'U255')]
                    ptr_content = np.genfromtxt(ptr_path, delimiter='\t', names='series, slice, index, path', skip_header=1, dtype=datatype)
                    cim_ptr_content = np.genfromtxt(cim_ptr_path, delimiter='\t', names='series, slice, index, path', skip_header=1, dtype=datatype)
                    with open(ptr_path, "r") as f:  #obtain header to be written later to the output file together with the matching slices
                        header=f.readline().strip()

                    # only extract the first frame of each slice (cine_and_tagged, and cim)
                    condition_1 = np.logical_and(ptr_content["series"] == 0, ptr_content["index"] == 0) #condition to find the first frames of each slice in normal cine series
                    condition_2 = np.logical_and(ptr_content["series"] == 1, ptr_content["index"] == 0)   #condition to find the first frames of each slice in tagged cine series
                    condition_3 = np.logical_and(cim_ptr_content["series"] == 0, cim_ptr_content["index"] == 0) #condition to find the first frames of each slice in tagged cine series (cim)
                    first_fr_cine = ptr_content[condition_1]   #extract the first frames from each slice in normal cine series
                    first_fr_tagged = ptr_content[condition_2]   #extract the first frames from each slice in tagged cine series
                    cim_first_fr_tagged = cim_ptr_content[condition_3]  #extract the first frames from each slice in tagged cine series of the cim ptr file
                    
                    # loop through the slices in cim tagged cine
                    tagged_array = []   #reset tagged array 
                    cine_array = [] #reset cine array 
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
                        
                        tagged_image_file = tagged_slice["path"].replace("IMAGEPATH", filepath)
                        
                        if not os.path.exists(tagged_image_file):   #if file is in the 2014 folder
                            tagged_image_file = tagged_slice["path"].replace("IMAGEPATH", "E:\\Original Images\\2014")
                            filepath = "E:\\Original Images\\2014"
                        
                        ds_tag = pydicom.dcmread(tagged_image_file, specific_tags=["SliceLocation", "PatientPosition"])    #get the slice location from metaheader

                        # Loop through each slice in the cine series
                        for k, frame_c in enumerate(first_fr_cine):
                            cine_image_file = frame_c["path"].replace("IMAGEPATH", filepath)    #get the file path
                            ds_cine = pydicom.dcmread(cine_image_file, specific_tags=["SliceLocation", "PatientPosition"]) #read to get slice location
                            if ds_tag.PatientPosition == ds_cine.PatientPosition:
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
    # create the output directories if they don't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(output_dir_missing):
        os.makedirs(output_dir_missing)

    if not os.path.exists(output_dir_match):
        os.makedirs(output_dir_match)

    # prompt the user which functions the user wants to use
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
                    move_img_ptrs(filepath, output_dir, output_dir_missing, suffix)
                except KeyboardInterrupt:
                    break
            if s == 3:
                try:
                    find_match(filepath, output_dir, cim_models, cim_dir, output_dir_match, suffix)
                except KeyboardInterrupt:
                    break
        
        hrs, mins, secs = calculate_time_elapsed(start_)
        output_messages = ["====================MAIN PROGRAM FINISHED!====================",
                            "Total elapsed time: {} hours {} minutes {} seconds\n".format(hrs, mins, secs)]
        log_and_print(output_messages)

    except KeyboardInterrupt:
        print("Operation Interrupted")
        print("Log file stored in: {}".format(os.path.join(basepath,logname)))


'''
This is the main function
To be modified: basepath, filepath, cim_dir, cim_models
'''
if __name__ == "__main__":
    # start logging
    start_ = time() # to keep time
    ts = datetime.fromtimestamp(start_).strftime('%Y-%m-%d') #time stamp for the log file
    logname = "{}-prepare-img-ptrs.log".format(ts)
    logging.basicConfig(filename=logname, level=logging.DEBUG)
    output_messages = ["====================STARTING MAIN PROGRAM====================",
                        "Operation started at {}\n".format(datetime.now().time())]
    log_and_print(output_messages)
    
    # where you want the outputs to be stored
    basepath = os.path.join(os.getcwd())
    #basepath = "" #uncomment and specify if you want the basepath to be different from the working directory

    # where the multipatient folders are stored
    filepath = "E:\\Original Images"

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

    prepare_img_ptrs(basepath, filepath, suffix, cim_dir, cim_models, output_dir, output_dir_missing, output_dir_match)