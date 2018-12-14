from prepare_h5_files import get_data_from_h5_file
import logging
import os
from prepare_data_functions import get_cim_patients, get_cim_path, log_and_print, get_slices
from datetime import datetime
from time import time
import numpy as np

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # start logging
    start = time() # to keep time
    ts = datetime.fromtimestamp(start).strftime('%Y-%m-%d') #time stamp for the log file
    logname = "{}-check-for-duplicates.log".format(ts)
    logging.basicConfig(filename=logname, level=logging.DEBUG)
    output_messages = ["====================STARTING MAIN PROGRAM====================",
                        "Operation started at {}".format(datetime.now().time())]
    log_and_print(output_messages)

    # where the multipatient files are stored
    filepaths = ["E:\\Original Images\\2014", "E:\\Original Images\\2015"]

    # where the pointer files with matching series and cim files
    ptr_files_path = "C:\\Users\\arad572\\Documents\\Summer Research\\code\\prepare_data\\img_ptrs\\new_matches_final"

    # edwards h5 files
    eds_h5_filepath = "C:\\Users\\arad572\\Documents\\MR-tagging\\dataset-localNet\\data_sequence_original"

    # where the cim models are
    cim_dir = "C:\\Users\\arad572\\Downloads\\all CIM"
    cim_models = ["CIM_DATA_AB", "CIM_DATA_EL1", "CIM_DATA_EL2", "CIM_DATA_EM", "CIM_DATA_KF", "CIM_Data_ze_1", "CIM_DATA_ze_2", "CIM_DATA_ze_3", "CIM_DATA_ze_4"]
    cim_patients = get_cim_patients(cim_dir, cim_models)

    # get all the pointer file paths
    ptr_files = [f for f in os.listdir(ptr_files_path) if f.endswith("_match.img_imageptr")]

    for i, ptr in enumerate(ptr_files):
        patient_name = ptr.replace("_match.img_imageptr", "")   #get the patient name

        # get the cim path for the patient MODEL\PatientName
        cim_path = get_cim_path(patient_name, cim_patients)

        # get the path of the pointer
        ptr_path = os.path.join(ptr_files_path, ptr)

        # read the content of the image pointer
        datatype = [('series', '<i4'), ('slice', '<i4'), ('index', '<i4'), ('path', 'U255')]
        ptr_content = np.genfromtxt(ptr_path, delimiter='\t', names='series, slice, index, path', skip_header=1, dtype=datatype)

        ptr_slices = get_slices(ptr_content)

        print("Checking patient {} of {}".format(i+1, len(ptr_files)))
        patient_names, cim_paths, slices, bbox_corners, landmark_coords = get_data_from_h5_file(eds_h5_filepath, cim_path, patient_name, ptr_slices)

