import cv2
import numpy as np
import h5py

h5_file = "C:\\Users\\arad572\\Documents\\Summer Research\\code\\prepare_data\\h5_files\\UK_Biobank_20cases.h5"

with h5py.File(h5_file, "r") as hf:
    train = hf.get("train")
    cine_info = hf["/train/cine"]
    cine_dicom_paths = np.array(cine_info.get("cine_dicom_paths"))[0,0]
    patient_name = np.array(train.get("patients"))[0]
    print(cine_dicom_paths)
    print(patient_name)



