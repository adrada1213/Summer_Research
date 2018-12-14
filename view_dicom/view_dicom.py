import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import pydicom
import h5py
import glob
import fnmatch
from datetime import datetime
from time import time
import logging

def calculate_time_elapsed(start):
    end = time()
    hrs = (end-start)//60//60
    mins = ((end-start) - hrs*60*60)//60
    secs = int((end-start) - mins*60 - hrs*60*60)

    return hrs, mins, secs

def get_cim_info(patient_name, cim_patients):
    try:
        cim_ptr_path = [p for p in cim_patients if patient_name.replace("_Bio", "").lower() in p.lower()][0] #get the cim path of current patient
        cim_pat_name = os.path.basename(cim_ptr_path)
        cim_model_name = os.path.basename(os.path.dirname(cim_ptr_path))
    except IndexError:   
        return None, None

    return cim_model_name, cim_pat_name

def get_first_frames(ptr_path):
    # read data from text
    datatype = [('series', '<i4'), ('slice', '<i4'), ('index', '<i4'), ('path', 'U255')]
    ptr_content = np.genfromtxt(ptr_path, delimiter='\t', names='series, slice, index, path', skip_header=1, dtype=datatype)

    # obtained the tagged series slices and cine series slices separately
    cine_con = np.logical_and(ptr_content["series"] == 0, ptr_content["index"] == 0)
    tagged_con = np.logical_and(ptr_content["series"] == 1, ptr_content["index"] == 0)
    cine_first_frames = ptr_content[cine_con]
    tagged_first_frames = ptr_content[tagged_con]

    return cine_first_frames, tagged_first_frames

def get_data_from_hdf5(patient_name, cim_pat_name, cim_model_name, slice_num, h5py_files):
    # gets the paths of the h5 files of the specified observer
    paths = [f for f in h5py_files if cim_model_name in f]

    # loop through the paths
    for p in paths:
        with h5py.File(p, 'r') as hf:
            patients = np.array(hf.get("patients"))
            p_indices = np.array(np.where(patients==cim_pat_name))[0]
            if len(p_indices) != 0:
                try:
                    p_index = p_indices[slice_num]
                    ed_coords = np.array(hf.get("ed_coords"))[p_index][0]   #only get the x and y coordinates in the ed frame
                    return ed_coords
                except IndexError:
                    return None
            else:
                continue        

def resize_images(cine_img, tagged_img):
    black = [0, 0, 0]
    tag_h, tag_w = tagged_img.shape
    h_diff = 256-tag_h
    w_diff = 256-tag_w

    cine_img  = cv2.resize(cine_img, (tag_w,tag_h), interpolation = cv2.INTER_CUBIC)
    tagged_resized = cv2.copyMakeBorder(tagged_img, h_diff//2, h_diff-(h_diff//2), w_diff//2, w_diff-(w_diff)//2, cv2.BORDER_CONSTANT, value=black )
    cine_resized = cv2.copyMakeBorder(cine_img, h_diff//2, h_diff-(h_diff//2), w_diff//2, w_diff-(w_diff)//2, cv2.BORDER_CONSTANT, value=black )

    return cine_resized, tagged_resized

def plot_images(patient_name, cine_img, tagged_img, ed_coords, pp_diff, save_image):
    cine_resized, tagged_resized = resize_images(cine_img, tagged_img)

    # overlap the two images
    overlap = cv2.addWeighted(cine_resized, 0.5, tagged_resized, 1, 0)
    
    # plot the data
    fig,([ax1, ax2, ax3]) = plt.subplots(1,3)

    fig.set_tight_layout(True)
    '''
    for i in range(256):
        for j in range(256):
            print("x: {}|y: {}|value: {}".format(i, j, cine_resized[i][j]))

    return
    '''
    # add the images to the axes
    ax1.imshow(cine_resized, cmap = 'gray')
    ax2.imshow(overlap, cmap = 'gray')
    ax3.imshow(tagged_resized, cmap = 'gray')

    # add the landmark points to the axes
    #print(type(ed_coords[0][0]))
    ax1.scatter(ed_coords[0], ed_coords[1], s=2, color="cyan")
    ax1.scatter(ed_coords[0][0:7], ed_coords[1][0:7], s=2, color="yellow")
    
    ax2.scatter(ed_coords[0], ed_coords[1], s=2, color="cyan")
    ax2.scatter(ed_coords[0][0:7], ed_coords[1][0:7], s=2, color="yellow")

    ax3.scatter(ed_coords[0], ed_coords[1], s=2, color="cyan")
    ax3.scatter(ed_coords[0][0:7], ed_coords[1][0:7], s=2, color="yellow")
    
    # remove the tick marks and labels from the axes
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    if (save_image):
        output_dir = os.path.join(os.getcwd(), "images")
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        plt.savefig("{}".format(os.path.join(output_dir, patient_name)))
    else:
        plt.show()

def main():
    '''
    if turning into a function we need: filepath, ptr_path
    '''
    # where multipatient folders are located
    filepath = "E:\\Original Images\\2015"

    # where the pointers with matching cine and tagged series are located
    ptr_files_path = "C:\\Users\\arad572\\Documents\\Summer Research\\code\\prepare_data\\img_ptrs\\new_matches"

    # where the h5py files are located
    h5py_filepath = "C:\\Users\\arad572\\Documents\\MR-tagging\\dataset-localNet\\data_sequence_original"

    # where the cim folders are
    cim_path = "C:\\Users\\arad572\\Downloads\\all CIM"
    cim_dirnames = ["CIM_DATA_AB", "CIM_DATA_EL1", "CIM_DATA_EL2", "CIM_DATA_EM", "CIM_DATA_KF", "CIM_Data_ze_1", "CIM_DATA_ze_2", "CIM_DATA_ze_3", "CIM_DATA_ze_4"]
    
    # get the pointer files
    ptr_files = [f for f in os.listdir(ptr_files_path) if f.endswith(".img_imageptr")]

    # get the list of h5py files
    h5py_files = [os.path.join(h5py_filepath, f) for f in os.listdir(h5py_filepath) if fnmatch.fnmatch(f, "*.seq.noresize.?.h5")]

    # get the list of cim directories
    cim_models = [os.path.join(cim_path, d) for d in cim_dirnames]
    cim_patients = []
    for model in cim_models:
        cim_patients += [os.path.join(model, p) for p in os.listdir(model)]

    for i, ptr in enumerate(ptr_files):
        if i == 1 or i == 7 or i == 9: #test for the first one
            # get the location of the current pointer
            ptr_path = os.path.join(ptr_files_path, ptr)
            patient_name = ptr.replace("_match.img_imageptr", "")

            # get the cim model name for the current patient
            cim_model_name, cim_pat_name = get_cim_info(patient_name, cim_patients)
            #print(type(patient_name), type(cim_model_name), type(cim_pat_name))

            cine_first_frames, tagged_first_frames = get_first_frames(ptr_path)

            # to handle image pointers where not all tagged slices have a matching slice in the cine series
            slice_num = cine_first_frames[:]["slice"]
            if len(slice_num) > len(tagged_first_frames[:]["slice"]):
                slice_num = tagged_first_frames[:]["slice"]

            # loop through the slices
            for j in slice_num:
                
                cine_image_file = cine_first_frames[cine_first_frames["slice"] == j]["path"][0].replace("IMAGEPATH" , filepath) 
                tagged_image_file = tagged_first_frames[tagged_first_frames["slice"] == j]["path"][0].replace("IMAGEPATH" , filepath) 
                if not os.path.exists(cine_image_file):
                    cine_image_file = cine_image_file.replace("\\2015\\", "\\2014\\")
                    tagged_image_file = tagged_image_file.replace("\\2015\\", "\\2014\\")
                ds_cine = pydicom.dcmread(cine_image_file)
                ds_tagged = pydicom.dcmread(tagged_image_file)

                # get the pixel arrays of both images
                cine_img = ds_cine.pixel_array
                tagged_img = ds_tagged.pixel_array

                # get the difference between image patient position
                cine_img_info = np.array([ds_cine.ImageOrientationPatient, ds_cine.ImagePositionPatient])
                tagged_img_info = np.array([ds_tagged.ImageOrientationPatient, ds_tagged.ImagePositionPatient])
                pp_diff = np.subtract(cine_img_info[1], tagged_img_info[1])  
                
                # get the end diastole coordinates from hdf5 files
                try:
                    ed_coords = get_data_from_hdf5(patient_name, cim_pat_name, cim_model_name, j, h5py_files)
                except:
                    logger.info("Oh no")
                    continue
                try:
                    plot_images(patient_name, cine_img, tagged_img, ed_coords, pp_diff, save_image=True)
                except:
                    continue

# start logging
start = time() # to keep time
ts = datetime.fromtimestamp(start).strftime('%Y-%m-%d') #time stamp for the log file
logname = "{}-test.log".format(ts)
logging.basicConfig(filename=logname, level=logging.DEBUG)
#logging.basicConfig(filename="test.log", level=logging.DEBUG)
logger = logging.getLogger(__name__)
output_messages = ["====================STARTING MAIN PROGRAM====================",
                    "Operation started at {}\n".format(datetime.now().time())]

                
main()