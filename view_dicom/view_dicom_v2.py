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

def get_data_from_hdf5(h5py_file, grp):
    with h5py.File(h5py_file, 'r') as hf:
        group = hf["/{}".format(grp)]
        cine_group = hf["/{}/cine".format(grp)]
        tagged_group = hf["/{}/tagged".format(grp)]

        patient_names = np.array(group.get("patients")[:20])
        cine_imgs = np.array(cine_group.get("cine_images")[:20,:,:,:])
        tagged_imgs = np.array(tagged_group.get("tagged_images")[:20,:,:,:])
        landmark_coords = np.array(group.get("landmark_coords")[:20,:,:,:])
        cine_dicom_paths = np.array(cine_group.get("cine_dicom_paths")[:20,:])
        tagged_dicom_paths = np.array(tagged_group.get("tagged_dicom_paths")[:20,:])
        print(patient_names.shape)
        print(cine_imgs.shape)
        print(tagged_imgs.shape)
        print(landmark_coords.shape)
        print(cine_dicom_paths.shape)
        print(tagged_dicom_paths.shape)

    return patient_names, cine_imgs, tagged_imgs, landmark_coords, cine_dicom_paths, tagged_dicom_paths

def plot_images(patient_name, cine_img, tagged_img, ed_coords, save_image):
    # overlap the two images
    overlap = cv2.addWeighted(cine_img, 0.5, tagged_img, 1, 0)
    
    # plot the data
    

    fig,([ax1, ax2, ax3]) = plt.subplots(1,3)

    fig.set_tight_layout(True)
 
    # add the images to the axes
    ax1.imshow(cine_img, cmap = 'gray')
    ax2.imshow(overlap, cmap = 'gray')
    ax3.imshow(tagged_img, cmap = 'gray')

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

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
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
    # where the h5py files are located
    h5py_file = "C:\\Users\\arad572\\Documents\\Summer Research\\code\\prepare_data\\h5_files\\UK_Biobank_50cases.h5"

    group = "train"
    patient_names, cine_imgs, tagged_imgs, landmark_coords, cine_dicom_paths, tagged_dicom_paths = get_data_from_hdf5(h5py_file, group)

    for i in range(len(patient_names)):
        print(patient_names[i])
        plot_images(patient_names[i], cine_imgs[i][0], tagged_imgs[i][0], landmark_coords[i][0], save_image=False)


# start logging
start = time() # to keep time
ts = datetime.fromtimestamp(start).strftime('%Y-%m-%d') #time stamp for the log file
logname = "{}-viewing-from-created-h5.log".format(ts)
logging.basicConfig(filename=logname, level=logging.DEBUG)
#logging.basicConfig(filename="test.log", level=logging.DEBUG)
logger = logging.getLogger(__name__)
output_messages = ["====================STARTING MAIN PROGRAM====================",
                    "Operation started at {}\n".format(datetime.now().time())]

                
main()