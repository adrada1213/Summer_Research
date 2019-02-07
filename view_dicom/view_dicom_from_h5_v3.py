import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    start = 204
    end = 207
    with h5py.File(h5py_file, 'r') as hf:
        group = hf["/{}".format(grp)]
        cine_group = hf["/{}/cine".format(grp)]
        tagged_group = hf["/{}/tagged".format(grp)]

        patient_names = np.array(group.get("patients")[start:end])
        cine_images = np.array(cine_group.get("images")[start:end])
        cine_landmark_coords = np.array(cine_group.get("landmark_coords")[start:end])
        cine_centroids = np.array(cine_group.get("centroids")[start:end])
        cine_es_indices = np.array(cine_group.get("es_indices")[start:end])

        tagged_images = np.array(tagged_group.get("images")[start:end])
        tagged_landmark_coords = np.array(tagged_group.get("landmark_coords")[start:end])
        tagged_centroids = np.array(tagged_group.get("centroids")[start:end])
        tagged_es_indices = np.array(tagged_group.get("es_indices")[start:end])


    return patient_names, cine_images, tagged_images, cine_landmark_coords, tagged_landmark_coords, cine_es_indices, tagged_es_indices, cine_centroids, tagged_centroids

def plot_images(patient_name, cine_image, tagged_image, cine_ed_coords, tagged_ed_coords, cine_bbox, tagged_bbox, save_image):
    # plot the data
    

    fig,([ax1, ax2]) = plt.subplots(1,2)

    fig.set_tight_layout(True)
 
    # add the images to the axes
    ax1.imshow(cine_image, cmap = 'gray')
    ax2.imshow(tagged_image, cmap = 'gray')

    # add the landmark points to the axes
    #print(type(ed_coords[0][0]))
    ax1.scatter(cine_ed_coords[0], cine_ed_coords[1], s=2, color="cyan")
    ax1.scatter(cine_ed_coords[0][0:7], cine_ed_coords[1][0:7], s=2, color="yellow")
    
    ax2.scatter(tagged_ed_coords[0], tagged_ed_coords[1], s=2, color="cyan")
    ax2.scatter(tagged_ed_coords[0][0:7], tagged_ed_coords[1][0:7], s=2, color="yellow")
    
    box1 = patches.Rectangle((cine_bbox[0],cine_bbox[1]), (cine_bbox[2]-cine_bbox[0]), (cine_bbox[3]-cine_bbox[1]), fill=False, linewidth=1, color='green')
    box2 = patches.Rectangle((tagged_bbox[0],tagged_bbox[1]), (tagged_bbox[2]-tagged_bbox[0]), (tagged_bbox[3]-tagged_bbox[1]), fill=False, linewidth=1, color='green')

    ax1.add_patch(box1)
    ax2.add_patch(box2)
    # remove the tick marks and labels from the axes
    ax1.get_xaxis().set_visible(False)
    #ax1.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

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
    h5py_file = "C:\\Users\\arad572\\Documents\\Summer Research\\Summer Research Code\\prepare_data\\h5_files\\UK_Biobank.h5"

    group = "train"
    patient_names, cine_images, tagged_images, cine_landmark_coords, tagged_landmark_coords, cine_es_indices, tagged_es_indices, cine_centroids, tagged_centroids = get_data_from_hdf5(h5py_file, group)

    print(cine_centroids[0])
    for i in range(len(patient_names)):
        print(patient_names[i])
        cine_bbox = np.array([cine_centroids[i][0]-cine_centroids[i][2], cine_centroids[i][1]-cine_centroids[i][2], cine_centroids[i][0]+cine_centroids[i][2], cine_centroids[i][1]+cine_centroids[i][2]])
        tagged_bbox = np.array([tagged_centroids[i][0]-tagged_centroids[i][2], tagged_centroids[i][1]-tagged_centroids[i][2], tagged_centroids[i][0]+tagged_centroids[i][2], tagged_centroids[i][1]+tagged_centroids[i][2]])
        #cine_bbox = np.array([cine_centroids[i][0]-cine_centroids[i][2]/2, cine_centroids[i][1]-cine_centroids[i][2]/2, cine_centroids[i][0]+cine_centroids[i][2]/2, cine_centroids[i][1]+cine_centroids[i][2]/2])
        #tagged_bbox = np.array([tagged_centroids[i][0]-tagged_centroids[i][2]/2, tagged_centroids[i][1]-tagged_centroids[i][2]/2, tagged_centroids[i][0]+tagged_centroids[i][2]/2, tagged_centroids[i][1]+tagged_centroids[i][2]/2])
        for j in range(2):     
            if j ==0:
                plot_images(patient_names[i], cine_images[i][0], tagged_images[i][0], cine_landmark_coords[i][0], tagged_landmark_coords[i][0], cine_bbox, tagged_bbox, save_image=False)
            else:
                if cine_es_indices[i] != -1:
                    plot_images(patient_names[i], cine_images[i][cine_es_indices[i]], tagged_images[i][tagged_es_indices[i]], cine_landmark_coords[i][1], tagged_landmark_coords[i][tagged_es_indices[i]], cine_bbox, tagged_bbox, save_image=False)
                else:
                    print("No ES index")

# start logging
start = time() # to keep timek
ts = datetime.fromtimestamp(start).strftime('%Y-%m-%d') #time stamp for the log file
logname = "{}-viewing-from-created-h5.log".format(ts)
logging.basicConfig(filename=logname, level=logging.DEBUG)
#logging.basicConfig(filename="test.log", level=logging.DEBUG)
logger = logging.getLogger(__name__)
output_messages = ["====================STARTING MAIN PROGRAM====================",
                    "Operation started at {}\n".format(datetime.now().time())]

                
main()