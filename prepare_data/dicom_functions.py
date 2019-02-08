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
from prepare_data_functions import get_cim_path, get_slices, log_and_print
from math import sqrt
import logging
import random

logger = logging.getLogger(__name__)

def plot_images(patient_name, cine_image, tagged_image, cine_ed_coords, tagged_ed_coords, save_image):
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
    
    # remove the tick marks and labels from the axes
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
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

def get_image_path(gen_imagepath, filepaths):
    imagepath = gen_imagepath.replace("IMAGEPATH", filepaths[0])

    if not(os.path.isfile(imagepath)):
        imagepath = imagepath.replace(filepaths[0], filepaths[1])
    
    return imagepath

def get_dimensions(image, h_, w_):
    h, w = image.shape
    h_diff = h_-h
    w_diff = w_-w

    return h, w, h_diff, w_diff

def view_images(cine_img, tagged_img):
    # overlap the two images
    overlap = cv2.addWeighted(cine_img, 0.5, tagged_img, 1, 0)
    
    # plot the data
    fig,([ax1, ax2, ax3]) = plt.subplots(1,3)

    fig.set_tight_layout(True)

    # add the images to the axes
    ax1.imshow(cine_img, cmap = 'gray')
    ax2.imshow(overlap, cmap = 'gray')
    ax3.imshow(tagged_img, cmap = 'gray')
    '''
    # remove the tick marks and labels from the axes
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    '''

    plt.show()

def get_dicom_info(filepaths, ptr_content, index):
    cine_dicom_paths = []
    tagged_dicom_paths = []
    cine_images = []
    tagged_images = []

    # separate the cine frames and the tagged frames
    slice_con = ptr_content["slice"] == index
    cine_con = np.logical_and(ptr_content["series"]==0, slice_con)
    tagged_con = np.logical_and(ptr_content["series"]==1, slice_con)

    cine_frames = ptr_content[cine_con]
    tagged_frames = ptr_content[tagged_con]

    # only get 20 frames
    for j in range(20):
        if j < len(tagged_frames):
            gen_tagged_imagepath = tagged_frames["path"][j]

            tagged_dicom_paths.append(gen_tagged_imagepath)

            tagged_imagepath = get_image_path(gen_tagged_imagepath, filepaths)

            dst = pydicom.dcmread(tagged_imagepath)
            dst_img = dst.pixel_array
            tag_h, tag_w, h_diff, w_diff = get_dimensions(dst_img, 256, 256)
            try:
                # resize tagged image by padding
                dst_img_res = cv2.copyMakeBorder(dst_img, h_diff//2, h_diff-(h_diff//2), w_diff//2, w_diff-(w_diff//2), cv2.BORDER_CONSTANT, value=[0,0,0])
            except ValueError:
                log_and_print("Dicom width/height larger than 256 for patient {}".format(dst.PatientName))
                aspectRatio = tag_w/tag_h
                if tag_h > 256 and tag_h>tag_w:
                    tag_h = 256
                    tag_w = tag_h * aspectRatio
                elif tag_w >256 and tag_w>tag_h:
                    tag_w = 256
                    tag_h = tag_w/aspectRatio
                dst_img_res = cv2.resize(dst_img, (tag_w, tag_h), interpolation = cv2.INTER_CUBIC)
                tag_h, tag_w, h_diff, w_diff = get_dimensions(dst_img, 256, 256)
                dst_img_res = cv2.copyMakeBorder(dst_img_res, h_diff//2, h_diff-(h_diff//2), w_diff//2, w_diff-(w_diff//2), cv2.BORDER_COSNTANT, value = [0,0,0])

            tagged_images.append(dst_img_res)

            if j == 0:  # we'll use the end diastolic image info to resize our cine frames
                # only take 50 frames
                for k in range(50):
                    if k < len(cine_frames):
                        gen_cine_imagepath = cine_frames["path"][k]
                        cine_dicom_paths.append(gen_cine_imagepath)

                        cine_imagepath = get_image_path(gen_cine_imagepath, filepaths)

                        dsc = pydicom.dcmread(cine_imagepath)
                        dsc_img = dsc.pixel_array
                        if k == 0:
                            x_ratio = tag_w/dsc_img.shape[1]
                            y_ratio = tag_h/dsc_img.shape[0]
                            translation = [w_diff, h_diff]

                        dsc_img_res = cv2.resize(dsc_img, (tag_w,tag_h), interpolation = cv2.INTER_CUBIC)

                        dsc_img_res = cv2.copyMakeBorder(dsc_img_res, h_diff//2, h_diff-(h_diff//2), w_diff//2, w_diff-(w_diff//2), cv2.BORDER_CONSTANT, value=[0,0,0])

                        cine_images.append(dsc_img_res)

                    if len(cine_frames) < 50 and k >= len(cine_frames):
                        if k==len(cine_frames):
                            log_and_print("Adding {} cine frame(s) for patient {}".format(50-len(cine_frames), dsc.PatientName))
                        cine_dicom_paths.append("")
                        cine_images.append(np.zeros((256,256)))

        # if the slice has less than 20 frames
        if len(tagged_frames) < 20 and j >= len(tagged_frames):
            if j==len(cine_frames):
                log_and_print("Adding {} tagged frame(s) for patient {}".format(50-len(cine_frames), dst.PatientName))   
            tagged_dicom_paths.append("")
            tagged_images.append(np.zeros((256,256)))

    cine_images = np.array(cine_images)
    tagged_images = np.array(tagged_images)
            
    return cine_dicom_paths, cine_images, tagged_dicom_paths, tagged_images, x_ratio, y_ratio, translation[0], translation[1]

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

def get_3Dcorners(image_file):
    # initialise corners
    tlc = []
    trc = []
    blc = []

    # get required info from dicom header
    ds = pydicom.dcmread(image_file)
    img_size = ds.pixel_array.shape #(height, width)
    img_pos = ds.ImagePositionPatient
    img_orient = ds.ImageOrientationPatient
    px_size = ds.PixelSpacing[0]
    fov_x = px_size*img_size[1]
    fov_y = px_size*img_size[0]

    # calculate top left corner
    for i in range(3):
        tlc.append(img_pos[i]-px_size*0.5*(img_orient[i]+img_orient[i+3]))
        trc.append(tlc[i]+fov_x*img_orient[i])
        blc.append(tlc[i]+fov_y*img_orient[i+3])

    return np.array(tlc), np.array(trc), np.array(blc), img_size

def convert3D_to_2D(coords, img_size, tlc, trc, blc):
    img_x = []
    img_y = []

    for i in range(len(coords[0])):
        xside = np.subtract(trc, tlc)
        yside = np.subtract(blc, tlc) #instead of tlc-blc, we do blc-tlc

        r1 = img_size[1]/np.dot(xside, xside)
        r2 = img_size[0]/np.dot(yside, yside)

        point_3D = [coords[0][i],coords[1][i],coords[2][i]]
        transform = np.subtract(np.array(point_3D), tlc) #we transform from tlc instead of blc

        img_x.append(np.dot(transform, xside)*r1-0.5)
        img_y.append(np.dot(transform, yside)*r2-0.5)

    img_coords = [img_x, img_y]

    return img_coords