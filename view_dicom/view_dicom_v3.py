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
#from prepare_data_functions import get_cim_path, get_slices, log_and_print
from math import sqrt
import logging

logger = logging.getLogger(__name__)

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

    # remove the tick marks and labels from the axes
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)

    plt.show()

def prepare_dicom_images(filepaths, ptr_content, slices, view):
    cine_dicom_paths = []
    tagged_dicom_paths = []
    cine_images = []
    tagged_images = []
    cine_px_spaces = []
    tagged_px_spaces = []

    # loop through the slices
    for index, i in enumerate(slices):
        tmp_cine_dp = []
        tmp_tagged_dp = []
        tmp_cine_images = []
        tmp_tagged_images = []

        # separate the cine frames and the tagged frames
        slice_con = ptr_content["slice"] == i
        cine_con = np.logical_and(ptr_content["series"]==0, slice_con)
        tagged_con = np.logical_and(ptr_content["series"]==1, slice_con)

        cine_frames = ptr_content[cine_con]
        tagged_frames = ptr_content[tagged_con]

        # only get 20 frames
        for j in range(20):
            if j < len(tagged_frames):
                gen_tagged_imagepath = tagged_frames["path"][j]

                tmp_tagged_dp.append(gen_tagged_imagepath)

                tagged_imagepath = get_image_path(gen_tagged_imagepath, filepaths)

                dst = pydicom.dcmread(tagged_imagepath)
                dst_img = dst.pixel_array
                tag_h, tag_w, h_diff, w_diff = get_dimensions(dst_img, 256, 256)
                try:
                    dst_img_res = cv2.copyMakeBorder(dst_img, h_diff//2, h_diff-(h_diff//2), w_diff//2, w_diff-(w_diff//2), cv2.BORDER_CONSTANT, value=[0,0,0])
                except ValueError:
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

                tmp_tagged_images.append(dst_img_res)

                if j == 0:  # we'll use the end diastolic image info to resize our cine frames
                    # only take 50 frames
                    for k in range(50):
                        if k < len(cine_frames):
                            gen_cine_imagepath = cine_frames["path"][k]
                            tmp_cine_dp.append(gen_cine_imagepath)

                            cine_imagepath = get_image_path(gen_cine_imagepath, filepaths)

                            dsc = pydicom.dcmread(cine_imagepath)
                            dsc_img = dsc.pixel_array
                            dsc_img_res = cv2.resize(dsc_img, (256,256), interpolation = cv2.INTER_CUBIC)
                            #dsc_img_res = cv2.copyMakeBorder(dsc_img_res, h_diff//2, h_diff-(h_diff//2), w_diff//2, w_diff-(w_diff//2), cv2.BORDER_CONSTANT, value=[0,0,0])

                            tmp_cine_images.append(dsc_img_res)

                        if len(cine_frames) < 50 and k >= len(cine_frames):
                            #if k==len(cine_frames):
                                #log_and_print("Adding {} cine frame(s) for patient {}".format(50-len(cine_frames), dsc.PatientName))
                            tmp_cine_dp.append("")
                            tmp_cine_images.append(np.zeros((208,168)))

            # if the slice has less than 20 frames
            if len(tagged_frames) < 20 and j >= len(tagged_frames):
                #if j==len(cine_frames):
                    #log_and_print("Adding {} tagged frame(s) for patient {}".format(50-len(cine_frames), dst.PatientName))   
                tmp_tagged_dp.append("")
                tmp_tagged_images.append(np.zeros((256,256)))
    
        cine_dicom_paths.append(tmp_cine_dp)
        tagged_dicom_paths.append(tmp_tagged_dp)
        cine_images.append(tmp_cine_images)
        tagged_images.append(tmp_tagged_images)
        cine_px_spaces.append(dsc.PixelSpacing[0])
        tagged_px_spaces.append(dst.PixelSpacing[0])

        if index == len(slices)-1:
            cine_images = np.array(cine_images)
            tagged_images = np.array(tagged_images)

        if (view):
            view_images(cine_images[index][0], tagged_images[index][0]) #viewing the ED frame

    cine_px_spaces = np.array(cine_px_spaces)
    tagged_px_spaces = np.array(tagged_px_spaces)
            
    return cine_dicom_paths, tagged_dicom_paths, cine_images, tagged_images, cine_px_spaces, tagged_px_spaces 
                
if __name__ == "__main__":
    '''
    if turning into a function we need: filepath, ptr_path
    '''
    # where multipatient folders are located
    filepaths = ["E:\\Original Images\\2014", "E:\\Original Images\\2015"]

    # path to the image pointer
    ptr_files_path = "C:\\Users\\arad572\\Documents\\Summer Research\\code\\prepare_data\\img_ptrs\\new_matches_final"
    ptr_filename = "2U_JA_59_NV_Bio_match.img_imageptr"
    ptr = os.path.join(ptr_files_path, ptr_filename)

    # read the content of the image pointer
    datatype = [('series', '<i4'), ('slice', '<i4'), ('index', '<i4'), ('path', 'U255')]
    ptr_content = np.genfromtxt(ptr, delimiter='\t', names='series, slice, index, path', skip_header=1, dtype=datatype)

    # specify the slices to be resized
    slices = [0, 1, 2]
    slices = np.array(slices)

    view = True

    cine_dicom_paths, tagged_dicom_paths, cine_images, tagged_images, cine_px_spaces, tagged_px_spaces = prepare_dicom_images(filepaths, ptr_content, slices, view)