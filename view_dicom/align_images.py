from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import h5py
from datetime import datetime
from time import time
from skimage import img_as_ubyte
import logging

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

def align_images(unaligned_img, aligned_img, cine_mask):
    # Convert images to grayscale
    #unaligned_img_byte = cv2.cvtColor(unaligned_img, cv2.COLOR_BGR2GRAY)
    #aligned_img_byte = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)

    unaligned_img_byte = cv2.convertScaleAbs(unaligned_img)
    aligned_img_byte = cv2.convertScaleAbs(aligned_img)
    #aligned_img_byte = aligned_img.astype("uint8")

    #unaligned_img_byte = img_as_ubyte(unaligned_img)
    #aligned_img_byte = img_as_ubyte(aligned_img)


    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(unaligned_img_byte, None)
    keypoints2, descriptors2 = orb.detectAndCompute(aligned_img_byte, None)

    #print(keypoints1, descriptors1)
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    
    # Draw top matches
    imMatches = cv2.drawMatches(unaligned_img_byte, keypoints1, aligned_img_byte, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)
    
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    print(points1)
    print(points2)
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    print(h)
    
    # Use homography
    height, width = aligned_img.shape
    cine_aligned = cv2.warpPerspective(unaligned_img, h, (width, height))
    aligned_mask = cv2.warpPerspective(cine_mask, h, (width, height))
    
    return cine_aligned, aligned_mask

def get_data_from_hdf5(h5py_file):
    with h5py.File(h5py_file, 'r') as hf:
        group = hf.get("train")
        patient_names = np.array(group.get("patients"))
        cine_imgs = np.array(group.get("cine_images"))
        tagged_imgs = np.array(group.get("tagged_images"))

    return patient_names, cine_imgs, tagged_imgs

def main():
    '''
    if turning into a function we need: filepath, ptr_path
    '''
    # where the h5py files are located
    h5py_file = "C:\\Users\\arad572\\Documents\\Summer Research\\code\\prepare_data\\h5_files\\UK_Biobank_10cases.h5"

    patient_names, cine_images, tagged_images = get_data_from_hdf5(h5py_file)

    for i in range(len(patient_names)):
        if i == 0:
            #plot_images(patient_names[i], cine_imgs[i][0], tagged_imgs[i][0], landmark_coords[i][0], save_image=False)
            print("Aligning images ...")
            cine_aligned, h = align_images(cine_images[i][0], tagged_images[i][0])

            out_filename = "{} aligned.jpg".format(patient_names[i])
            print("Saving aligned image :", out_filename)
            cv2.imwrite(out_filename, cine_aligned)

            print("Estimated homography : \n", h)        

if __name__ == "__main__":
    main()