import matplotlib.pyplot as plt
import numpy as np
import imutils
import os
import cv2
import h5py
from math import pi


'''
code largely based on: https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
for detecting circle. circularity is calculated using the formula: 4pi(area/perimeter^2)
it's perfectly circle if it's equal to 1. We're gonna take the contour with the highest 
circularity.
'''
def localise(patient_name, cine_img, tagged_img):
    # convert dtype to uint8 from uint16 so we can use cv2 functions
    cine_img = cv2.convertScaleAbs(cine_img)
    cine_img_resized = cine_img[90:170,:]
    tagged_img = cv2.convertScaleAbs(tagged_img)
    cine_img_3layers = cv2.cvtColor(cine_img_resized, cv2.COLOR_GRAY2RGB)

    cine_bin = cv2.threshold(cine_img_resized, 127, 255, cv2.THRESH_BINARY)[1]
    '''
    cv2.imshow("cine", thresh1)
    cv2.waitKey(0)
    cv2.destroyAllWindows
    '''
    contours = cv2.findContours(cine_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)

    max_circ = 0
    max_area = 0
    max_perimeter = 0
    circ_contour = contours[0]
    #loop through the contours and find the contour of the circle
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0 or area < 300:
            continue
        else:
            circularity = 4*pi*(area/(perimeter*perimeter))
            if circularity > max_circ:
                max_area = area
                max_perimeter = perimeter
                max_circ = circularity
                circ_contour = c
    
    print("Contour area :", max_area)
    print("Contour perimeter :",max_perimeter)
    print("Circularity :",max_circ)
    #M = cv2.moments(circ_contour)
    #cX = int((M["m10"] / M["m00"]))
    #cY = int((M["m01"] / M["m00"]))

    cv2.drawContours(cine_img_3layers, [circ_contour], 0, (0, 255, 0), 2)
    plt.imshow(cine_img_3layers, cmap="gray")
    plt.show()
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
        

    return perimeter

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
    h5py_file = "C:\\Users\\arad572\\Documents\\Summer Research\\code\\prepare_data\\h5_files\\UK_Biobank_50cases.h5"

    patient_names, cine_imgs, tagged_imgs = get_data_from_hdf5(h5py_file)

    for i in range(len(patient_names)):
        roi = localise(patient_names[i], cine_imgs[i][0], tagged_imgs[i][0])


main()