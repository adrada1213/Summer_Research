"""
This script contains functions involving dicom files.
(ref code for get_3D_corners, and the conversions: CIM modeller source code)
Author: Amos Rada
Date:   25/02/2019
"""
# import libraries
import numpy as np
import os
import pydicom
import random

def get_patient_name(pat_path):
    '''
    Get the name of the patient from the dicom header
    Input:
        pat_path (string) = path to the dicom files of the patient
    Output:
        patient_name (string) = name of the patient based on the dicom header ('^' replaced with '_')
    Note: This function can be simplified.
    '''
    files = [f for f in os.listdir(pat_path) if f.endswith(".dcm")]
    try:
        random_index = random.randint(0,len(files))
    except ValueError:
        random_index = 0
    try:
        pfile = files[random_index]   #get one patient file
    except IndexError:
        pfile = files[random_index//2]
    ds = pydicom.dcmread(os.path.join(pat_path, pfile), specific_tags=["PatientName"])    #read the patient name
    try:
        patient_name = str(ds.PatientName).replace("^", "_")
    except AttributeError:
        patient_name = get_patient_name(pat_path)
        
    return patient_name

def get_3D_corners(dicom_header):
    '''
    This functions extracts the 3D coordinates (based on the MRI coordinate system) of the top left corner, 
    top right corner, and the bottom left corner of a dicom image.
    Input:
        dicom_header = header of the dicom file (read using pydicom)
    Ouput:
        blc (1x3 array) = global coordinates of the bottom left corner
        tlc (1x3 array) = global coordinates of the top left corner
        trc (1x3 array) = global coordinates of the top right corner
    '''
    # initialise corners
    tlc = []
    trc = []
    blc = []

    # get required info from dicom header
    img_size = dicom_header.pixel_array.shape #(height, width)
    img_pos = dicom_header.ImagePositionPatient
    img_orient = dicom_header.ImageOrientationPatient
    px_size = dicom_header.PixelSpacing[0]
    fov_x = px_size*img_size[1]
    fov_y = px_size*img_size[0]

    # calculate the corners
    for i in range(3):  #loops through x, y, z axis
        tlc.append(img_pos[i]-px_size*0.5*(img_orient[i]+img_orient[i+3]))
        trc.append(tlc[i]+fov_x*img_orient[i])
        blc.append(tlc[i]+fov_y*img_orient[i+3])

    return np.array(blc), np.array(tlc), np.array(trc)

def convert_3D_points_to_2D(points_3D, dicom_path):
    '''
    This function converts points in the MRI coordinate system (3D) into 2D points (coordinate
    system of the 2D dicom image).
    Inputs:
        points_3D (Nx3 list/array) = 3D points
        dicom_path (string) = path to the dicom image
    Output:
        points_2D (Nx2 list) = 2D points
    '''
    points_2D = []

    # read the dicom file first
    ds = pydicom.dcmread(dicom_path)

    # get the image size 
    img_size = ds.pixel_array.shape

    # get 3D corners of the dicom image (bottom left corner, top left corner, top right corner)
    blc, tlc, trc = get_3D_corners(ds)

    xside = np.subtract(trc, tlc)
    yside = np.subtract(blc, tlc) #instead of tlc-blc (from cim program), we do blc-tlc

    r1 = img_size[1]/np.dot(xside, xside)
    r2 = img_size[0]/np.dot(yside, yside)

    # loop through each point and convert to 2D
    for point in points_3D:
        transform = np.subtract(np.array(point), tlc) #we transform from tlc instead of blc (cim)

        x_coord = np.dot(transform, xside)*r1-0.5
        y_coord = np.dot(transform, yside)*r2-0.5
        
        points_2D.append([x_coord, y_coord])

    return points_2D

def convert_2D_points_to_3D(points_2D, dicom_path):
    '''
    This function converts points in the 2D dicom coordinate system (2D) into 3D (MRI coordinate
    system).
    Inputs:
        points_2D (Nx2 list/array) = 2D points
        dicom_path (string) = path to the dicom image
    Output:
        points_3D (Nx3 list) = 3D points
    '''
    # initialist list of 3D points
    points_3D = []

    # read the dicom file first
    ds = pydicom.dcmread(dicom_path)

    # get image size
    img_size = ds.pixel_array.shape #(height, width)

    # get 3D corners of the dicom image (bottom left corner, top left corner, top right corner)
    blc, tlc, trc = get_3D_corners(ds)

    xside = np.subtract(trc, tlc)
    yside = np.subtract(blc, tlc) #instead of tlc-blc (from cim program), we do blc-tlc

    # loop through each point and convert to 3D
    for point in points_2D:
        scalex = (point[0] + 0.5)/img_size[1]
        scaley = (point[1] + 0.5)/img_size[0]

        x_coord = tlc[0] + scalex*(xside[0]) + scaley*(yside[0])
        y_coord = tlc[1] + scalex*(xside[1]) + scaley*(yside[1])
        z_coord = tlc[2] + scalex*(xside[2]) + scaley*(yside[2])

        points_3D.append([x_coord, y_coord, z_coord])

    return points_3D