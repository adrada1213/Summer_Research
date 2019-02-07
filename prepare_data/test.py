import os
import csv
import numpy as np
from prepare_data_functions import get_slices
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pydicom
'''
This program will be used to:
-Find ES frame
-Get the points of the contours

'''

def read_mapping_file(mapping_file):
    with open(mapping_file, mode="r") as csv_file:
        f_ids = []
        p_ids = []
        csv_reader = csv.reader(csv_file, delimiter=',')

        for i, row in enumerate(csv_reader):
            if i != 0:
                f_ids.append(row[0][1:])
                p_ids.append(row[1][:8])

    return f_ids, p_ids

def get_uids_and_slice_ids(slice_info):
    with open(slice_info) as f:
        uids = []
        slice_ids = []
        for line in f:
            if "SOPInstanceUID" in line:
                # get the uid and corresponding slice id
                uid =  line.split(":")[1].strip()
                line = f.readline()
                slice_id = int(line.split(":")[1].strip())

                # append them to list
                uids.append(uid)
                slice_ids.append(slice_id)
    
    return uids, slice_ids

'''
This function extracts the slice ids (from slice info file) of the slices in the image pointer.
How it works:
    1. Initialise the lists needed to be returned
    2. Get the filepaths of the ed and es slice info files
    3. Get the uids and slice ids form the files
    4. Read the pointer and get the cine frames
    5. Loop through the cine frames
        -Index of 0 indicates ed frame
        -uid is extracted from the image pointer
        -checks the current uid is in the list of uids
        -if yes, get the slice id
        (For ed, the index will always be 0. For es, the index will be taken from the frame that contains the uid from slice info file)
    6. Return lists
'''
def get_slices_and_indices(f_path, ptr_content):
    # initialise the variables we need to return
    ed_slices = []
    ed_indices = []
    es_slices = []
    es_indices = []

    # get the filepaths for the file containing  the slice info
    ed_slice_info = os.path.join(f_path, "ED", "SliceInfoFile.txt")
    es_slice_info = os.path.join(f_path, "ES", "SliceInfoFile.txt")

    # read the ed and es slice info file and get the uids and slice ids
    ed_uids, ed_slice_ids = get_uids_and_slice_ids(ed_slice_info)
    es_uids, es_slice_ids = get_uids_and_slice_ids(es_slice_info)
    
    # get the cine slices
    cine_slices = get_slices(ptr_content)
    cine_frames = ptr_content[ptr_content["series"] == 0]

    # loop through each slice in the image pointer
    for i, sl in enumerate(cine_slices):
        # loop through the frames in that slice
        frames = cine_frames[cine_frames["slice"]==sl]
        for frame in frames:
            uid = os.path.basename(frame["path"].replace(".dcm", ""))
            if uid in ed_uids:
                # obtain the slice id of the current image
                index = ed_uids.index(uid)
                slice_id = ed_slice_ids[index]
                # add to list
                ed_slices.append(slice_id)
                if frame["index"] != 0:
                    print("ED frame not 0 for patient in {}".format(f_path))
                ed_indices.append(frame["index"])
            elif uid in es_uids:
                #print(uid, es_uids)
                # obtain the slice id of the current image
                index = es_uids.index(uid)
                slice_id = es_slice_ids[index]
                # add to list
                es_slices.append(slice_id)
                es_indices.append(frame["index"])

        if len(ed_slices) < i+1:
            ed_slices.append(-1)
            ed_indices.append(-1)
    
        # some sliceinfofiles do not contain the info for all the slices
        if len(es_slices) < len(ed_slices):
            es_slices.append(-1)
            es_indices.append(-1)
    

    return ed_slices, ed_indices, es_slices, es_indices

def plot_contour_points_3D(contour_pts, ed_slices):

    #%matplotlib inline
    ax = plt.axes(projection="3d")
    for i, ed_slice in enumerate(ed_slices):
        pts = contour_pts[ed_slices[i]]
        ax.scatter3D(pts[0], pts[1], pts[2])
    plt.show()

'''
Plot the contour points with the images
'''
'''
def plot_contour_points(contour_pts, ptr_content, ed_slices):
    #plt.plot(contour_pts[1],contour_pts[2], "ro")
    #plt.show()


    for i, ed_slice in enumerate(ed_slices):
        ax = plt.axes()
        if ed_slice != -1:
            # get the filepath of the cine dicom image
            cine_image_file = ptr_content[ptr_content["slice"] == i]["path"][ed_slice].replace("IMAGEPATH" , "E:\\Original Images\\2015")  
            if not os.path.isfile(cine_image_file):
                cine_image_file = cine_image_file.replace("\\2015\\, \\2014")
            ds_cine = pydicom.dcmread(cine_image_file)  #read the image file

            # get the pixel array
            cine_img = ds_cine.pixel_array

            # show the image in the axis
            ax.imshow(cine_img, cmap="gray")
            
            # show the contour points for the current slice
            pts = contour_pts[i]
            #pts = contour_pts
            ax.scatter(pts[0], pts[1], color="cyan", s=2)

            # show the plot
            plt.show()
'''
def plot_contour_points(contour_pts, cine_image, ed_slice):
    #plt.plot(contour_pts[1],contour_pts[2], "ro")
    #plt.show()

    ax = plt.axes()
    if ed_slice != -1:
        # show the image in the axis
        ax.imshow(cine_image, cmap="gray")
        
        # show the contour points for the current slice
        pts = contour_pts
        #pts = contour_pts
        ax.scatter(pts[0], pts[1], color="cyan", s=2)

        # show the plot
        plt.show()

def get_contour_points(f_path, ed_slices):
    # initialise the dictionary containing the x, y, z coordinates of the contour for each slice
    contour_pts_3D = {}
    for ed_slice in ed_slices:
        contour_pts_3D[int(ed_slice)] = [[],[],[]]

    # get the filepath for the file containing guide points
    ed_gp_file = os.path.join(f_path, "ED", "GPfile.txt")

    # read the file containing the guide points
    with open (ed_gp_file) as f:
           reader = csv.reader(f,delimiter ="\t")
           gpfile_content = list(reader)

    # extract the x, y, z coordinates by going through each line. 
    # We're only interested in the epicardial contour  (saepicardialControu) and the right ventricle septum (RVS)
    # we'll only get the coordinates from the slices that have matches with the tagged images
    for line in gpfile_content:
        if (line[4] == "saepicardialContour" or line[4] == "RVS") and int(line[5]) in ed_slices:
            if (len([line[0]]) == 1 and len([line[1]]) == 1 and len([line[2]])) == 1:
                contour_pts_3D[int(line[5])][0].append(float(line[0]))
                contour_pts_3D[int(line[5])][1].append(float(line[1]))
                contour_pts_3D[int(line[5])][2].append(float(line[2]))
            else:
                print(line[0], line[1], line[2])

    return contour_pts_3D

def convert_3D_points_to_2D(contour_pts_3D):
    contour_pts = []
    for ed_slice in contour_pts_3D:
        pts = contour_pts_3D[ed_slice]
        for i in range(len(pts[0])):
            x = pts[0][i]
            y = pts[1][i]
            z = pts[2][i]
            break

    return contour_pts
            

"""
How this code works:
1. Open csv file -DONE
    -get the folder ids -DONE
    -get the patient ids -DONE
2. Loop through the image pointers in the specified pointer paths (don't need to loop through unecessary LVModels)
    -get the patient name -DONE
    -get the patient id -DONE
    -find the matching folder id for the patient from mapping file -DONE
        -remove the underscores and only take first 8 characters
    -read data from the image pointer file
    -get ed and es slices and indices -DONE
    -get the contour points
    -plot points with the frames
"""

if __name__ == "__main__":
    # specify the location of the mapping file
    mapping_file = "E:\\confidential_bridging_file_r4.csv"

    # specify the location of the image pointers
    ptrs_path = "C:\\Users\\arad572\\Documents\\Summer Research\\code\\prepare_data\\img_ptrs\\matches"

    # specify the location of the modellers
    LVModel_path = "E:\\LVModellerFormatV2"

    # ref. code https://realpython.com/python-csv/
    f_ids, p_ids = read_mapping_file(mapping_file)

    # loop through the image pointers
    count = 0
    for ptr in os.listdir(ptrs_path):
        count += 1
        if count % 500 == 0:
            print("Looped through {}/{} image pointers".format(count, len(os.listdir(ptrs_path))))

        pat_name = ptr.replace("_match.img_imageptr", "")   #get the patient name
        pat_id = pat_name.replace("_", "")[:8]  #get the patient id to find the matching folder in the LVModel path using the csv file
        ptr_path = os.path.join(ptrs_path, ptr) #convert into path
        
        # read the content of the image pointer
        datatype = [('series', '<i4'), ('slice', '<i4'), ('index', '<i4'), ('path', 'U255')]
        ptr_content = np.genfromtxt(ptr_path, delimiter='\t', names='series, slice, index, path', skip_header=1, dtype=datatype)
        
        # get the matching folder of current patient
        # if patient doesn't match any folder, go to the next patient
        try:
            p_index = p_ids.index(pat_id)
        except ValueError:
            continue
        f_id = f_ids[p_index]
        f_path = os.path.join(LVModel_path, f_id)   #get the folder path of the patient

        # get the slices and indices of the ed and es frames
        # if folder doesn't exist, go to the next patient
        try:
            ed_slices, ed_indices, es_slices, es_indices = get_slices_and_indices(f_path, ptr_content)
        except FileNotFoundError:
            continue

        # Found out that some patients have two of the same slice names in the cine series
        if len(ed_slices) > 3 or len(es_slices) > 3:
            print("Patient {} has duplicate slice names")
            continue
        
        # get the contour points for all the ed slices
        contour_pts_3D = get_contour_points(f_path, ed_slices)
        contour_pts = convert_3D_points_to_2D(contour_pts_3D)

        # visualise the contour points with the dicom image
        plot_contour_points(contour_pts_3D, ptr_content, ed_slices)
        plot_contour_points_3D(contour_pts_3D, ed_slices)
        break
    
    print(count)