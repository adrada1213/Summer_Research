'''
Extracting data from cvi42 files
'''
from test import read_mapping_file
from lxml import etree
from cvi42_functions import get_contour_points, get_cvi42_id, get_root
import numpy as np
import os
import csv
from test import plot_contour_points, get_slices_and_indices
from pointer_functions import load_ptr_content, get_slices
from prepare_data_functions import calculate_centroid




if __name__ == "__main__":
    # specify CVI42 filepath
    cvi42_path = "E:\\ContourFiles\\CVI42"

    # specify where the mapping file is
    mapping_file = "E:\\confidential_bridging_file_r4.csv"

    # specify where the image pointers are
    ptrs_path = "C:\\Users\\arad572\\Documents\\Summer Research\\code\\prepare_data\\img_ptrs\\matches"

    # specify the location of the modellers
    LVModel_path = "E:\\LVModellerFormatV2"

    # get the folder ids and the patient ids from the mapping file
    cvi42_ids, p_ids = read_mapping_file(mapping_file)

    # put the pointers in a list
    ptrs = os.listdir(ptrs_path)[0:10]

    duplicate_count = 0
    file_not_found_count = 0
    for ptr in ptrs:
        ed_centroids = []
        es_centroids = []
        pat_name = ptr.replace("_match.img_imageptr", "")   #get the patient name
        pat_id = pat_name.replace("_", "")[:8]  #get the patient id to find the matching folder in the LVModel path using the csv file
        ptr_path = os.path.join(ptrs_path, ptr) #convert into path
        
        ptr_content = load_ptr_content(ptr_path)

        # if patient doesn't match any folder, go to the next patient
        try:
            cvi42_id = get_cvi42_id(cvi42_ids, p_ids, pat_id)
        except ValueError:
            continue

        # get the slices and indices of the ed and es frames
        # if folder doesn't exist, go to the next patient
        try:
            f_path = os.path.join(LVModel_path, cvi42_id)
            ed_slices, ed_indices, es_slices, es_indices = get_slices_and_indices(f_path, ptr_content)
        except FileNotFoundError:
            file_not_found_count += 1
            continue

        ptr_slices = get_slices(ptr_content)

        # Found out that some patients have two of the same slice names in the cine series
        if len(ed_slices) > 3 or len(es_slices) > 3:
            print("Patient {} has duplicate slice names")
            duplicate_count += 1
            continue

        # get contour points for all slices for the current patient
        # Points will be translated to match a 256x256 cine image (resized to the matching slice in the tagged series and padded to make it 256x256)
        root = get_root(cvi42_path, cvi42_id)
        ed_contour_pts = get_contour_points(root, ptr_content, ptr_slices, [0]*len(ptr_slices))
        es_contour_pts = get_contour_points(root, ptr_content, ptr_slices, es_indices)

        # check which ed slice don't have epi/endo contours (no contours means we can't calculate centroid. No centroid means we can't
        # translate landmark points properly so, we're going to ignore that slice
        ptr_slices = np.append(ptr_slices, 0)
        ed_contour_pts.append([[-1],[-1]])
        deduct = 0
        for i in range(len(ptr_slices)):
            if ed_contour_pts[i-deduct] == [[-1],[-1]]:
                del ed_contour_pts[i-deduct]
                ptr_slices = np.delete(ptr_slices, i-deduct)
                deduct += 1

        print(ptr_slices)
        '''
        for i in range(len(ptr_slices)):
            ed_centroid = calculate_centroid(ed_contour_pts[i])
            es_centroid = calculate_centroid(es_contour_pts[i])
            ed_centroids.append(ed_centroid)
            es_centroids.append(es_centroid)
            print(ed_centroid)
            print(es_centroid)
        # TO DO:
        # Translate contour points to a 256x256 image
        '''
        # check which ed slice don't have epi/endo contours (no contours means we can't calculate centroid. No centroid means we can't
        # translate landmark points properly so, we're going to ignore that slice
        for i in ptr_slices:
            cine_dicom_paths, cine_images, tagged_dicom_paths, tagged_images, x_ratio, y_ratio, w_diff, h_diff = get_dicom_info(filepaths, ptr_content, i)


        
        plot_contour_points(ed_contour_pts, ptr_content, [0]*len(ptr_slices))
        