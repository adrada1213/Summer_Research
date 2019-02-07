"""
This python file contains the functions needed to work with .cvi42wsx files (XML files)

Author: Amos Rada
Date:   16/01/2019
"""
# import libraries
import os
import numpy as np
from io import BytesIO
import zipfile
import xml.etree.ElementTree as ET
from lxml import etree
from prepare_data_functions import calculate_centroid, log_error_and_print
import csv

# These are the namespaces used in the cvi42 file
namespaces = {
    'Hash': "http://www.circlecvi.com/cvi42/Workspace/Hash/",
    'Workspace': "http://www.circlecvi.com/cvi42/Workspace/",
    'List': "http://www.circlecvi.com/cvi42/Workspace/List/",
    'Point': "http://www.circlecvi.com/cvi42/Workspace/Point/" }

def get_uids(slice_info):
    with open(slice_info) as f:
        uids = []
        for line in f:
            if "SOPInstanceUID" in line:
                # get the uid and corresponding slice id
                uid =  line.split(":")[1].strip()
                line = f.readline()

                # append to list
                uids.append(uid)
    
    return uids

def get_indices(LVModel_path, cvi42_id, ptr_content, slices):
    # initialise the variables we need to return
    es_indices = []

    # get the folder path
    f_path = os.path.join(LVModel_path, cvi42_id)

    # get the filepaths for the file containing  the slice info
    es_slice_info = os.path.join(f_path, "ES", "SliceInfoFile.txt")

    # read the ed and es slice info file and get the uids and slice ids
    try:
        es_uids = get_uids(es_slice_info)
    except FileNotFoundError:
        log_error_and_print("SliceInfoFile not found. Pat ID: {}".format(cvi42_id))
        return np.array([-1]*len(slices))

    # get the cine frames
    cine_frames = ptr_content[ptr_content["series"] == 0]

    # loop through each slice in the image pointer
    for sl in slices:
        # loop through the frames in that slice
        frames = cine_frames[cine_frames["slice"]==sl]
        for j, frame in enumerate(frames):
            uid = os.path.basename(frame["path"].replace(".dcm", ""))
            if uid in es_uids and j != 0:
                #print(uid, es_uids)
                # add to list
                es_indices.append(frame["index"])
                break
    
            if j == len(frames)-1:
                es_indices.append(-1)
    
    return np.array(es_indices)

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

def get_root(cvi42_path, cvi42_id):
    # get the cvi42 path of the patient
    zip_path = os.path.join(cvi42_path, cvi42_id + "_cvi42.zip")   

    # open the zip file, extract only the cvi42wsx file (XML file)
    cvi42_file = load_cvi42_content(zip_path, cvi42_id)

    # parse the xml file
    cvi_tree = etree.parse(cvi42_file)
    root = cvi_tree.getroot()   #get the root

    return root

def get_contour_points(root, ptr_content, slices, indices):
    # initialise the lists needed
    contour_pts = []

    # loop through the slices
    for i, index in enumerate(indices):
        if index != -1:
            # get the filepath of the cine dicom image
            uid = os.path.basename(ptr_content[ptr_content["slice"] == slices[i]]["path"][index]).replace(".dcm", "") #only get the uid

            # get the epicardial x and y coordinates
            saepiContour = root.findall("./Hash:item/[@Hash:key='StudyMapStates']/List:item/Hash:item[@Hash:key='ImageStates']/Hash:item[@Hash:key='{}']/"
                "Hash:item[@Hash:key='Contours']/Hash:item[@Hash:key='saepicardialContour']/Hash:item[@Hash:key='Points']".format(uid), namespaces=namespaces)

            try:
                x_coords = saepiContour[0].xpath("./List:item/Point:x/text()", namespaces=namespaces)
                y_coords = saepiContour[0].xpath("./List:item/Point:y/text()", namespaces=namespaces)
                subpixel_res = saepiContour[0].xpath("../Hash:item[@Hash:key='SubpixelResolution']/text()", namespaces = namespaces )
            except IndexError:
                try:
                    print("EpiCardial Contour not found, Taking the EndoCardial Contour instead")
                    # get the endocardial x and y coordinates instead
                    saendoContour = root.findall("./Hash:item/[@Hash:key='StudyMapStates']/List:item/Hash:item[@Hash:key='ImageStates']/Hash:item[@Hash:key='{}']/"
                        "Hash:item[@Hash:key='Contours']/Hash:item[@Hash:key='saendocardialContour']/Hash:item[@Hash:key='Points']".format(uid), namespaces=namespaces)

                    x_coords = saendoContour[0].xpath("./List:item/Point:x/text()", namespaces=namespaces)
                    y_coords = saendoContour[0].xpath("./List:item/Point:y/text()", namespaces=namespaces)
                    subpixel_res = saendoContour[0].xpath("../Hash:item[@Hash:key='SubpixelResolution']/text()", namespaces = namespaces )

                except IndexError:
                    print("No Epi and Endo Contours")
                    contour_pts.append([[-1],[-1]])
                    continue

            # divide the coordinates by the subpixel resolution to match the coordinates in the image
            x_coords = [float(x)/float(subpixel_res[0]) for x in x_coords]
            y_coords = [float(y)/float(subpixel_res[0]) for y in y_coords]

            contour_pts.append([x_coords,y_coords])
        else:
            contour_pts.append([[-1],[-1]])

    return contour_pts

def get_cvi42_id(cvi42_ids, p_ids, pat_id):
    # get the matching folder of current patient
    
    p_index = p_ids.index(pat_id)
    cvi42_id = cvi42_ids[p_index]

    return cvi42_id

def load_cvi42_content(zip_path, cvi42_id):
    with open(zip_path, "rb") as zip_file:
        cvi42_file = BytesIO(zipfile.ZipFile(zip_file).read(cvi42_id+"_cvi42.cvi42wsx"))

    return cvi42_file
