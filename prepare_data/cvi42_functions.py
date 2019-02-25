"""
This python file contains the functions needed to work with .cvi42wsx files (XML files)
(ref code: E:\\cardiac\\dataset\\biobank\\cvi42.py by tfin440)
Author: Amos Rada
Date:   25/02/2019
"""
# import needed libraries
from io import BytesIO
import zipfile
import xml.etree.ElementTree as ET
from lxml import etree
import os
from image_pointer_functions import load_ptr_content

# These are the namespaces used in the cvi42 file
namespaces = {
    'Hash': "http://www.circlecvi.com/cvi42/Workspace/Hash/",
    'Workspace': "http://www.circlecvi.com/cvi42/Workspace/",
    'List': "http://www.circlecvi.com/cvi42/Workspace/List/",
    'Point': "http://www.circlecvi.com/cvi42/Workspace/Point/" }

def get_root(cvi42_path, cvi42_id):
    '''
    An xml file has a folder like structure. This function gets the "root directory" in the xml file.
    Inputs:
        cvi42_path (string) = path to the cvi42 zip files
        cvi42_id (string) = the folder id of the patient (based on the confidential mapping file)
    Output:
        root = root directory in the xml file
    '''
    # get the cvi42 path of the patient
    zip_path = os.path.join(cvi42_path, cvi42_id + "_cvi42.zip")   

    # open the zip file, extract only the cvi42wsx file (XML file)
    cvi42_file = load_cvi42_content(zip_path, cvi42_id)

    # parse the xml file
    cvi_tree = etree.parse(cvi42_file)
    root = cvi_tree.getroot()   #get the root

    return root

def get_contour_points(cvi42_path, cvi42_id, ptr_content, slice_num):
    '''
    This function obtains a set of epicardial contour points.
    Inputs:
        cvi42_path (string) = path to the cvi42 zip files
        cvi42_id (string) = folder id of the patient (based on the confidential mapping file)
        ptr_content (array) = content of the patient's image pointer
        slice_num (int) = slice of interest
    Output:
        contour_pts (2xN list) = 2D epi cardial contour points
    
    Note: This is used for translating the landmark points (based on centre of the contour points).
    Here I'm assuming that the centre of the shape of the first frame is not really far from the centre
    of the succeeding frames (i.e. ideally we want to translate the ED landmarks based on the ED frame but,
    sometimes the ED contours don't exist. So, we loop through the frames until we get contour points. We 
    then use the centre of that contour to translate the ED landmarks.)
    '''
    # get the root of the xml file
    root = get_root(cvi42_path, cvi42_id)

    cine_all = ptr_content[ptr_content["series"]==0]    #grab all the cine slices
    cine_slice = cine_all[cine_all["slice"]==slice_num] #grab only the cine slice of interest

    # loop through the frames until we get a set of epicardial contour point (ideally we want the contour points for the ed frame)
    for frame in cine_slice:
        # get the uid of the cine dicom image
        uid = os.path.basename(frame["path"]).replace(".dcm", "")

        # get the epicardial x and y coordinates
        # here we're navigating through the xml file to find the epicardial contour of the current frame (based on their uid)
        saepiContour = root.findall("./Hash:item/[@Hash:key='StudyMapStates']/List:item/Hash:item[@Hash:key='ImageStates']/Hash:item[@Hash:key='{}']/"
            "Hash:item[@Hash:key='Contours']/Hash:item[@Hash:key='saepicardialContour']/Hash:item[@Hash:key='Points']".format(uid), namespaces=namespaces)
        try:
            x_coords = saepiContour[0].xpath("./List:item/Point:x/text()", namespaces=namespaces)
            y_coords = saepiContour[0].xpath("./List:item/Point:y/text()", namespaces=namespaces)
        except IndexError: # if contour points don't exist for the current frame
            if frame["frame"] == len(cine_slice)-1: #if contour points don't exist for all the frames in the slice
                return -1
            continue

        # get the subpixel resolution
        subpixel_res = saepiContour[0].xpath("../Hash:item[@Hash:key='SubpixelResolution']/text()", namespaces = namespaces )
        break   #once we have a set of epicardial contour points, we stop the loop

    # divide the coordinates by the subpixel resolution to match the coordinates in the dicom image
    x_coords = [float(x)/float(subpixel_res[0]) for x in x_coords]
    y_coords = [float(y)/float(subpixel_res[0]) for y in y_coords]

    # compile in a list
    contour_pts = [x_coords,y_coords]

    return contour_pts

def load_cvi42_content(zip_path, cvi42_id):
    '''
    This function load the content of the .cvi42wsx file inside the patient zip file
    Inputs:
        zip_path (string) = path to the patients zip file
        cvi42_id (string) = folder id of the patient (based on the confidential mapping file)
    Output:
        cvi42_file = content of the xml file
    '''
    # read the content of the xml file
    with open(zip_path, "rb") as zip_file:
        cvi42_file = BytesIO(zipfile.ZipFile(zip_file).read(cvi42_id+"_cvi42.cvi42wsx"))

    return cvi42_file

# ========== testing the functions ==========
if __name__ == "__main__":
    cvi42_path = "E:\\ContourFiles\\CVI42"
    folder_id = "4081363"
    ptr_content = load_ptr_content("C:\\Users\\arad572\\Documents\\Summer Research\\Summer Research Code\\prepare_data\\img_ptrs\\matches\\2P_5U_64_6Q_Bio_match.img_imageptr")
    slice_num = 1

    epi_contours = get_contour_points(cvi42_path, folder_id, ptr_content, slice_num)
    print(epi_contours)