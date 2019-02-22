"""
This python file contains the functions needed to work with .fwsx files (XML files)

Author: Amos Rada
Date:   16/01/2019
"""
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
    # get the cvi42 path of the patient
    zip_path = os.path.join(cvi42_path, cvi42_id + "_cvi42.zip")   

    # open the zip file, extract only the cvi42wsx file (XML file)
    cvi42_file = load_cvi42_content(zip_path, cvi42_id)

    # parse the xml file
    cvi_tree = etree.parse(cvi42_file)
    root = cvi_tree.getroot()   #get the root

    return root

def get_contour_points(cvi42_path, cvi42_id, ptr_content, slice_num):
    root = get_root(cvi42_path, cvi42_id)

    # initialise the lists needed
    cine_all = ptr_content[ptr_content["series"]==0]
    cine_slice = cine_all[cine_all["slice"]==slice_num]

    for frame in cine_slice:
        # get the uid of the cine dicom image
        uid = os.path.basename(frame["path"]).replace(".dcm", "") #only get the uid

        # get the epicardial x and y coordinates
        saepiContour = root.findall("./Hash:item/[@Hash:key='StudyMapStates']/List:item/Hash:item[@Hash:key='ImageStates']/Hash:item[@Hash:key='{}']/"
            "Hash:item[@Hash:key='Contours']/Hash:item[@Hash:key='saepicardialContour']/Hash:item[@Hash:key='Points']".format(uid), namespaces=namespaces)

        try:
            x_coords = saepiContour[0].xpath("./List:item/Point:x/text()", namespaces=namespaces)
            y_coords = saepiContour[0].xpath("./List:item/Point:y/text()", namespaces=namespaces)
        except IndexError:
            if frame["frame"] == len(cine_slice)-1:
                return -1
            continue

        subpixel_res = saepiContour[0].xpath("../Hash:item[@Hash:key='SubpixelResolution']/text()", namespaces = namespaces )
        break


    # divide the coordinates by the subpixel resolution to match the coordinates in the image
    x_coords = [float(x)/float(subpixel_res[0]) for x in x_coords]
    y_coords = [float(y)/float(subpixel_res[0]) for y in y_coords]

    contour_pts = [x_coords,y_coords]

    return contour_pts

def load_cvi42_content(zip_path, cvi42_id):
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