import numpy as np
from dicom_functions import convert_2D_points_to_3D, convert_3D_points_to_2D
from image_functions import print_image_with_landmarks, pad_image
from coordinates_functions import plot_2D_coords, plot_3D_coords
from image_pointer_functions import load_ptr_content
import pydicom




#trs_path

'''
    Get the list of coordinates, regions, and other properties from the _strain.dat file
    Return a tuple of (x coord, y coords, LV (AHA) segment, elements (4 segment), division, radial index )
'''
def get_model_points_and_label_from_file(filepath):
    data = np.genfromtxt(filepath, delimiter=' ', names=True)
    #points = (data['imageX'], data['imageY'], "{0}{1}{2}{3}".format(data['Region'],data['Element'],data['SubDivXi1'],data['SubDivXi2']))
    points_2D = (data['imageX'], data['imageY'])
    points_3D = (data['patientX'], data['patientY'], data['patientZ'])
    return points_2D, points_3D

ptr = "C:\\Users\\arad572\\Documents\\Summer Research\\Summer Research Code\\prepare_data\\img_ptrs\\matches\\2B_38_Z7_7S_Bio_match.img_imageptr"
cim_model_path = "C:\\Users\\arad572\\Downloads\\all CIM\\CIM_DATA_AB\\2B_38_Z7_7S_Bio-ab\\model_2B_38_Z7_7S_Bio-ab\\series_1_slice_1\\2B_38_Z7_7S_Bio-ab_1_samplePt_strain.dat"

points_2D, points_3D = get_model_points_and_label_from_file(cim_model_path)
#plot_3D_coords(points_3D)
ptr_content = load_ptr_content(ptr)

tagged = ptr_content[ptr_content["series"]==1]
image_path = tagged["path"][0].replace("IMAGEPATH", "E:\\Original Images\\2015")
cine_path = ptr_content["path"][0].replace("IMAGEPATH", "E:\\Original Images\\2015")

ds = pydicom.dcmread(image_path)
ds_cine = pydicom.dcmread(cine_path)

image = ds.pixel_array
cine_image = ds_cine.pixel_array
#points = [points[0], image.shape[0]-points[1]]
#image = pad_image(image, 256)
points_2D = [points_2D[0], image.shape[0]-points_2D[1]]

#print_image_with_landmarks(image, points_2D)

points_3D = np.stack((points_3D[0], points_3D[1], points_3D[2]), axis=-1)

converted_pts = convert_3D_points_to_2D(points_3D, image_path)
converted_pts_cine = convert_3D_points_to_2D(points_3D, cine_path)
new_points_2D = [[],[]]
new_points_2D_cine = [[], []]

for i in range(len(converted_pts)):
    new_points_2D[0].append(converted_pts[i][0])
    new_points_2D[1].append(converted_pts[i][1])
    new_points_2D_cine[0].append(converted_pts_cine[i][0])
    new_points_2D_cine[1].append(converted_pts_cine[i][1])

plot_2D_coords(new_points_2D_cine)
print_image_with_landmarks(image, new_points_2D)
print_image_with_landmarks(cine_image, new_points_2D_cine)

