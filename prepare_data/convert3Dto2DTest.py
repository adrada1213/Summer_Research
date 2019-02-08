from dicom_functions import get_3Dcorners, convert3D_to_2D
from test import get_slices_and_indices, get_contour_points, plot_contour_points
from pointer_functions import load_ptr_content
import pydicom
import numpy as np

ptr = "C:\\Users\\arad572\\Documents\\Summer Research\\code\\prepare_data\\img_ptrs\\matches\\2B_38_Z7_7S_Bio_match.img_imageptr"
ptr_content = load_ptr_content(ptr)
f_path = "E:/LVModellerFormatV2/4640537"

ed_slices, ed_indices, es_slices, es_indices = get_slices_and_indices(f_path, ptr_content)
coords_3D = get_contour_points(f_path, ed_slices)
coords_3D_es = get_contour_points(f_path, es_slices)
print(ed_slices)

for i in range(len(ed_slices)):
    
    if ed_slices[i] != -1:
        ed_path = ptr_content[np.logical_and(ptr_content["series"]==0, ptr_content["slice"]==i)]["path"][ed_indices[i]]
        img_file = ed_path.replace("IMAGEPATH", "E:\\Original Images\\2015")
        tlc, trc, blc, img_size = get_3Dcorners(img_file)
        img_coords = convert3D_to_2D(coords_3D[ed_slices[i]], img_size, tlc, trc, blc)
        img = pydicom.dcmread(img_file).pixel_array

        plot_contour_points(img_coords, img, 0)
    
    if es_slices[i] != -1:
        es_path = ptr_content[np.logical_and(ptr_content["series"]==0, ptr_content["slice"]==i)]["path"][es_indices[i]]
        img_file = es_path.replace("IMAGEPATH", "E:\\Original Images\\2015")
        tlc, trc, blc, img_size = get_3Dcorners(img_file)
        img_coords = convert3D_to_2D(coords_3D_es[es_slices[i]], img_size, tlc, trc, blc)
        img = pydicom.dcmread(img_file).pixel_array

        plot_contour_points(img_coords, img, 0)