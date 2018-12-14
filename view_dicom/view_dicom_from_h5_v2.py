import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import cv2
from skimage import img_as_uint
from align_images import align_images

def plot_images(patient_name, my_cine_img, my_tagged_img, cine_img, mask, aligned_mask, ed_coords, save_image):
    mask = img_as_uint(mask)
    aligned_mask = img_as_uint(aligned_mask)
    # overlap the two images
    overlap = cv2.addWeighted(aligned_mask, 0.5, my_cine_img, 1, 0)
    overlap_2 = cv2.addWeighted(mask, 0.5, cine_img, 1, 0)
    
    # plot the data
    fig,([ax1, ax2, ax3]) = plt.subplots(1,3)

    fig.set_tight_layout(True)

    # add the images to the axes
    ax1.imshow(my_cine_img, cmap = 'gray')
    ax2.imshow(overlap, cmap = 'gray')
    ax3.imshow(overlap_2, cmap = 'gray')
    '''
    # add the landmark points to the axes
    #print(type(ed_coords[0][0]))
    ax1.scatter(ed_coords[0], ed_coords[1], s=2, color="cyan")
    ax1.scatter(ed_coords[0][0:7], ed_coords[1][0:7], s=2, color="yellow")
    
    ax2.scatter(ed_coords[0], ed_coords[1], s=2, color="cyan")
    ax2.scatter(ed_coords[0][0:7], ed_coords[1][0:7], s=2, color="yellow")

    ax3.scatter(ed_coords[0], ed_coords[1], s=2, color="cyan")
    ax3.scatter(ed_coords[0][0:7], ed_coords[1][0:7], s=2, color="yellow")
    '''
    # remove the tick marks and labels from the axes
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    if (save_image):
        output_dir = os.path.join(os.getcwd(), "images")
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        plt.savefig("{}".format(os.path.join(output_dir, patient_name)))
    else:
        plt.show()

def get_mask(h5File, df):
    datasets = ["train", "test", "validation"]
    found = False
    with h5py.File(h5File, "r") as hf:
        for d in datasets:
            print(d)
            dataset = hf.get("{}".format(d))
            filenames = np.array(dataset.get("filenames"))
            for i, f in np.ndenumerate(filenames):
                if df in f:
                    print(i)
                    mask = np.array(dataset.get("masks"))[i[0],:,:]
                    dicom_img = np.array(dataset.get("dicoms"))[i[0],:,:]
                    found = True
                    print(found)
                    break
            if found:
                break
    
    return mask, dicom_img


def get_images(h5File, df):
    groups = ["train", "test", "validation"]
    with h5py.File(h5File, "r") as hf:
        for g in groups:
            grp = hf.get(g)
            cine = hf["/{}/cine".format(g)]
            tagged = hf["/{}/tagged".format(g)]

            cine_dicom_paths = np.array(cine.get("cine_dicom_paths"))
            for index, cine_dp in np.ndenumerate(cine_dicom_paths):
                #print(index)
                if df in cine_dp:
                    cine_img = np.array(cine.get("cine_images"))[index[0],index[1],:,:]
                    tagged_img = np.array(tagged.get("tagged_images"))[index[0],index[1],:,:]
                    lm_coords = np.array(grp.get("landmark_coords"))[index[0],0,:,:]
                    #print(cine_img.shape)
                    print("Images loaded...")
                    return cine_img, tagged_img, lm_coords
                

    return None, None, None

if __name__ == '__main__':
    hdf5_filepath = "C:\\Users\\arad572\\Documents\\Summer Research\\code\\prepare_data\\h5_files\\sa-oc-lvsc_new-filenames.hdf5"

    my_hdf5_filepath = "C:\\Users\\arad572\\Documents\\Summer Research\\code\\prepare_data\\h5_files\\UK_Biobank_20cases.h5"

    patient_name = "2B UK 9P 26 Bio"
    dicom_filename = "1.3.12.2.1107.5.2.18.41754.2015060817574722496124508.dcm"

    my_cine_img, my_tagged_img, lm_coords = get_images(my_hdf5_filepath, dicom_filename)

    mask, img_hdf5 = get_mask(hdf5_filepath, dicom_filename)

    
    aligned_cine_img, aligned_mask = align_images(img_hdf5, my_cine_img, mask)
    print(aligned_cine_img.dtype, img_hdf5.dtype)
    plot_images(patient_name, aligned_cine_img, my_tagged_img, img_hdf5, mask, aligned_mask, lm_coords, save_image=False)
    #evaluate_on_hdf5('F://sa-oc-lvsc_new-filenames.hdf5')
    print('Finished')