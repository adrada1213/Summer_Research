import h5py
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt


# CROPPING AND RESIZING IMAGES TO BE LOADED
def crop_image_bbox(img, bbox_corners):
    left_x = math.floor(bbox_corners[0])
    low_y  = math.floor(bbox_corners[1])

    right_x = math.ceil(bbox_corners[2])
    high_y  = math.ceil(bbox_corners[3])

    # special case where low_x or left_x is negative, we need to add some padding
    pad_x = 0
    pad_y = 0
    if (left_x < 0):
        pad_x = 0 - left_x
    if (low_y < 0):
        pad_y = 0 - low_y
    
    if (pad_x == 0 and pad_y == 0):
        return img[low_y:high_y, left_x:right_x ]
    else:
        print('Cropping image with extra padding due to negative start index')
        new_img = np.pad(img, ((pad_y,pad_y),(pad_x,pad_x)), 'constant')
        return new_img[pad_y+low_y:pad_y+high_y, pad_x+left_x:pad_x+right_x ]

def resize_image(img, new_size):
    new_img = cv2.resize(img, dsize=(new_size,new_size), interpolation=cv2.INTER_CUBIC)
    
    ratio_x = new_size / img.shape[0]
    # coords = np.array(coords)
    # # assumption x and y has the same ratio
    # new_coords = coords * ratio_x
    return new_img, ratio_x

def crop_and_resize_all_frames(img_sequences, corners, new_img_size):
    # TODO: this might not be optimal, fix the loop later pls
    cropped_frames = np.zeros(shape=(img_sequences.shape[0], new_img_size, new_img_size))

    # for every image sequences
    for i in range(len(img_sequences)):
        #bbox_corners = corners[i]
        bbox_corners = corners
    
        frame = img_sequences[i,:,:]

        new_img = crop_image_bbox(frame, bbox_corners)
        # Resample the image and keep the ratio
        new_img, resize_ratio = resize_image(new_img, new_img_size)

        # fill it in
        cropped_frames[i] = new_img

    return cropped_frames, resize_ratio

def translate_coords(landmark_coords, corners, resize_ratios):
    new_landmark_coords = []
    corners_x = np.array([corners[0]]*168)
    corners_y = np.array([corners[1]]*168)

    for coords in landmark_coords:
        # deduct the x and y corners by the distance from origin to the top left corner of the bbox
        coords_x = (coords[0]-corners_x)*resize_ratios
        coords_y = (coords[1]-corners_y)*resize_ratios

        new_landmark_coords.append([coords_x, coords_y])

    return np.array(new_landmark_coords)

def plot_image(images, landmark_coords):
    for i in range(len(images)):
        image = images[i]
        coords = landmark_coords[i]

        plt.imshow(image, cmap="gray")
        plt.scatter(coords[0], coords[1], s=2, color="cyan")
        plt.scatter(coords[0][0:7], coords[1][0:7], s=2, color="yellow")

        plt.show()

'''
with h5py.File(h5_file, 'r') as hf:
    grp = hf["/{}".format(group)]
    grp_cine = hf["/{}/cine".format(group)]
    es_indices = np.array(grp_cine.get("es_indices"))

    images = np.asarray(grp_cine.get('images')[0,[0,19],:,:])
    landmark_coords = np.asarray(grp_cine.get('landmark_coords')[0])

print(images.shape)
print(landmark_coords.shape)
length = len(es_indices)    
indices = np.arange(length)
neg_es = np.argwhere(es_indices<0)
print(neg_es)

indices = np.delete(indices, neg_es)
new_length = len(indices)
es_indices = np.delete(es_indices, neg_es)

indices = np.concatenate(([indices], [es_indices]), axis = 0)

filepaths = [h5_file]*new_length
groups = [group]*new_length

print(indices[1][3])
'''

if __name__ == "__main__":
    fpath = "C:\\Users\\arad572\\Documents\\Summer Research\\Summer Research Code\\prepare_data\\h5_files\\UK_Biobank.h5"
    group = "train"
    idx = 204
    es_idx = 17
    with h5py.File(fpath, 'r') as hl:
        grp_cine = hl["/{}/cine".format(group)]
        centroid = np.asarray(grp_cine.get('centroids')[idx])
        images = np.asarray(grp_cine.get('images')[idx,[0,es_idx],:,:])
        landmark_coords = np.asarray(grp_cine.get('landmark_coords')[idx])

    # calculate bbox from centroid
    corners = [centroid[0]-centroid[2], centroid[1]-centroid[2], centroid[0]+centroid[2], centroid[1]+centroid[2]]
    # --- Crop and resize --- 
    cropped_frames, resize_ratios = crop_and_resize_all_frames(images, corners, 128)
    # translate the landmark coords to fit the new image
    landmark_coords = translate_coords(landmark_coords, corners, resize_ratios)

    plot_image(cropped_frames, landmark_coords)

