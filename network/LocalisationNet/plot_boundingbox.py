import re
import math
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import scipy.ndimage as ndimage

def print_image_with_bbox(img, bbox, pred_box, title, savemode=False):
    print('Printing image', title)
    print('bbox', bbox)
    print('predicted', pred_box)
    print('error', pred_box-bbox)

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,20))
    # ax1.set_title('Original ED frame - {}'.format(title))
    ax1.set_title(title)
    ax1.imshow(img, cmap='gray')

    # box1 = patches.Rectangle((bbox[0],bbox[1]), (bbox[2]-bbox[0]), (bbox[3]-bbox[1]), fill=False, linewidth=1, color='red', label='Ground truth')
    box1 = patches.Rectangle((pred_box[0],pred_box[1]), (pred_box[2]-pred_box[0]), (pred_box[3]-pred_box[1]), fill=False, linewidth=1, color='green', label='Prediction')
    box2 = patches.Rectangle((bbox[0],bbox[1]), (bbox[2]-bbox[0]), (bbox[3]-bbox[1]), fill=False, linewidth=1, color='red', label='Ground truth')
    # ax1.add_patch(box1)
    ax1.add_patch(box1)
    ax1.add_patch(box2)
    # ax1.legend(handles=[box1, box1, box2])
    ax1.legend(handles=[box1, box2])

    ax2.set_title('Ground truth bounding box (adjusted)')
    img2 = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]) ]
    img2 = cv2.resize(img2, dsize=(128,128), interpolation=cv2.INTER_CUBIC)
    ax2.imshow(img2, cmap='gray')

    ax3.set_title('Predicted bounding box')
    img3 = img[int(pred_box[1]):int(pred_box[3]), int(pred_box[0]):int(pred_box[2]) ]
    img3 = cv2.resize(img3, dsize=(128,128), interpolation=cv2.INTER_CUBIC)
    ax3.imshow(img3, cmap='gray')

    if (not savemode):
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()

        plt.show()
    else:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        fig.savefig('{}/{}.png'.format(output_dir, title), bbox_inches='tight', dpi=300)
    plt.close()


    # mng = plt.get_current_fig_manager()
    # mng.window.showMaximized()

    # plt.show()
    # plt.close()

def adjust_bounding_box(arr, adjustment_fraction = 0):
    if adjustment_fraction == 0:
        return arr

    diff_x  = arr[:,2]-arr[:,0]
    diff_y = arr[:,3]-arr[:,1]

    # stack em vertically
    stack = np.vstack((-diff_x, -diff_y, diff_x, diff_y))
    stack = np.transpose(stack)
    # should look like this now [[-dx, -dy, +dx +dy]... [..]]
    stack = adjustment_fraction * stack

    # add the adjustment
    return arr + stack

def rotate_coords_bbox_on_origin(coords):
    new_coords = np.asarray([-coords[1], coords[0], -coords[3], coords[2]])
    
    return new_coords

def rotate_coords_bbox(coords, angle):
    center_point = [128,128]
    
    point = [coords[0], coords[1]]
    x1 = rotate_point(center_point, point, angle)
    
    point = [coords[2], coords[3]]
    x2 = rotate_point(center_point, point, angle)

    min_x = min(x1[0], x2[0])
    min_y = min(x1[1], x2[1])

    max_x = max(x1[0], x2[0])
    max_y = max(x1[1], x2[1])

    # new_coords = np.asarray([x1[0], x1[1], x2[0], x2[1]])
    # print(new_coords)
    new_coords = np.asarray([min_x, min_y, max_x, max_y])
    print(new_coords)
    return new_coords

def rotate_point(centerPoint,point,angle):
    """Rotates a point around another centerPoint. Angle is in degrees.
    Rotation is counter-clockwise"""
    angle = math.radians(angle)
    temp_point = point[0]-centerPoint[0] , point[1]-centerPoint[1]
    temp_point = ( temp_point[0]*math.cos(angle)-temp_point[1]*math.sin(angle) , temp_point[0]*math.sin(angle)+temp_point[1]*math.cos(angle))
    temp_point = temp_point[0]+centerPoint[0] , temp_point[1]+centerPoint[1]
    return temp_point

# EM.0 -  51,52
# i = 0
# img = imgs[i]
# adj_box = aboxes[i]
# print_image_with_bbox(img, bboxes[i], adj_box, pred_boxes[i], i)

# adj_box = rotate_coords_bbox(adj_box, -90)
# img = ndimage.rotate(img, 90, reshape=False)
# print_image_with_bbox(img, bboxes[i], adj_box, pred_boxes[i], i)

# adj_box = rotate_coords_bbox(adj_box, -90)
# img = ndimage.rotate(img, 90, reshape=False)
# print_image_with_bbox(img, bboxes[i], adj_box, pred_boxes[i], i)

# adj_box = rotate_coords_bbox(adj_box, -90)
# img = ndimage.rotate(img, 90, reshape=False)
# print_image_with_bbox(img, bboxes[i], adj_box, pred_boxes[i], i)

def augment_rotate(ed_img, es_img, bbox, angle):
    ed_img = ndimage.rotate(ed_img, angle, reshape=False)
    es_img = ndimage.rotate(es_img, angle, reshape=False)
    bbox = rotate_coords_bbox(bbox, -angle)

    return ed_img, es_img, bbox

# eds = []
# ess = []
# bbs = []
# pats = []

# i = 0

# pats.append(patients[i])
# eds.append(ed_imgs[i])
# ess.append(es_imgs[i])
# bbs.append(bboxes[i])

# pats.append(patients[i])
# d,s,b = augment_rotate(ed_imgs[i], es_imgs[i], bboxes[i], 90)
# eds.append(d)
# ess.append(s)
# bbs.append(b)

# pats.append(patients[i])
# d,s,b = augment_rotate(ed_imgs[i], es_imgs[i], bboxes[i], 180)
# eds.append(d)
# ess.append(s)
# bbs.append(b)

# pats.append(patients[i])
# d,s,b = augment_rotate(ed_imgs[i], es_imgs[i], bboxes[i], 270)
# eds.append(d)
# ess.append(s)
# bbs.append(b)

# pats = np.array(pats, dtype=object)
# # save it
# string_dt = h5py.special_dtype(vlen=str)
# hf = h5py.File(os.path.join(base_path, 'test-rotate.h5'), 'w')
# hf.create_dataset("patients", data=pats, dtype=string_dt)
# hf.create_dataset("ed_imgs", data=eds)
# hf.create_dataset("es_imgs", data=ess)
# hf.create_dataset("bbox_corners", data=bbs)

# hf.close()


# ===== Main ===

def print_all_in_file():
    # fileprefix = 'test-rotate'
    fileprefix = 'CIM_DATA_AB.seq.noresize.0'
    filename = '{}.h5'.format(fileprefix) 
    bbox_file = '{}.bbox.h5'.format(fileprefix)
    

    # Load ground truth
    with h5py.File(os.path.join(data_path,filename), 'r') as hl:
        imgs = np.asarray(hl.get("ed_imgs"))[:,0,:,:] # get the ED only
        bboxes = np.asarray(hl.get("bbox_corners"))

    # adjusted ground truth
    aboxes = adjust_bounding_box(bboxes, percentage)

    # Load prediction
    with h5py.File(os.path.join(bbox_path,bbox_file), 'r') as hl:
        pred_boxes = np.asarray(hl.get("bbox_preds"))

    i = 65
    print_image_with_bbox(imgs[i], bboxes[i], aboxes[i], pred_boxes[i], i)
    for i, box in enumerate(bboxes):
        print_image_with_bbox(imgs[i], bboxes[i], aboxes[i], pred_boxes[i], i)

if __name__ == "__main__":
    percentage = 0.0 # no need to adjust again, it is now a part of prepare_data

    base_path = 'C:\\Users\\arad572\\Documents\\Summer Research\\Summer Research Code'
    data_path = '{}\\prepare_data\\h5_files'.format(base_path)
    bbox_path = '{}\\network\\LocalisationNet\\ds_local_all_output'.format(base_path)

    output_dir = '{}\\img'.format(bbox_path)
    savemode=False
    # this is to print all in a loop
    # mapping_file = '{}/dataset_test_mapping.h5'.format(data_path) 
    group_to_print = 'test'
    start_index = 0

    img_filename = 'UK_Biobank.h5'
    bbox_filename = 'UK_Biobank.result.h5'
    
    # load img
    img_file = os.path.join(data_path,img_filename)
    pred_file = os.path.join(bbox_path,bbox_filename)

    # read file here
    with h5py.File(img_file, 'r') as hf:
        group = hf["/{}".format(group_to_print)]
        patients = np.asarray(group.get("patients"))
        slices = np.asarray(group.get("slices"))

    # so we gonna loop through the list
    for i, patient in enumerate(patients):
        # Load ground truth
        with h5py.File(img_file, 'r') as hf:
            grp_cine = hf["/{}/cine".format(group_to_print)]
            img = np.asarray(grp_cine.get("images")[i,0,:,:]) # get the ED only
            centroid = np.asarray(grp_cine.get("centroids")[i])
            bbox = np.array([centroid[0]-centroid[2], centroid[1]-centroid[2], centroid[0]+centroid[2], centroid[1]+centroid[2]])
            #bbox = np.array([centroid[0]-centroid[2]/2, centroid[1]-centroid[2]/2, centroid[0]+centroid[2]/2, centroid[1]+centroid[2]/2])
            
        # adjusted ground truth
        #aboxes = adjust_bounding_box(bboxes, percentage)

        # Load prediction
        with h5py.File(pred_file, 'r') as hl:
            pred_centroid = np.asarray(hl.get("centroid_preds")[i])
            pred_bbox = np.array([pred_centroid[0]-pred_centroid[2], pred_centroid[1]-pred_centroid[2], pred_centroid[0]+pred_centroid[2], pred_centroid[1]+pred_centroid[2]])
            #pred_bbox = np.array([pred_centroid[0]-pred_centroid[2]/2, pred_centroid[1]-pred_centroid[2]/2, pred_centroid[0]+pred_centroid[2]/2, pred_centroid[1]+pred_centroid[2]/2])
        print_image_with_bbox(img, bbox, pred_bbox, '{}-{}'.format(patient,slices[i]), savemode=False)
