from random import randint
import scipy.ndimage as ndimage
import numpy as np
import math
import h5py
import re

'''
    Obsolete
'''
def _load_hd5_file_index(fpath, idx):
    hd5path = fpath
    with h5py.File(hd5path, 'r') as hl:
        ed_img = np.asarray(hl.get('ed_imgs')[idx])
        bbox = np.asarray(hl.get('bbox_corners')[idx])
    # !!! IMPORTANT, we need to cast this to the proper data type like below
    # Default type is double/float64
    return ed_img.astype('float32'), bbox.astype('float32')

# ----- Used by localisation network -----
'''
    Load an image from a pair of filename-index
    Only the ED frame is taken from the image sequence

'''
def _load_hd5_img_and_centroid_from_sequence(fpath, group, idx):
    fpath = fpath.decode('utf-8')
    group = group.decode('utf-8')
    #fpath = re.search("\'.*\'", fpath).string
    #group = re.search("\'.*\'", group).string
    with h5py.File(fpath, 'r') as hf:
        grp_cine = hf["//{}//cine".format(group)]

        # [n, time=0, 256, 256]
        image = np.asarray(grp_cine.get('images')[idx,0,:,:]) # we only need the first frame
        # [n, 4]
        centroid = np.asarray(grp_cine.get('centroids')[idx])

    # !!! IMPORTANT, we need to cast this to the proper data type like below
    # Default type is double/float64
    return image.astype('float32'), centroid.astype('float32')
    
'''
    Rotate the image and bounding box
    Currently it only does 90 degree multiplication
'''
def _rotate_img_and_centroid(img, centroid):
    # TODO: adjust this, currently if the number falls to 0, or more than 3, the image is not rotated
    rnd = randint(0,7) 
    # print('random',rnd)
    if (rnd == 0 or rnd > 3):
        # we give higher chance the image is not being rotated
        return img, centroid
    else:
        # we do only rotation for 90, 180, and 270 for now
        angle = rnd * 90
        new_img = ndimage.rotate(img, angle, reshape=False)
        new_bbox = rotate_coords_centroid(img.shape, centroid, -angle)

        # return new_img.astype('float32'), new_bbox
        return new_img, new_bbox.astype('float32')

def rotate_coords_centroid(img_shape, coords, angle):
    center_point = np.asarray(img_shape)/2

    point = [coords[0], coords[1]]
    x = rotate_single_point(point, angle, center_point)

    new_coords = np.asarray([x[0], x[1], coords[2]])

    return new_coords

def rotate_single_point(point,angle, centerPoint):
    """Rotates a point around another centerPoint. Angle is in degrees.
    Rotation is counter-clockwise"""
    angle = math.radians(angle)
    temp_point = point[0]-centerPoint[0] , point[1]-centerPoint[1]
    temp_point = ( temp_point[0]*math.cos(angle)-temp_point[1]*math.sin(angle) , temp_point[0]*math.sin(angle)+temp_point[1]*math.cos(angle))
    temp_point = temp_point[0]+centerPoint[0] , temp_point[1]+centerPoint[1]
    return temp_point
# -----

# ----- Used by RNNCNN Landmark tracking network -----
'''
    Load an image sequence and coords sequence for a certain filename-index pair
    output shape: [t, 128, 128], [t, 2, 168]
    t is number of frames (t=20)
'''
def _load_hd5_img_and_coords_sequence(fpath, group, idx, es_idx):
    fpath = fpath.decode('utf-8')
    group = group.decode('utf-8')
    with h5py.File(fpath, 'r') as hl:
        grp_cine = hl["/{}/cine".format(group)]
        images = np.asarray(grp_cine.get('images')[idx,[0,es_idx],:,:])
        landmark_coords = np.asarray(grp_cine.get('landmark_coords')[idx])
    # !!! IMPORTANT, we need to cast this to the proper data type like below
    return images.astype('float32'), landmark_coords.astype('float32')
# -----------

def _load_features_and_coords_sequence(hd5path, feature_path, idx):
    with h5py.File(hd5path, 'r') as hl:
        ed_coords = np.asarray(hl.get('ed_coords')[idx])
        
    with h5py.File(feature_path, 'r') as hl:
        features = np.asarray(hl.get('dense2')[idx])
        
    # !!! IMPORTANT, we need to cast this to the proper data type like below
    return features.astype('float32'), ed_coords.astype('float32')

def _load_preds_and_coords_sequence(hd5path, feature_path, idx):
    with h5py.File(hd5path, 'r') as hl:
        ed_coords = np.asarray(hl.get('ed_coords')[idx])
        
    with h5py.File(feature_path, 'r') as hl:
        features = np.asarray(hl.get('ed_coords_preds')[idx])
        features = np.reshape(features, [features.shape[0], features.shape[1] * features.shape[2]])
        
    # !!! IMPORTANT, we need to cast this to the proper data type like below
    return features.astype('float32'), ed_coords.astype('float32')


'''
    Rotate image and coordinates sequence
    img_seq: (time_step, width, height)
    coords_seq: (time_step, x_arr, y_arr)
'''
def _rotate_img_and_coords_sequence(img_seq, coords_seq):
    # TODO: we need to do the probability properly here..
    rnd = randint(0,7) 
    # print('random',rnd)
    if (rnd == 0 or rnd > 3):
        # we give higher chance the image is not being rotated
        return img_seq, coords_seq
    else:
        # we do only rotation for 90, 180, and 270 for now
        angle = rnd * 90
        # TODO: Handle blank image here
        new_img_seq = np.zeros(img_seq.shape)
        for i in range(0,len(img_seq)):
            new_img_seq[i] = ndimage.rotate(img_seq[i], angle, reshape=False)

        # TODO: don't hardcode center point here
        center_point = np.asarray((img_seq.shape[1],img_seq.shape[2]),'float32')/2
        new_coords = rotate_sequence_points(coords_seq, -angle, centerPoint=center_point)
        
        return new_img_seq.astype('float32'), new_coords.astype('float32')

def rotate_sequence_points(points, angle, centerPoint=(0,0)):
    n = points.shape[0]
    # print(points.shape)
    seq = points
    seq = np.transpose(seq,(1,0,2))
    seq = np.reshape(seq, [2, -1])
    # print(seq)
    seq = rotate_points(seq, angle, centerPoint)
    seq = np.reshape(seq, [2,n,-1])
    seq = np.transpose(seq, (1,0,2))
    return seq

def rotate_points(points,angle,centerPoint=(0,0)):
    """Rotates a point around another centerPoint. Angle is in degrees.
    Rotation is counter-clockwise"""
    angle = math.radians(angle)
    
    temp_coords = np.zeros(points.shape)
    temp_coords[0] = points[0]-centerPoint[0]
    temp_coords[1] = points[1]-centerPoint[1]

    new_coords = np.zeros(points.shape)
    new_coords[0] = temp_coords[0]*math.cos(angle)-temp_coords[1]*math.sin(angle)
    new_coords[1] = temp_coords[0]*math.sin(angle)+temp_coords[1]*math.cos(angle)
    
    new_coords[0] = new_coords[0]+centerPoint[0]
    new_coords[1] = new_coords[1]+centerPoint[1]
    return new_coords