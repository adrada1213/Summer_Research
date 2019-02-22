import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

def pad_image(image, new_size):
    h_diff = new_size-image.shape[0]
    w_diff = new_size-image.shape[1]

    padded_img = cv2.copyMakeBorder(image, h_diff//2, h_diff-(h_diff//2), w_diff//2, w_diff-(w_diff//2), cv2.BORDER_CONSTANT, value = [0,0,0])

    return padded_img

def resize_image(image, new_shape):
    aspectRatio = image.shape[1]/image.shape[0] #w/h
    if new_shape[0] >= new_shape[1]:
        new_h = new_shape[0]
        new_w = int(new_h * aspectRatio)      
    elif new_shape[1] > new_shape[0]:
        new_w = new_shape[1]
        new_h = int(new_w//aspectRatio)

    if new_h > 256:
        new_h = 256
        new_w = int(new_h*aspectRatio)
    elif new_w > 256:
        new_w = 256
        new_h = int(new_w//aspectRatio)

    new_img = cv2.resize(image, (new_w, new_h), interpolation = cv2.INTER_CUBIC) 

    return new_img

def overlay_images(image1, image2):
    combined = cv2.addWeighted(image1, 0.5, image2, 1, 0)

    fig,([ax1, ax2, ax3]) = plt.subplots(1,3)

    fig.set_tight_layout(True)
 
    # add the images to the axes
    ax1.imshow(image1, cmap = 'gray')
    ax2.imshow(image2, cmap = 'gray')
    ax3.imshow(combined, cmap = 'gray')
    
    # remove the tick marks and labels from the axes
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
    '''
    if (save_image):
        output_dir = os.path.join(os.getcwd(), "images")
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        plt.savefig("{}".format(os.path.join(output_dir, patient_name)))
    else:
        plt.show()
    '''

def rotate_image(image, angle):
    print(tuple(np.array(image.shape[1::-1])/2))
    img_centre = tuple(np.array(image.shape[1::-1])/2)
    rot_mat = cv2.getRotationMatrix2D(img_centre, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result

def print_image_with_landmarks(image, landmarks): 
    # add the images to the axes
    plt.imshow(image, cmap = 'gray')

    # add the landmark points to the axes
    plt.scatter(landmarks[0], landmarks[1], s=2, color="cyan")
    plt.scatter(landmarks[0][0:7], landmarks[1][0:7], s=2, color="yellow")
    
    # remove the tick marks and labels from the axes
    #plt.get_xaxis().set_visible(False)
    #plt.get_yaxis().set_visible(False)

    plt.show()

def print_images_with_landmarks(patient_name, cine_image, tagged_image, cine_ed_coords, tagged_ed_coords, save_image):
    # plot the data
    fig,([ax1, ax2]) = plt.subplots(1,2)

    fig.set_tight_layout(True)
 
    # add the images to the axes
    ax1.imshow(cine_image, cmap = 'gray')
    ax2.imshow(tagged_image, cmap = 'gray')

    # add the landmark points to the axes
    #print(type(ed_coords[0][0]))
    ax1.scatter(cine_ed_coords[0], cine_ed_coords[1], s=2, color="cyan")
    ax1.scatter(cine_ed_coords[0][0:7], cine_ed_coords[1][0:7], s=2, color="yellow")
    
    ax2.scatter(tagged_ed_coords[0], tagged_ed_coords[1], s=2, color="cyan")
    ax2.scatter(tagged_ed_coords[0][0:7], tagged_ed_coords[1][0:7], s=2, color="yellow")
    
    # remove the tick marks and labels from the axes
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    if (save_image):
        output_dir = os.path.join(os.getcwd(), "images")
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        plt.savefig("{}".format(os.path.join(output_dir, patient_name)))
    else:
        plt.show()
