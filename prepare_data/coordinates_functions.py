"""
This script contains functions that will handle 2D and 3D coordinates
Author: Amos Rada
Date    22/02/2019
"""
# import needed libraries
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from math import sqrt

def plot_3D_coords(coords):
    '''
    Plot 3D coordinates
    Input:
        coords (3xN array) = 3D coordinates
    ''' 
    ax = plt.axes(projection="3d")
    ax.scatter3D(coords[0], coords[1], coords[2])
    plt.show()

def plot_2D_coords(coords):
    '''
    Plot 2D coordinates
    Input:
        coords (2xN array) = 2D coordinates
    ''' 
    plt.scatter(coords[0], coords[1])
    plt.show()

def calculate_centroid(coords):
    '''
    Calculate the centre of a circular shaped thing
    Input:
        coords (2xN array) = 2D coordinates
    Output:
        centroid (1x2 list) = x and y coordinates of the centre 
    '''
    center_x = (max(coords[0]) + min(coords[0])) /2  
    center_y = (max(coords[1]) + min(coords[1])) /2 

    centroid = [center_x, center_y]

    return centroid

def calculate_edge_length(centroid, coords):
    '''
    Calculate HALF the edge length
    Inputs:
        centroid (1x2 list) = x and y coordinates of the centroid
        coords (2xN array/list) = 2D coordinates of the circular shaped thing
    Output:
        edge_length (float) = half the edge length adjusted by 30%
    '''
    # calculates half the edge length!!!
    edge_length_x = max(coords[0])-centroid[0]
    edge_length_y = max(coords[1])-centroid[1]

    edge_length = max(edge_length_x, edge_length_y)
    # adjust the edge length by 30% 
    edge_length = edge_length + (edge_length*0.3)

    return edge_length

def pad_coordinates(coordinates, old_image_size, new_image_size):
    '''
    Translates the coordinates so that it will fit a padded image
    Inputs:
        coordinates (2xN array/list) = 2D coordinates
        old_image_size (1x2 list) = height and width of the old image, respectively
        new_image_size (1x2 list) = height and width of the padded image, respectively
    Output:
        new_coords (2xN list) = padded/translated 2D coordinates
    '''
    # get the height difference
    h_diff = new_image_size[0]-old_image_size[0]
    # get the width difference
    w_diff = new_image_size[1]-old_image_size[1]

    # initialise the new coords
    new_x_coords = []
    new_y_coords = []
    # pad the coordinates by adding the h and w diff (divided by 2) to the original coords
    for i in range(len(coordinates[0])):
        new_x_coords.append(coordinates[0][i]+w_diff//2)
        new_y_coords.append(coordinates[1][i]+h_diff//2)

    new_coords = [new_x_coords, new_y_coords]

    return new_coords

def translate_coordinates(coordinates, from_centre, to_centre):
    '''
    Translates coordinates based on the distance between one centre point to another
    Inputs:
        coordinates (2xN list) = 2D coordinates we want to translate
        from_centre (1x2 list) = x and y coordinates of the original centre
        to_centre (1x2 list) = x and y coordinates of the new centre
    Output:
        trans_coords (2xN list) = translated coordinates
    '''
    # calculate the horizontal and vertial distance from the old centre to the new centre
    x_trans = to_centre[0]-from_centre[0]
    y_trans = to_centre[1]-from_centre[1]

    # initialise list for translate x and y coordinates
    trans_x_coords = []
    trans_y_coords = []
    # translate the x and y coordinates by adding the x and y distances.
    for i in range(len(coordinates[0])):
        trans_x_coords.append(coordinates[0][i]+x_trans)
        trans_y_coords.append(coordinates[1][i]+y_trans)

    trans_coords = [trans_x_coords, trans_y_coords]

    return trans_coords

def calculate_distance(point1, point2):
    '''
    Calculate distance between two points
    Inputs:
        point1 (1x2 list) = x and y coordinates of 1st point
        point2 (1x2 list) = x and y coordinates of 2nd point
    Output:
        distance = distance between the two points 
    '''    
    a = abs(point1[0]-point2[0])
    b = abs(point1[1]-point2[1])

    # pythogoras
    distance = sqrt(a**2+b**2)

    return distance
