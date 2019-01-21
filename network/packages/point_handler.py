import numpy as np
import pydicom as dicom
import os

def get_model_points_from_file(filepath, delimiter= ' ', names=True):
    data = np.genfromtxt(filepath, delimiter=' ', names=True)
    points = (data['imageX'], data['imageY'])
    return points

'''
    Get the list of coordinates, regions, and other properties from the _strain.dat file
    Return a tuple of (x coord, y coords, LV (AHA) segment, elements (4 segment), division, radial index )
'''
def get_model_points_and_label_from_file(filepath, delimiter= ' ', names=True):
    data = np.genfromtxt(filepath, delimiter=' ', names=True)
    #points = (data['imageX'], data['imageY'], "{0}{1}{2}{3}".format(data['Region'],data['Element'],data['SubDivXi1'],data['SubDivXi2']))
    points = (data['imageX'], data['imageY'], data['Region'],data['Element'],data['SubDivXi1'],data['SubDivXi2'])
    return points

'''
    Not used in the current network
'''
# get a list of [[x points], [y points]] from a file
def get_grid_from_file(filepath, delimiter= ' ', names=True):
    data = np.genfromtxt(filepath, delimiter=' ', names=True)

    grid = []
    for el in np.unique(data['Element']):
        elData = data[data['Element'] == el]
        for div1 in np.unique(elData['SubDivXi1']):
            div1Data = elData[elData['SubDivXi1'] == div1]
            grid.append((div1Data['imageX'], div1Data['imageY']))

        for div2 in np.unique(elData['SubDivXi2']):
            div2Data = elData[elData['SubDivXi2'] == div2]
            grid.append((div2Data['imageX'], div2Data['imageY']))
    return grid

# Plot a grid, a grid is a list of pairs of ([x points], [y points])
def plot_grid(ax, grid, title, ConstPixelDims, grid_color, legend):
    diff  = abs(ConstPixelDims[0]-ConstPixelDims[1])/2
    count = 0
    for line in grid:
        if count == 0:
            ax.plot(line[0] - diff, ConstPixelDims[0] - line[1], linewidth=0.5, color=grid_color, label=legend)
        else:
            ax.plot(line[0] - diff, ConstPixelDims[0] - line[1], linewidth=0.5, color=grid_color)
            
        count += 1
    ax.set_title(title)
    return ax

def plot_model_points(ax, points, title, ConstPixelDims, grid_color, legend):
    diff  = abs(ConstPixelDims[0]-ConstPixelDims[1])/2
    ax.scatter(points[0] - diff, ConstPixelDims[0] - points[1], s=0.5, color=grid_color, label=legend)
    ax.set_title(title)
    return ax

'''
    Points is an array with 6 elements
    See: get_model_points_and_label_from_file
    ['imageX'], ['imageY'], ['Region'], ['Element'],['SubDivXi1'],['SubDivXi2'])
'''
def to_dicom_coords(labeled_points, dimensions):
    coords = [labeled_points[0], (dimensions[0]-labeled_points[1])]
    return coords

def get_regions(labeled_points):
    return labeled_points[2]

# For this to work, the dimension must be the same
def plot_to_mask(points, dimensions):
    mask = np.zeros(shape=dimensions)
    x = np.rint(points[0]).astype(int)
    y = np.rint(dimensions[0]-points[1]).astype(int)
    mask[y,x] = 1
    #mask[x,y] = "{0}{1}{2}{3}".format(points[2],points[3],points[4],points[5])
    return mask

def save_grid(plot, output_dir, filename):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # plot.savefig(f'{output_dir}\\{filename}', dpi=600)
    plot.savefig('{}\\{}'.format(output_dir, filename), dpi=600)

def get_all_image_filepath_from_ptr(filepath):
    pathlist = np.genfromtxt(filepath, delimiter='\t', names='series, slice, index, path', skip_header=1, dtype=None)
    return pathlist
