'''
Handles code for converting between contours and masks

Author: tfin440
'''
from typing import List, Dict

import numpy as np
import cv2

from . import cvi42

def to_contours( masks: Dict[str, np.ndarray] ) -> cvi42.ContourSet:
    '''
    Converts from masks to a contour set.

    # Arguments:

    masks: A dictionary matching the name to the binary mask.
    '''

    contours = {}

    for name, mask in masks.items():
        _, contours[name], _ = cv2.findContours( mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset = (0, 0) )

    contour_set = cvi42.ContourSet( contours )
    contour_set.dicom_id = 'extracted' # To signify it was extracted from masks

    return contour_set

def from_contours( contour_set: cvi42.ContourSet, layers: List[str], original_shape, precision: int = 1 ) -> Dict[str, np.ndarray]:
    '''
    Converts a contour set to a single binary mask image with multiple channels

    # Arguments
    contour_set: The set of contours that can be converted.
    layers: The list of contours to convert to masks, in the order they shall appear as channels in the output
    original_shape: The x and y dimensions of the image these contours were taken from. If they are not exact, then
        the contour masks will be misaligned

    # Returns
    A list of masks, of shape [x, y] and type boolean and ordered according to the input layers
    '''
    masks = []
    
    precision_shape = np.array( original_shape ) * precision
    for contour_name in layers:
        contour = contour_set.get_contour( contour_name )

        if contour is not None:
            # Use as much precision as possible, as drawContours requires ints
            contour = (contour * precision).astype( np.int )
            image = np.zeros( [*precision_shape, 3], np.uint8 )

            cv2.drawContours( image, [contour], -1, (1, 0, 0), thickness = cv2.FILLED, offset = (0, 0) )
            # And interpolate back down to original size
            masks.append( cv2.resize( image[..., 0], original_shape[::-1], interpolation = cv2.INTER_NEAREST ) )
        else:
            masks.append( np.zeros( original_shape, dtype = np.uint8 ) )

    return masks