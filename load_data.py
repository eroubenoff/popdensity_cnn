"""
Loads all data in data/data and parses into a format that is useful
for tensorflow

Ethan Roubenoff

August 2021
"""

import numpy as np
import re
import os
# import tensorflow as tf


def load_data(dirname="data/data", verbose=True):
    """ Loads all files in `dirname` and returns them in numpy object

    Parameters
    ----------
    dirname : string
        - Location to look for images
    size : 2 element list
        - Size to reshape images
    Returns
    -------
    (np.ndarray, np.array)
        - Arrays of image data (n x w x h x b) and pop density (n)

    """

    def process_file(fname, dirname="data/data"):
        """ Loads and processes a single file.

        Assumes that all files are in numpy format, with image band order.

        Parameters
        ----------
        fname: string
            - Object to be processed
        size : 2-element list
            - Size of resized image
        dirname : string
            - Location of object
        Returns
        -------
        (np.ndarray, float)
            - image and associated value

        """

        f = np.load(os.path.join(dirname, fname))

        fname = re.split('_|\.', fname)
        popdensity = int(fname[3])/int(fname[5])/(1e-6)

        return(f, popdensity)

    flist = os.listdir(dirname)

    datalist = []
    valuelist = []

    for i, f in enumerate(flist):
        if i % 1000 == 0:
            print("Loading file", i, "of", len(flist))
        try:
            tmp = process_file(f, dirname)
            # if ~all(tmp[0].shape): # Check if any dimensions are 0
            #   continue
        except ZeroDivisionError:
            continue
        datalist.append(tmp[0])
        valuelist.append(tmp[1])

    # datalist = np.concatenate(datalist)
    # valuelist = np.concatenate(valuelist)

    return(datalist, valuelist)


def process_data(datalist: list, valuelist: list, shape=(224, 224)):
    """ Processes the output of load_data into the desired format

    Instead of resampling, this will create tiles of the input
    shape as subsamples

    Parameters
    ---------
    datalist, valuelist: list
        - Output of load_data
    shape: list or tuple
        - Desired resultant image shape

    Returns
    ------
    np.ndarray, np.array
        - Inputs and corresponding outputs, concatenated
    """

    def reshape_split(image: np.ndarray, shape: tuple):
        """ Helper function that takes in a single image/ndarray
        and returns that image split into tiles of shape `shape`

        Uses help fromhttps://towardsdatascience.com/efficiently-splitting-an-
            image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7

        Paramters
        ---------
        image: ndarray
            - Image to be split, in format (width x height x channels)
        shape :tuple

        Returns
        ------
        np.ndarray
            - Tiled image as an ndarray
        """

        img_height, img_width, channels = image.shape
        tile_height, tile_width = shape

        # If image height/width is not a multiple of tile height/width, pad:
        img_height = tile_height*(img_height // tile_height + 1)
        img_width = tile_width*(img_width // tile_width + 1)

        image = np.pad(image, ((0, img_height - image.shape[0]),
                               (0, img_width - image.shape[1]),
                               (0, 0)))

        tiled_array = image.reshape(img_height // tile_height,
                                    tile_height,
                                    img_width // tile_width,
                                    tile_width,
                                    channels)

        tiled_array = tiled_array.swapaxes(1, 2)

        # Reshape to be all along the first axis
        tiled_array = tiled_array.reshape(-1, tile_height, tile_width, 3)
        return(tiled_array)

    for d in range(len(datalist)):
        if d % 1000 == 0:
            print("Processing", d, "of", len(datalist))

        datalist[d] = reshape_split(datalist[d], shape)

    # Repeat values however many times needed
    valuelist = np.repeat(valuelist, [x.shape[0] for x in datalist])
    datalist = np.concatenate(datalist)

    return(datalist, valuelist)


if __name__ == "__main__":
    datalist, valuelist = load_data()
    datalist, valuelist = process_data(datalist, valuelist)
