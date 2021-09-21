"""
Loads all data in data/data and parses into a format that is useful
for tensorflow

Ethan Roubenoff

August 2021
"""

import numpy as np
import re
import os
import random
# import tensorflow as tf


def load_files(dirname="data/data", verbose=True):
    """ Loads all files in `dirname` and returns them in numpy object

    Parameters
    ----------
    dirname : string
        - Location to look for images
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
        dirname : string
            - Location of object
        Returns
        -------
        (np.ndarray, float)
            - image and associated value

        """

        f = np.load(os.path.join(dirname, fname))

        fname = re.split('_|\.', fname)
        popdensity = float(fname[4] + "." + fname[5])

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
    """ Processes the output of load_files into the desired format

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


def load_data(dirname='data/data', shape=(224, 224),
              cache=True,
              overwrite_cache=False,
              random_state = 49):
    """ Wrapper around load_files and process_data. Takes in a directory
    and returns a tuple of training and test data and values.

    Now works using mmaped arrays, so data are effectively always cached.

    Parameters
    ---------
    dirname : string
        - Location of the split files
    shape : tuple
        - Dimensions of sub-images
    cache : bool
        - if True, save a version of the data in the 'data' folder
    overwrite_cache : bool
        - if True, delete cached versions
    tt_split : bool
        - store/retrieve the data pre-split into test + train

    Returns
    ------
    tuple: data, values
    """

    if cache and \
            "X_train.npy" in os.listdir("data") and \
            "X_test.npy" in os.listdir("data") and \
            "y_train.npy" in os.listdir("data") and \
            "y_test.npy" in os.listdir("data"):
        if overwrite_cache:
            os.remove('data/X_train.npy')
            os.remove('data/X_test.npy')
            os.remove('data/y_train.npy')
            os.remove('data/y_test.npy')
        else:
            X_train = np.load("data/X_train.npy", mmap_mode = "r")
            X_test = np.load("data/X_test.npy", mmap_mode = "r")
            y_train = np.load("data/y_train.npy", mmap_mode = "r")
            y_test = np.load("data/y_test.npy", mmap_mode = "r")

            if (datalist.shape[1] != shape[0]) or \
                    (datalist.shape[2] != shape[1]):
                raise ValueError("Cached data shape doesn't match requested \
                                 data shape")
            return(datalist, valuelist)

    datalist, valuelist = load_files(dirname)

    """
    if shape is None:
        shapelist = [x.shape for x in datalist]
        dim0 = [x[0] for x in shapelist]
        dim1 = [x[1] for x in shapelist]
        vals,counts = np.unique(array, return_counts=True)
        index = np.argmax(counts)
    """
    datalist, valuelist = process_data(datalist, valuelist, shape)


    datashape = datalist.shape
    valueshape = valuelist.shape
    n = datashape[0]
    random.seed(random_state)
    shuffled = random.sample(range(n-1),  k=(n-1))
    train_idx = shuffled[(n//3):]
    test_idx = shuffled[:(n//3)]


    # Create mmap arrays
    X_train = np.memmap("data/X_train", dtype="uint8", mode="w+",
                       shape=(len(train_idx), shape[0], shape[1], 3))
    X_test = np.memmap("data/X_test", dtype="uint8", mode="w+",
                       shape=(len(test_idx), shape[0], shape[1], 3))
    y_train = np.memmap("data/y_train", dtype="float", mode="w+",
                        shape=(len(train_idx)))
    y_test = np.memmap("data/y_test", dtype="float",  mode="w+",
                       shape=(len(test_idx)))

    X_train = datalist[train_idx, :, :, :]
    X_test = datalist[test_idx, :, :, :]
    y_train = valuelist[train_idx]
    y_test = valuelist[test_idx]


    return(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    # datalist, valuelist = process_data(datalist, valuelist)
