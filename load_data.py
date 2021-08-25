"""
Loads all data in data/data and parses into a format that is useful 
for tensorflow

Ethan Roubenoff

August 2021
"""

import numpy as np
import re
import os
import tensorflow as tf

def process_file(fname, size = [10,10], dirname = "data/data"):
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
	f = tf.image.resize(f, size)


	fname = re.split('_|\.', fname)
	popdensity = int(fname[3])/int(fname[5])/(1e-6)

	return(f, popdensity)


def load_data(dirname = "data/data", size = [10,10]):
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



	flist = os.listdir(dirname)

	datalist = []
	valuelist = []

	for f in flist:
		try: 
			tmp = process_file(f, size, dirname)
			# if ~all(tmp[0].shape): # Check if any dimensions are 0
			# 	continue
		except ZeroDivisionError as e:
			continue
		datalist.append(tmp[0])
		valuelist.append(tmp[1])

	# datalist = np.concatenate(datalist)
	# valuelist = np.concatenate(valuelist)

	return(datalist, valuelist)

