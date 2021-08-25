"""
Scrit that takes the large image raster, overlays a block group, and saves
the resultant raster to a file with name format:

bg_{block group GEOID}_pop_{Total Population}_aland_{Area in Meters}

File format is numpy binary (fastest and most compressed)


Ethan Roubenoff
August 2021

"""



import rasterio as rio
# from rasterio.windows import Window
from rasterio.mask import mask
from rasterio.plot import show
import geopandas as gpd 
import pandas as pd
import fiona
import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np
from rasterio.plot import reshape_as_raster, reshape_as_image
import os
import re

def load_rasters(dirname = "data/ca_sentinel"):
	""" Loads multiple rasters and returns a dataframe of their metadata

	Parameters
	----------
	dirname : str
		- Location of the rasters to load
	Return
	------
	pd.DataFrame
		- Dataframe with columns: xmin, ymin, r, which contain (respectively)
		the minimum x and y pixels, and a raster dataset reader
	"""

	flist = os.listdir(dirname)
	flist = [f for f in flist if ".tif" in f]

	df = pd.DataFrame(columns = ["xmin", "ymin", "r"])

	for f in flist:
		r = rio.open(os.path.join(dirname, f)) 
		f = re.split('-|\.', f)
		f = {'xmin' : int(f[1]), 'ymin': int(f[2]), 'r': r}

		df = df.append(f, ignore_index = True)

	return(df)




def clip_raster(geo, rstr_df):
	"""Function that clips multiple rasters to a specified geometry

	Takes a single shapely geometry and a raster dataframe. Expects
	the raster dataframe to have columns (xmin, ymin, r).
	If the poly overlaps with any of the rasters, return the intersection.

	Original plan was to check the extent but rasterio.mask.mask does that
	automatically, so just catch the rasterio.errors.WindowError and continue.

	Parameters
	----------
	geo : shapely geometry (polygon)
		- The geometry to clip the raster to
	rstr_df : pd.DataFrame
		- Dataframe with columns (xmin: int, ymin: int, r: rasterio.reader())

	Returns
	-------
	list of np.ndarray
		- Each item is a raster
	"""

	# bounds = geo.bounds

	# rstr = rstr.read([1, 2, 3],
	# 	window = rio.windows.from_bounds(
	# 		bounds[0], bounds[1], bounds[2], bounds[3], 
	# 		transform = ca_rasters.transform)
	# 	)

	rlist = []

	for r in rstr_df.r:
		try:
			rstr, _ = mask(r, geo.geometry, crop = True)
			rlist.append(rstr)
		except (ValueError, rio.errors.WindowError):
			continue


	return(rlist)






if __name__ == "__main__":
	# Load rasters
	ca_rasters = load_rasters()

	# Load blockgroups shapefile
	blockgroups_path = "data/ca_blockgroup_shp/tl_2020_06_bg.shp"
	ca_blockgroups = gpd.read_file(blockgroups_path)
	ca_blockgroups = ca_blockgroups.to_crs(ca_rasters.loc[0, "r"].crs)

	# Load blockgroups data
	pop_path = "data/acs2019totalpop/ACSDT5Y2019.B01003_data_with_overlays_2021-08-23T193810.csv"
	ca_pop = pd.read_csv(pop_path)

	# Join blockgroups data with shapefile
	ca_pop['GEOID'] = [str(x)[9:] for x in ca_pop['GEO_ID']]
	ca_pop = ca_pop.rename(columns= {'B01003_001E' : 'pop'})
	ca_blockgroups = ca_blockgroups.merge(ca_pop[['GEOID', 'pop']])


	for i in range(len(ca_blockgroups)):
		"""
		Cycle through all block groups, create the mask, and write out
		to a numpy binary. This chosen for space efficiency over tif images.
		"""

		if i % 1000 == 0:
			print("On", i, "of", len(ca_blockgroups))

		# Select the correct block group as a 1-element dataframe
		bg = ca_blockgroups.loc[[i]]

		# Clip the rasters to extent (return is a list of ndarrays)
		rasters = clip_raster(bg, ca_rasters)

		for j, r in enumerate(rasters):
			# Reshape to be bands-last order
			arr = reshape_as_image(r)

			# File name
			fname = "data/data/bg_" + bg.iloc[0]["GEOID"] + \
					"_pop_" + bg.iloc[0]["pop"] + \
					"_aland_" + str(bg.iloc[0]["ALAND"]) + \
					"_" + str(j)

			np.save(fname, arr, allow_pickle = False)









