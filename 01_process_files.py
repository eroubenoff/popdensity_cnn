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



def clip_raster(geo, rstr):
	"""Function that clips a raster to a specified geometry

	Takes a single shapely geometry and a raster. Determines the maximum 
	extent of the geometry and reads the (windowed) raster. Then clips.

	Parameters
	----------
	geo : shapely geometry (polygon)
		- The geometry to clip the raster to
	rstr : rasterio reader
		- Full CA raster reader

	Returns
	-------
	rasterio clipped raster
	"""

	# bounds = geo.bounds

	# rstr = rstr.read([1, 2, 3],
	# 	window = rio.windows.from_bounds(
	# 		bounds[0], bounds[1], bounds[2], bounds[3], 
	# 		transform = ca_raster.transform)
	# 	)

	rstr, _ = mask(rstr, geo.geometry, crop = True)

	return(rstr)






if __name__ == "__main__":
	# Load raster
	raster_path = "data/ca_sentinel2_try2.tif"
	ca_raster = rio.open(raster_path)

	# Load blockgroups shapefile
	blockgroups_path = "data/ca_blockgroup_shp/tl_2020_06_bg.shp"
	ca_blockgroups = gpd.read_file(blockgroups_path)
	ca_blockgroups = ca_blockgroups.to_crs(ca_raster.crs)

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
		bg = ca_blockgroups.loc[[i]]
		arr = reshape_as_image(clip_raster(bg, ca_raster))
		fname = "data/data/bg_" + bg.iloc[0]["GEOID"] + "_pop_" + \
				bg.iloc[0]["pop"] + \
				"_aland_" + str(bg.iloc[0]["ALAND"]) 

		np.save(fname, arr, allow_pickle = False)









