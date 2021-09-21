"""
Script that processes the 1-km Gridded Population of the World (1km, 30 arc-sec)
dataset into a format useful for CNN tranfer learning.

Assumes:
    - data/gpw/gpw_v4_population_density_rev11_2020_30_sec.tif
    - data/ca_blockgroup_shp/tl_2020_06_bg.shp
    - data/acs2019totalpop/ACSDT5Y2019.B01003_data_with_overlays_2021-08-23T193810.csv

"""

import os
import rasterio as rio
from rasterio.mask import mask
import pdb
from shapely.geometry import Polygon, box
from process_files import load_rasters, clip_raster
import numpy as np
from rasterio.plot import reshape_as_image
from joblib import Parallel, delayed
from rasterio.enums import Resampling


def load_gpw(gpw='data/gpw/gpw_v4_population_density_rev11_2020_30_sec.tif',
             extent=[-124.409591, 32.534156, -114.131211, 42.009518],
             rescale_factor = 1.0,
             out='data/gpw/gpw_CA.tif',
             overwrite=True):
    """ Loads the Gridded Population of the World dataset and crops it
    to the extent of California

    Parameters
    ----------
    gpw: string
        - Location of GWP dataset (geotifF)
    extent : list or tuple
        - Cropping extent
    rescale_factor
        - Factor by which to upscale or downscale images (recommened: 1/2)
    out : string
        - Location of the output file
    overwrite: bool
        - Flag to overwrite existing file

    Returns
    -------
    rio.DatasetReader
        - link to the cropped reader
    """

    if os.path.isfile(out) and not overwrite:
        return(rio.open(out))

    with rio.open(gpw) as src:
        out_image, out_transform = mask(src, [box(extent[0], extent[1],
                                                  extent[2], extent[3])],
                                        crop = True)
        out_image[out_image < 0] = 0
        out_meta = src.meta

    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rio.open(out, "w", **out_meta) as dest:
        dest.write(out_image)

    if rescale_factor != 1.0:
        with rio.open(out, "r") as dataset:
            # Resample to target shape
            data = dataset.read(
                out_shape=(
                    dataset.count,
                    int(dataset.height * rescale_factor),
                    int(dataset.width * rescale_factor)
                ),
                resampling=Resampling.average
            )

            # scale image transform
            transform = dataset.transform * dataset.transform.scale(
                (dataset.width / data.shape[-1]),
                (dataset.height / data.shape[-2])
            )

            out_meta = dataset.meta
            out_meta.update({'driver': 'GTiff',
                             'height': data.shape[1],
                             'width': data.shape[2],
                             'transform': transform})

        with rio.open(out, 'w', **out_meta) as dest:
            dest.write(data)

    return(rio.open(out))

def extract_raster_cell(r, x, y):
    """ Extracts the coordinates of a raster cell and returns as polygon

    Parameters
    ---------
    r: rasterio.DataSetReader
        - the raster to read
    x, y: int
        - x and y indices of the pixel to be extracted

    Returns
    ------
    tuple: (float, shapely.polygon)
    """
    return(
        float(r.read(1,
                     window=((x, x+1), (y, y+1))
            )
        ),
        Polygon([
            r.xy(x, y, 'ul'),
            r.xy(x, y, 'ur'),
            r.xy(x, y, 'lr'),
            r.xy(x, y, 'll'),
            r.xy(x, y, 'll')
        ])
    )

def run(i, j, gpw, ca_rasters):
    """Procedure to run a single raster cell

    Parameters
    ----------
    i,j : int
        - indices
    gpw : rio.DatasetReader
    ca_rasters: pd.Dataframe

    Return
    ------
    None

    """
    if (i*gpw.shape[0]  + j) % 1000 == 0:
        print("On", i*gpw.shape[0] + j, "of", gpw.shape[0]*gpw.shape[1])

    cell = list(extract_raster_cell(gpw, i, j))
    cell[1] = clip_raster([cell[1]], ca_rasters)

    # Filter nan values
    if np.abs(cell[0] ) > 10e10:
        return
    # Filter all-zero images
    if np.nanmean(cell[1][0]) == 0.0:
        return

    # r is a tuple of (value, [list of ndarrays])
    for k, r in enumerate(cell[1]):
        r = reshape_as_image(r)
        fname = "data/data/gpw_{x}_{y}_dens_{d}_{k}".format(
            x = i, y = j, d = cell[0], k = k)
        np.save(fname, r, allow_pickle=False)

    return

if __name__ == '__main__':
    # Load Rasters
    ca_rasters = load_rasters()

    # Load population data and convert to geopandas
    gpw = load_gpw(rescale_factor = 1/4)

    for i in range(gpw.shape[0]):
        for j in range(gpw.shape[1]):
            run(i, j, gpw, ca_rasters)



