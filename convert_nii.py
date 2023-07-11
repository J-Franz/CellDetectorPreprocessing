import os
import time
import zarr
import dask.array as da
from JonasTools.omero_tools import refresh_omero_session, get_image, get_pixels, get_tile_coordinates
from utils import extract_system_arguments, unpack_parameters
import numpy as np
import nibabel as nib


image = nib.load("/share/Work/Neuropathologie/MicrogliaDetection/SliceAlign/84590_image_012.nii")


print("Test")