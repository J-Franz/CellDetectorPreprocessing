# This is to connect to omero
from MainFunctions.extract_cellpose_nucleiV2 import extract_cellpose_nucleiV2
import sys
# This is the actual application of cellpose
from MainFunctions.get_histogram_dask import get_histogram_dask
from MainFunctions.save_omero_to_zarr import save_omero_to_zarr

from Utils.utils import pack_system_arguments, pack_parameters



if __name__ == '__main__':
    sys_arguments = pack_system_arguments()

    parameters = pack_parameters()

    zarr_storage = save_omero_to_zarr(sys_arguments, parameters)
    if zarr_storage == 0:
        print("Zarr storage already initiated.")
    elif zarr_storage == 1:
        print("Zarr storage is initiated now saved in this file system.")
