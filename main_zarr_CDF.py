# This is to connect to omero
from MainFunctions.extract_cellpose_nucleiV2 import extract_cellpose_nucleiV2
import sys
# This is the actual application of cellpose
from MainFunctions.get_histogram_dask import get_histogram_dask
from MainFunctions.save_omero_to_zarr import save_omero_to_zarr


def pack_system_arguments():
    ## Get credentials from first argumentma
    # run like main.py password
    user = "Franz"
    pw = sys.argv[1]
    imageId = sys.argv[2]
    base = sys.argv[3]
    gpu = (sys.argv[4]=="True")
    try:
        c_fluorescence = int(sys.argv[5])
    except IndexError:
        c_fluorescence = None
    try:
        c_dapi = int(sys.argv[6])
    except IndexError:
        c_dapi = None
    # pack sys argv
    sys_arguments = {"user": user,
                     "pw": pw,
                     "imageId": imageId,
                     "base": base,
                     "gpu": gpu,
                     "c_fluorescence": c_fluorescence,
                     "c_dapi": c_dapi}
    return sys_arguments



def pack_parameters():
    # define parameters:
    maximum_crop_size = 1000
    overlap = 100
    evaluated_crop_size = maximum_crop_size - overlap
    width = 101
    height = 101
    half_width = int(width / 2.)
    half_height = int(height / 2.)
    parameters = {"maximum_crop_size": maximum_crop_size,
                  "overlap": overlap,
                  "evaluated_crop_size": evaluated_crop_size,
                  "width": width,
                  "height": height,
                  "half_width": int(width / 2.),
                  "half_height": int(height / 2.)}
    return parameters





if __name__ == '__main__':
    sys_arguments = pack_system_arguments()

    parameters = pack_parameters()

    # rewrite and better pull out zarr storage and histogram dask on CPU
    zarr_storage = save_omero_to_zarr(sys_arguments, parameters)
    if zarr_storage == 0:
        print("Zarr storage already initiated.")
    elif zarr_storage == 1:
        print("Zarr storage is initiated now saved in this file system.")

    hist_calculation = get_histogram_dask(sys_arguments, parameters)
    print(hist_calculation)
    if hist_calculation == 0:
        print("CDF already extracted.")
    elif hist_calculation == 1:
        print("CDF is now extracted and saved to omero.")