# This is to connect to omero
from upload_only_cellpose import upload_cellpose_nucleiV2
import sys
# This is the actual application of cellpose
from Dockerfile_build.MainFunctions.get_histogram_dask import get_histogram_dask
from Dockerfile_build.MainFunctions.save_omero_to_zarr import save_omero_to_zarr


def pack_system_arguments():
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

    extraction = upload_cellpose_nucleiV2(sys_arguments, parameters)
    if extraction == 0:
        print("Cell pose coordinates already extracted.")
    elif extraction == 1:
        print("Cell pose coordinates are now extracted and saved to omero.")

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