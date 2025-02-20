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


    hist_calculation = get_histogram_dask(sys_arguments, parameters)
    print(hist_calculation)
    if hist_calculation == 0:
        print("CDF already extracted.")
    elif hist_calculation == 1:
        print("CDF is now extracted and saved to omero.")