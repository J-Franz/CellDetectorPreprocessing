import os
import sys



def extract_system_arguments(sys_arguments):
    omero_instance = sys_arguments["omero_instance"]
    user = sys_arguments["user"]
    pw = sys_arguments["pw"]
    imageId = sys_arguments["imageId"]
    base = sys_arguments["base"]
    gpu = sys_arguments["gpu"]
    c_fluorescence = sys_arguments["c_fluorescence"]
    c_dapi = sys_arguments["c_dapi"]
    return c_dapi, c_fluorescence, imageId, base, gpu, pw, user, omero_instance


def unpack_parameters(parameters):
    # load parameters
    maximum_crop_size = parameters["maximum_crop_size"]
    overlap = parameters["overlap"]
    evaluated_crop_size = parameters["evaluated_crop_size"]
    width = parameters["width"]
    height = parameters["height"]
    half_width = parameters["half_width"]
    half_height = parameters["half_height"]
    return evaluated_crop_size, half_height, half_width, height, maximum_crop_size, overlap, width


def pack_system_arguments():
    ## Get credentials from first argumentma
    # run like main_cellpose.py password
    omero_instance = sys.argv[1]
    user = sys.argv[2]
    pw = sys.argv[3]
    imageId = sys.argv[4]
    base = sys.argv[5]
    gpu = (sys.argv[6]=="True")
    try:
        c_fluorescence = int(sys.argv[7])
    except IndexError:
        c_fluorescence = None
    try:
        c_dapi = int(sys.argv[8])
    except IndexError:
        c_dapi = None
    # pack sys argv
    sys_arguments = {"omero_instance": omero_instance,
                     "user": user,
                     "pw": pw,
                     "imageId": imageId,
                     "base": base,
                     "gpu": gpu,
                     "c_fluorescence": c_fluorescence,
                     "c_dapi": c_dapi}
    return sys_arguments



def pack_parameters():
    # define parameters:
    maximum_crop_size = 10000
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