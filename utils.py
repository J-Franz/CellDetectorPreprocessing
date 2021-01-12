import os


def define_path(local):
    if local:
        path = "/share/Work/Neuropathologie/MicrogliaDetection/Playground/Temporary/"
    else:
        path = "/scratch/users/jfranz/Analysis/Cellpose/MicrogliaDepletion/"
        os.makedirs(path, exist_ok=True)
        print("Results are saved to directory: " + path)
    return path


def extract_system_arguments(sys_arguments):
    user = sys_arguments["user"]
    pw = sys_arguments["pw"]
    imageId = sys_arguments["imageId"]
    local = sys_arguments["local"]
    c_fluorescence = sys_arguments["c_fluorescence"]
    c_dapi = sys_arguments["c_dapi"]
    return c_dapi, c_fluorescence, imageId, local, pw, user


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