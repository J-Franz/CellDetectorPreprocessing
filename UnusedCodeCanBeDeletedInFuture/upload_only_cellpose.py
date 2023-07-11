import os

from JonasTools.omero_tools import refresh_omero_session, get_image, UploadArrayAsTxtToOmero, \
    check_fname_omero, make_omero_file_available
from Utils.utils import extract_system_arguments, unpack_parameters


def upload_cellpose_nucleiV2(sys_arguments, parameters):
    ## Get credentials from first argumentma
    # run like main.py password
    c_dapi, c_fluorescence, imageId, base, gpu, pw, user = extract_system_arguments(sys_arguments)

    if gpu:
        use_gpu = False
        plot_cellpose = False
    # TODO: Refactor plot_cellpose
    else:
        use_gpu = True  # requires cellpose to be installed in gpu usable fashion
        plot_cellpose = False  # should be false for cluster

    evaluated_crop_size, half_height, half_width, height, maximum_crop_size, overlap, width = unpack_parameters(
        parameters)
    os.system("echo \"Hello!\"")
    # set working directory
    cellpose_path = "%sCelldetectorPreprocessed/Cellpose/"%base
    os.makedirs(cellpose_path,exist_ok=True)
    # Load image for the first time

    with refresh_omero_session(None, user, pw) as conn:
        image = get_image(conn, imageId)
        group_name = image.getDetails().getGroup().getName()
        max_c = image.getSizeC() - 1  # counting starts from 0
        # Set up plan to iterate over whole slide
        nx_tiles = int(image.getSizeX() / evaluated_crop_size) + 1
        ny_tiles = int(image.getSizeY() / evaluated_crop_size) + 1
        n_runs = ny_tiles * nx_tiles
        # Define channels automatically if not specified
        if c_dapi is None:
            c_dapi = 0
        if c_fluorescence is None:
            c_fluorescence = max_c
        #check if already extracted
        fname = str(imageId) + "_Cellpose2AllNucleiCentroidsV2_c" + str(c_fluorescence) + ".txt"
        already_extracted = check_fname_omero(fname, image)

        if already_extracted:
            make_omero_file_available(image, fname, cellpose_path)
            return 0

    os.system("echo \"We will start to upload the cellpose file.\"")

    verbose_upload = True
    UploadArrayAsTxtToOmero(cellpose_path + fname, None, group_name, imageId, pw, user, verbose_upload)
    return 1


