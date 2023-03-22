import os
import time

import numpy as np
from scipy.stats import skew

from image_tools import get_coordinates, delete_cells_at_border
from JonasTools.omero_tools import refresh_omero_session, get_image, get_pixels, get_tile_coordinates, UploadArrayAsTxtToOmero, \
    check_fname_omero, make_omero_file_available
from utils import extract_system_arguments, unpack_parameters


def extract_cellpose_nucleiV2(sys_arguments, parameters):
    ## Get credentials from first argumentma
    # run like main.py password
    c_dapi, c_fluorescence, imageId, base, gpu, pw, user = extract_system_arguments(sys_arguments)

    if gpu:
        use_gpu = False
        plot_cellpose = False
    else:
        use_gpu = True  # requires cellpose to be installed in gpu usable fashion
        plot_cellpose = False  # should be false for cluster

    evaluated_crop_size, half_height, half_width, height, maximum_crop_size, overlap, width = unpack_parameters(
        parameters)

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
    print("We will start to crop the image in " + str(n_runs) + " tiles and start the analysis now.")

    os.system("echo \"We will start to crop the image in " + str(n_runs) + " tiles and start the analysis now.\"")


    os.system("echo \"We will store the image in " + cellpose_path + " tiles and start the analysis now.\"")

    list_all_x = []
    list_all_y = []
    list_all_mean = []
    list_all_median = []
    list_all_var = []
    list_all_skew = []

    n_run = 0

    for nx in range(nx_tiles):  # [nx_tiles-1]:#
        for ny in range(ny_tiles):  # [ny_tiles-2]:#
            print("I just started with tile nr.:" + str(nx * ny_tiles + ny))
            start_time = time.time()
            # To avoid lost connection reconnect every time before downloading
            with refresh_omero_session(None, user, pw) as conn:
                image = get_image(conn, imageId)
                pixels = get_pixels(conn, image)

                # Define local crop area
                tile_coordinates = get_tile_coordinates(image, nx, ny, evaluated_crop_size, maximum_crop_size)

                # display current crop for potential debugging
                print("The current crop is crop_width,crop_height,current_crop_x,current_crop_y:",
                      tile_coordinates)

                # load Dapi Channel and Fluorescence channel
                # getTile switches width and height...
                # this is compensated by get_coordinates to return switched coordinates
                conn.c.enableKeepAlive(60)
                tile_dapi = pixels.getTile(0, theC=c_dapi, theT=0, tile=tile_coordinates.values())
                conn.c.enableKeepAlive(60)
                tile_fluorescence = pixels.getTile(0, theC=c_fluorescence, theT=0, tile=tile_coordinates.values())

            # Apply cellpose to extract coordinates, potentially one could also save masks and boundaries
            [list_nuclei_x_coords, list_nuclei_y_coords] = get_coordinates(tile_dapi, plot_cellpose, use_gpu)

            # cast list to numpy
            x_coords_nuclei = np.array(list_nuclei_x_coords)
            y_coords_nuclei = np.array(list_nuclei_y_coords)

            # take only nuclei within border of overlap/2+1
            # also take into account that cropped regions might be smaller than maximum_crop_size

            x_coords_nuclei, y_coords_nuclei = delete_cells_at_border(x_coords_nuclei, y_coords_nuclei, overlap,
                                                                      tile_coordinates)

            for id_x, x in enumerate(x_coords_nuclei):
                temp_crop_x = (x - half_width)
                temp_crop_y = (y_coords_nuclei[id_x] - half_height)
                tmp_array = tile_fluorescence[temp_crop_y:(temp_crop_y + height), temp_crop_x:(temp_crop_x + width)]

                list_all_mean.append(np.mean(tmp_array.flatten()))
                list_all_median.append(np.median(tmp_array.flatten()))
                list_all_var.append(np.var(tmp_array.flatten()))
                list_all_skew.append(skew(tmp_array.flatten()))

                # transform from local coordinates of tile to coordinates of wsi

            x_coords_nuclei = x_coords_nuclei + tile_coordinates['current_crop_x']
            y_coords_nuclei = y_coords_nuclei + tile_coordinates['current_crop_y']

            list_all_x.append(x_coords_nuclei)
            list_all_y.append(y_coords_nuclei)

            stop_time = time.time()
            report = "Processing of this tile took: %s" % (stop_time - start_time)
            os.system("echo \"%s\"" % report)
    print(n_run)

    list_all_x = np.concatenate(list_all_x)
    list_all_y = np.concatenate(list_all_y)

    concat_array = np.array((list_all_x,
                             list_all_y,
                             list_all_mean,
                             list_all_median,
                             list_all_var,
                             list_all_skew), dtype=float)

    verbose_upload = True
    UploadArrayAsTxtToOmero(cellpose_path + fname, concat_array.T, group_name, imageId, pw, user, verbose_upload)
    return 1


