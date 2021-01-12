import os
import time
import zarr

from omero_tools import refresh_omero_session, get_image, get_pixels, get_tile_coordinates
from utils import extract_system_arguments, unpack_parameters, define_path


def save_omero_to_zarr(sys_arguments, parameters):
    c_dapi, c_fluorescence, imageId, local, pw, user = extract_system_arguments(sys_arguments)

    evaluated_crop_size, half_height, half_width, height, maximum_crop_size, overlap, width = unpack_parameters(
        parameters)

    with refresh_omero_session(None, user, pw) as conn:
        image = get_image(conn, imageId)

        size_x = image.getSizeX()
        size_y = image.getSizeY()
        size_c = image.getSizeC()
        max_c = size_c - 1  # counting starts from 0

    # Set up plan to iterate over whole slide
    evaluated_crop_size = maximum_crop_size  # to load non_overlapping

    nx_tiles = int(size_x / evaluated_crop_size) + 1
    ny_tiles = int(size_y / evaluated_crop_size) + 1
    n_runs = ny_tiles * nx_tiles
    # Define channels automatically if not specified
    if c_dapi is None:
        c_dapi = 0
    if c_fluorescence is None:
        c_fluorescence = max_c

    path = define_path(local)

    fname = str(imageId) + "_image.zarr"
    for filename_ in os.listdir(path):
        if filename_ == fname:
            return 0

    # initialize zarr file
    z1 = zarr.open(path + fname, mode='w', shape=(size_x, size_y, size_c), chunks=(500, 500), dtype='i2')

    os.system("echo \"We will start to crop the image in " + str(n_runs) + " tiles and start saving as Zarr now.\"")

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

                current_crop_x = tile_coordinates["current_crop_x"]
                current_crop_y = tile_coordinates["current_crop_y"]
                current_crop_x_plus_width = current_crop_x + tile_coordinates["crop_width"]
                current_crop_y_plus_width = current_crop_y + tile_coordinates["crop_height"]
                for c in range(size_c):
                    conn.c.enableKeepAlive(60)
                    # Transpose as getTile returns Y,X to get X,Y
                    tile_ = pixels.getTile(0, theC=c, theT=0, tile=tile_coordinates.values()).T
                    z1[current_crop_x:current_crop_x_plus_width,
                        current_crop_y:current_crop_y_plus_width, c] = tile_

            stop_time = time.time()
            report = "Processing of this tile took: %s" % (stop_time - start_time)
            os.system("echo \"%s\"" % report)
    return 1
