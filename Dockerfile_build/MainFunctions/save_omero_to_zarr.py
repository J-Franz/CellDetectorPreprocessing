import os
import time
import zarr
import numpy as np

from JonasTools.omero_tools import refresh_omero_session, get_image, get_pixels, get_tile_coordinates
from Dockerfile_build.Utils.utils import extract_system_arguments, unpack_parameters


def save_omero_to_zarr(sys_arguments, parameters):
    c_dapi, c_fluorescence, imageId, base, gpu, pw, user = extract_system_arguments(sys_arguments)

    evaluated_crop_size, half_height, half_width, height, maximum_crop_size, overlap, width = unpack_parameters(
        parameters)

    with refresh_omero_session(None, user, pw) as conn:
        image = get_image(conn, imageId)

        size_x = image.getSizeX()
        size_y = image.getSizeY()
        size_c = image.getSizeC()
        pixel_range = image.getPixelRange()
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

    zarr_path = "%sCelldetectorPreprocessed/Zarr/"%base
    os.makedirs(zarr_path, exist_ok=True)
    print("Results are saved to directory: " + zarr_path)

    fname = str(imageId) + "_image.zarr"
    for filename_ in os.listdir(zarr_path):
        if filename_ == fname:
            return 0

    # initialize zarr file
    # this is from Q3 2023 in support of unsigned 16 bit integer
    # also included a check if all pixel files are in range
    zarr_dtype = "u2"
    if (np.iinfo(zarr_dtype).min<=pixel_range[0]) & (np.iinfo(zarr_dtype).max >= pixel_range[1]):
        z1 = zarr.open(zarr_path + fname, mode='w', shape=(size_x, size_y, size_c), chunks=(500, 500), dtype=zarr_dtype)
    else:
        raise ValueError("The input pixel range is NOT covered by the current supported zarr format 16 bit unsigned integer. Abort.")

    os.system("echo \"We will start to crop the image in " + str(n_runs) + " tiles and start saving as Zarr now.\"")
    verbose = False
    for nx in range(nx_tiles):  # [nx_tiles-1]:#
        for ny in range(ny_tiles):  # [ny_tiles-2]:#
            process_tile(evaluated_crop_size, imageId, maximum_crop_size, nx, ny, ny_tiles, pw, size_c, user, verbose,
                         z1)
    return 1


def process_tile(evaluated_crop_size, imageId, maximum_crop_size, nx, ny, ny_tiles, pw, size_c, user, verbose, z1):
    """
    Process a single tile of an image.

    Args:
        evaluated_crop_size (int): The evaluated crop size.
        imageId (int): The ID of the image to process.
        maximum_crop_size (int): The maximum crop size.
        nx (int): The current x index of the tile.
        ny (int): The current y index of the tile.
        ny_tiles (int): The total number of y tiles.
        pw (str): The password for the OMERO connection.
        size_c (int): The number of channels.
        user (str): The user name for the OMERO connection.
        verbose (bool): Whether to print verbose output.
        z1 (numpy.ndarray): The array to store the processed tile.

    Returns:
        None: This function updates the z1 array in place.

    Note:
        This function retrieves and processes a specific tile from an OMERO image.
        It establishes a connection, fetches the required image and pixel data,
        defines a local crop area, loads channel data, and populates the z1 array
        with the processed tile data. Timing information is printed if verbose is True.
    """
    if verbose:
        print("I just started with tile nr.:" + str(nx * ny_tiles + ny))
        start_time = time.time()
    # To avoid lost connection reconnect every time before downloading
    with refresh_omero_session(None, user, pw) as conn:
        image = get_image(conn, imageId)
        pixels = get_pixels(conn, image)

        # Define local crop area
        tile_coordinates = get_tile_coordinates(image, nx, ny, evaluated_crop_size, maximum_crop_size)

        # display current crop for potential debugging
        if verbose:
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
    if verbose:
        stop_time = time.time()
        report = "Processing of this tile took: %s" % (stop_time - start_time)
        os.system("echo \"%s\"" % report)