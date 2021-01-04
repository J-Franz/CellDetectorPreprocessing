import sys
import os
import time
from scipy.stats import skew
# This is to connect to omero
from omero_tools import refresh_omero_session, get_pixels, get_image, get_tile_coordinates, UploadArrayAsTxtToOmero
import numpy as np

# This is the actual application of cellpose
from image_tools import get_coordinates, delete_cells_at_border
from utils import define_path


def main():
    ## Get credentials from first argumentma
    # run like main.py password
    user = "Franz"
    pw = sys.argv[1]
    imageId = sys.argv[2]
    local = (sys.argv[3] == "True")
    if local:
        use_gpu = False
        plot_cellpose = False
    else:
        use_gpu = True  # requires cellpose to be installed in gpu usable fashion
        plot_cellpose = False  # should be false for cluster
    try:
        c_fluorescence = int(sys.argv[4])
    except IndexError:
        c_fluorescence = None
    try:
        c_dapi = int(sys.argv[5])
    except IndexError:
        c_dapi = None
    # establish connection
    # Load image for the first time

    with refresh_omero_session(None, user, pw) as conn:
        conn.getGroupFromContext()
        conn.SERVICE_OPTS.setOmeroGroup('-1')
        image = conn.getObject("Image", imageId)
        group_id = image.getDetails().getGroup().getId()
        group_name = image.getDetails().getGroup().getName()
        max_c = image.getSizeC()-1 #counting starts from 0
        print("Switched Group to Group of image: ", group_id, " with name: ", group_name)

        # Set up plan to iterate over whole slide
        # TODO: make dict, save in h5 as metadata

        maximum_crop_size = 1000
        overlap = 100
        evaluated_crop_size = maximum_crop_size - overlap
        nx_tiles = int(image.getSizeX() / evaluated_crop_size) + 1
        ny_tiles = int(image.getSizeY() / evaluated_crop_size) + 1
        n_runs = ny_tiles * nx_tiles
        width = 101
        height: int = 101
        half_width = int(width / 2.)
        half_height = int(height / 2.)
        if c_dapi is None:
            c_dapi=0
        # TODO: get c_fluorescence
        if c_fluorescence is None:
            c_fluorescence = max_c

    print("We will start to crop the image in " + str(n_runs) + " tiles and start the analysis now.")

    os.system("echo \"We will start to crop the image in " + str(n_runs) + " tiles and start the analysis now.\"")

    path = define_path(local)

    os.system("echo \"We will store the image in " + path + " tiles and start the analysis now.\"")

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
                image = get_image(conn, user, pw, imageId)
                pixels = get_pixels(conn, image)

                # Define local crop area
                tile_coordinates = get_tile_coordinates(image, nx, ny, evaluated_crop_size, maximum_crop_size)

                # display current crop for potential debugging
                print("The current crop is crop_width,crop_height,current_crop_x,current_crop_y:",
                      tile_coordinates)

                # load Dapi Channel and Fluorescence channel
                # getTile switches width and height...
                # this is compensated by get_coordinates to return switched coordinates
                tile_dapi = pixels.getTile(0, theC=c_dapi, theT=0, tile=tile_coordinates.values())
                tile_fluorescence = pixels.getTile(0, theC=c_fluorescence, theT=0, tile=tile_coordinates.values())

            # Apply cellpose to extract coordinates, potentially one could also save masks and boundaries
            [list_nuclei_x_coords, list_nuclei_y_coords] = get_coordinates(tile_dapi, plot_cellpose, use_gpu)

            # cast list to numpy
            x_coords_nuclei = np.array(list_nuclei_x_coords)
            y_coords_nuclei = np.array(list_nuclei_y_coords)

            # take only nuclei within border of overlap/2+1
            # also take into account that cropped regions might be smaller than maximum_crop_size

            x_coords_nuclei, y_coords_nuclei = delete_cells_at_border(x_coords_nuclei, y_coords_nuclei, overlap, tile_coordinates)

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

    fname = str(imageId) + "_CellposeAllNucleiCentroidsV2_c" + str(c_fluorescence) + ".txt"
    verbose_upload = True
    UploadArrayAsTxtToOmero(path + fname, concat_array.T, group_name, imageId, pw, user, verbose_upload)


if __name__ == '__main__':
    main()
