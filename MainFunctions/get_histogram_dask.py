import sys
import os
import time
## This is to connect to omero
from JonasTools.omero_tools import refresh_omero_session, get_image, UploadArrayAsTxtToOmero, check_fname_omero, \
    make_omero_file_available
import numpy as np
import omero
from shapely import geometry
from shapely import affinity
from scipy.ndimage import median_filter
from JonasTools.shapley_tools import analyse_polygon_histogram,get_polygon_as_shape
import dask.array as da

from Utils.utils import extract_system_arguments, unpack_parameters


def get_histogram_dask(sys_arguments, parameters):
    c_dapi, c_fluorescence, imageId, base, gpu, pw, user = extract_system_arguments(sys_arguments)
    evaluated_crop_size, half_height, half_width, height, maximum_crop_size, overlap, width = unpack_parameters(
            parameters)

    with refresh_omero_session(None, user, pw) as conn:
        image = get_image(conn, imageId)
        group_id = image.getDetails().getGroup().getId()
        group_name = image.getDetails().getGroup().getName()
        conn.setGroupForSession(group_id)

        size_x = image.getSizeX()
        size_y = image.getSizeY()
        size_c = image.getSizeC()
        max_c = size_c - 1  # counting starts from 0
        pixel_range = np.arange(image.getPixelRange()[1])
        os.system("echo \"%s pixel range\""%pixel_range)
        roi_service = conn.getRoiService()


        ## Screen all tissue for tissue_XXX and hole_XXX
        tissue_list = []
        hole_list = []

        rois_of_image = roi_service.findByImage(image.getId(), None)


    for roi in rois_of_image.rois:
        # print("ROI:  ID:", roi.getId().getValue())
        for s in roi.copyShapes():
            if type(s) in (
                    omero.model.LabelI, omero.model.PolygonI):
                if s.getTextValue():
                    text_value = s.getTextValue().getValue()
                    if text_value.find("hole_")!=-1:
                        hole_list.append([s])
                    if text_value.find("tissue_")!=-1:
                        tissue_list.append([s])


    polygon_tissue_list = get_polygon_as_shape(tissue_list)
    polygon_hole_list = get_polygon_as_shape(hole_list)

    # Define channels automatically if not specified
    if c_dapi is None:
        c_dapi = 0
    if c_fluorescence is None:
        c_fluorescence = max_c
    # set working directory
    zarr_path = "%sCelldetectorPreprocessed/Zarr/"%base
    CDF_path = "%sCelldetectorPreprocessed/CDF/"%base
    os.makedirs(CDF_path,exist_ok=True)
    fname = str(imageId) + "_image.zarr"

    zarr_image = da.from_zarr(zarr_path+fname)
    # Set up plan to iterate over slide
    evaluated_crop_size = 2000#maximum_crop_size  # to load non_overlapping

    for ptissue_id,ptissue in enumerate(polygon_tissue_list):
        # exclude holes first
        fname_upload = str(imageId) + "_CDF_c" + str(c_fluorescence) + "_tissue_"+str(ptissue_id)+"V1.txt"
        with refresh_omero_session(None, user, pw) as conn:
            image = get_image(conn, imageId)
            already_extracted = check_fname_omero(fname_upload, image)

            if already_extracted:
                make_omero_file_available(image, fname_upload, CDF_path)
                continue
        for hole in polygon_hole_list:
            ptissue=ptissue.difference(hole)
        if ptissue.geom_type=='MultiPolygon':
            print("ptissue is splitted to multi polygon. Not yet supported")
            sys.exit(1)
        min_y = int(min(ptissue.exterior.xy[1]))
        min_x = int(min(ptissue.exterior.xy[0]))
        max_y = int(max(ptissue.exterior.xy[1]))
        max_x = int(max(ptissue.exterior.xy[0]))
        nx_tiles = int((max_x-min_x) / evaluated_crop_size) + 1
        ny_tiles = int((max_y-min_y) / evaluated_crop_size) + 1
        n_runs = ny_tiles * nx_tiles
        print(min_y,max_y,min_x,max_x)

        print("We will start to crop the image in " + str(n_runs) + " tiles and start the analysis now.")

        os.system("echo \"We will start to crop the image in " + str(n_runs) + " tiles and start the analysis now.\"")


        os.system("echo \"We will store the CDF in " + CDF_path + " tiles and start the analysis now.\"")

        list_all_cdfs = []
        list_analysed_area = []

        n_run = 0
        for nx in range(nx_tiles):
            for ny in range(ny_tiles):
                print("I just started with tile nr.:" + str(nx * ny_tiles + ny))
                start_time = time.time()

                # Define local crop area
                current_crop_x = nx * evaluated_crop_size + min_x
                current_crop_y = ny * evaluated_crop_size + min_y
                ## Continue hiere!
                crop_width = min(evaluated_crop_size, int(max_x - current_crop_x))
                crop_height = min(evaluated_crop_size, int(max_y - current_crop_y))

                # display current crop for potential debugging
                #print("The current crop is crop_width,crop_height,current_crop_x,current_crop_y:",
                #      str(crop_width), str(crop_height), str(current_crop_x), str(current_crop_y))
                # load Dapi Channel and Fluorescence channel
                # getTile switches width and height...
                # this is compensated by get_coordinates to return switched coordinates
                cropped_box = geometry.box(current_crop_x, current_crop_y, current_crop_x + crop_width,
                                           current_crop_y + crop_height)
                analyse_polygon = ptissue.intersection(cropped_box)

                if analyse_polygon.area>0:
                    tile_fluorescence = zarr_image[current_crop_x:(current_crop_x+crop_width),
                                        current_crop_y:(current_crop_y+crop_height),c_fluorescence]
                    tile_fluorescence = tile_fluorescence.compute()

                    # Apply median filter to delete pixel errors
                    median_filtered_tile = median_filter(tile_fluorescence, (3, 3))
                    # First shift polygon to current frame with x=0, and y=0
                    analyse_polygon = affinity.translate(analyse_polygon, -current_crop_x, -current_crop_y)

                    if analyse_polygon.geom_type=='MultiPolygon':
                        print(analyse_polygon)
                        for polygon in analyse_polygon.geoms:
                            print(analyse_polygon)
                            try:
                                #os.system("echo \"Polygon area = %d\""%polygon.area)
                                local_ecdf = analyse_polygon_histogram(polygon,median_filtered_tile)
                                #os.system("echo \"Polygon area = %s\""%local_ecdf(pixel_range)[::50])
                                if local_ecdf is not None:
                                    list_all_cdfs.append(local_ecdf)
                                    # save area of local polygon
                                    list_analysed_area.append(polygon.area)
                                else:
                                    print("We couldn't extract coordinates in %d x and %d y"%(current_crop_x,current_crop_y))
                            except ZeroDivisionError:
                                continue

                    else:

                        #os.system("echo \"Polygon area = %d\""%analyse_polygon.area)
                        try:
                            local_ecdf = analyse_polygon_histogram(analyse_polygon,median_filtered_tile)

                            if local_ecdf is not None:
                                # save area of local polygon
                                list_analysed_area.append(analyse_polygon.area)
                                # os.system("echo \"Polygon area = %s\""%local_ecdf(pixel_range)[::50])
                                list_all_cdfs.append(local_ecdf)
                            else:
                                print("We couldn't extract coordinates in %d x and %d y"%(current_crop_x,current_crop_y))
                        except ZeroDivisionError:
                            continue



                    stop_time = time.time()
                    report = "Processing of this tile took: %s" %(stop_time-start_time)
                    #os.system("echo \"%s\"" %report)
                n_run +=1

        all_ecdfs = np.ones((len(pixel_range),len(list_all_cdfs)))

        # check why None sometimes in ecdf?
        for ecdf_id,ecdf in enumerate(list_all_cdfs):
            all_ecdfs[:,ecdf_id]= ecdf(pixel_range)

        weighted_areas =np.array(list_analysed_area)

        ECDF = np.sum(all_ecdfs * weighted_areas, 1) / np.sum(all_ecdfs * weighted_areas, 1).max()

        UploadArrayAsTxtToOmero(CDF_path + fname_upload, ECDF, group_name, imageId, pw, user)
        return 0
