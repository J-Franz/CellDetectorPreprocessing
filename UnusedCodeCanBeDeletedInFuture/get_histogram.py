import sys
import os
import time
## This is to connect to omero
from JonasTools.omero_tools import refresh_omero_session
import numpy as np
import omero
from shapely import geometry
from shapely import affinity
from scipy.ndimage import median_filter
from JonasTools.shapley_tools import analyse_polygon_histogram,get_polygon_as_shape


user = "Franz"
pw = sys.argv[1]
imageId = sys.argv[2]
local = (sys.argv[3] == "True")
##establish connection
conn = refresh_omero_session(None, user, pw)
conn.getGroupFromContext()

## Load image for the first time
try:
    conn.SERVICE_OPTS.setOmeroGroup('-1')
    image = conn.getObject("Image", imageId)
    group_id = image.getDetails().getGroup().getId()
    group_name = image.getDetails().getGroup().getName()
    conn.setGroupForSession(group_id)
    print("Switched Group to Group of image: ", group_id, " with name: ", group_name)
except:
    print("Unable to load image for the first time.")
    sys.exit(1)

if local:
    path = ""
else:
    path = "/scratch/users/jfranz/Analysis/Histogram/MicrogliaDepletion/"
    try:
        os.mkdir(path)
    except:
        print("Results are saved to existing directory: " + path)

roi_service = conn.getRoiService()



## Screen all tissue for tissue_XXX and hole_XXX
tissue_list = []
hole_list = []

result = roi_service.findByImage(image.getId(), None)
for roi in result.rois:
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

evaluated_crop_size = 2000
# Set up plan to iterate over whole slide
for ptissue_id,ptissue in enumerate(polygon_tissue_list):
    # exclude holes first
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
    c_fluorescence = 3

    print("We will start to crop the image in " + str(n_runs) + " tiles and start the analysis now.")

    os.system("echo \"We will start to crop the image in " + str(n_runs) + " tiles and start the analysis now.\"")


    os.system("echo \"We will store the image in " + path + " tiles and start the analysis now.\"")

    list_all_cdfs = []
    list_analysed_area = []

    n_run = 0
    for nx in range(nx_tiles):#[nx_tiles-1]:#
        for ny in range(ny_tiles):#[ny_tiles-2]:#
            print("I just started with tile nr.:" + str(nx * ny_tiles + ny))
            start_time = time.time()
            # To avoid lost connection reconnect every time before downloading
            conn.close()
            conn = refresh_omero_session(conn, user, pw)
            conn.SERVICE_OPTS.setOmeroGroup('-1')
            image = conn.getObject("Image", imageId)
            group_id = image.getDetails().getGroup().getId()
            conn.setGroupForSession(group_id)
            pixels = image.getPrimaryPixels()

            # Define local crop area
            current_crop_x = nx * evaluated_crop_size + min_x
            current_crop_y = ny * evaluated_crop_size + min_y
            ## Continue hiere!
            crop_width = min(evaluated_crop_size, int(max_x - current_crop_x))
            crop_height = min(evaluated_crop_size, int(max_y - current_crop_y))

            # display current crop for potential debugging
            print("The current crop is crop_width,crop_height,current_crop_x,current_crop_y:",
                  str(crop_width), str(crop_height), str(current_crop_x), str(current_crop_y))
            # load Dapi Channel and Fluorescence channel
            # getTile switches width and height...
            # this is compensated by get_coordinates to return switched coordinates
            cropped_box = geometry.box(current_crop_x, current_crop_y, current_crop_x + crop_width,
                                       current_crop_y + crop_height)
            analyse_polygon = ptissue.intersection(cropped_box)

            # TODO implement support of Multi Polygons
            if analyse_polygon.area>0:

                tile_fluorescence = pixels.getTile(0, theC=c_fluorescence, theT=0,
                                                   tile=[current_crop_x, current_crop_y, crop_width, crop_height])

                # Apply median filter to delete pixel errors
                median_filtered_tile = median_filter(tile_fluorescence, (3, 3))
                # First shift polygon to current frame with x=0, and y=0
                analyse_polygon = affinity.translate(analyse_polygon, -current_crop_x, -current_crop_y)

                if analyse_polygon.geom_type=='MultiPolygon':
                    for polygon in analyse_polygon:
                        # save area of local polygon
                        list_analysed_area.append(polygon.area)
                        local_ecdf = analyse_polygon_histogram(polygon,median_filtered_tile)
                        list_all_cdfs.append(local_ecdf)

                else:

                    # save area of local polygon
                    list_analysed_area.append(analyse_polygon.area)
                    local_ecdf = analyse_polygon_histogram(analyse_polygon,median_filtered_tile)
                    list_all_cdfs.append(local_ecdf)


                stop_time = time.time()
                report = "Processing of this tile took: %s" %(stop_time-start_time)
                os.system("echo \"%s\"" %report)
    x = np.arange(image.getPixelRange()[1])
    all_ecdfs = np.ones((len(x),len(list_all_cdfs)))
    for ecdf_id,ecdf in enumerate(list_all_cdfs):
        all_ecdfs[:,ecdf_id]= ecdf(x)
    weighted_areas =np.array(list_analysed_area)
    ECDF = np.sum(all_ecdfs * weighted_areas, 1) / np.sum(all_ecdfs * weighted_areas, 1).max()
    fname = str(imageId) + "_CDF_c" + str(c_fluorescence) + "_tissue_"+str(ptissue_id)+"V1.txt"
    np.savetxt(path + fname, ECDF, delimiter=',', fmt='%f')
    ## Refresh connection and get group name of image
    conn.close()
    conn = refresh_omero_session(conn, user, pw)
    conn.SERVICE_OPTS.setOmeroGroup('-1')
    image = conn.getObject("Image", imageId)
    group_name = image.getDetails().getGroup().getName()

    # Upload file via omero client in bash system steered by python to the omero server and link to the image

    login_command = "omero login " + user + "@134.76.18.202 -w " + pw + " -g \"" + group_name + "\""

    stream = os.popen(login_command)
    output = stream.read()
    command = "omero upload " + path + fname
    stream = os.popen(command)
    output = stream.read()
    print(output)
    command = "omero obj new FileAnnotation file=" + output
    stream = os.popen(command)
    output = stream.read()
    print(output)
    command = "omero obj new ImageAnnotationLink parent=" + "Image:" + str(imageId) + " child=" + output
    stream = os.popen(command)
    output = stream.read()
    print(output)

    stream.close()
    conn.close()
