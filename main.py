import sys
import os
import time
from scipy.stats import skew
## This is to connect to omero
from omero_tools import refresh_omero_session
import numpy as np

## This is the actual application of cellpose
from image_tools import get_coordinates

## Get credentials from first argument
#run like main.py password

user = "Franz"
pw = sys.argv[1]
imageId = sys.argv[2]
local = sys.argv[3]

##establish connection
conn = refresh_omero_session(None,user,pw)
conn.getGroupFromContext()

## Load image for the first time
try:
    conn.SERVICE_OPTS.setOmeroGroup('-1')
    image = conn.getObject("Image", imageId)
    group_id = image.getDetails().getGroup().getId()
    group_name = image.getDetails().getGroup().getName()
    print("Switched Group to Group of image: ", group_id, " with name: ", group_name)
except:
    print("Unable to load image for the first time.")
    sys.exit(1)

# Set up plan to iterate over whole slide
maximum_crop_size = 2000
overlap = 100
evaluated_crop_size = maximum_crop_size-overlap
nx_tiles =int(image.getSizeX()/evaluated_crop_size)+1
ny_tiles =int(image.getSizeY()/evaluated_crop_size)+1
n_runs = ny_tiles*nx_tiles
width=101
height: int=101
half_width=int(width/2.)
half_height=int(height/2.)
c_dapi = 0
c_fluorescence = 1

print("We will start to crop the image in " + str(n_runs)+ " tiles and start the analysis now.")

list_all_x = []
list_all_y = []
list_all_mean = []
list_all_median= []
list_all_var = []
list_all_skew = []

n_run = 0
for nx in range(nx_tiles):
    for ny in range(ny_tiles):
        print("I just started with tile nr.:"+ str(nx*ny_tiles+ny))
        start_time = time.time()
        # To avoid lost connection reconnect every time before downloading
        conn.close()
        conn = refresh_omero_session(conn,user,pw)
        conn.SERVICE_OPTS.setOmeroGroup('-1')
        image = conn.getObject("Image", imageId)
        group_id = image.getDetails().getGroup().getId()
        conn.setGroupForSession(group_id)
        pixels = image.getPrimaryPixels()

        # Define local crop area
        current_crop_x = nx*evaluated_crop_size
        current_crop_y = ny*evaluated_crop_size
        crop_width= min(maximum_crop_size,int(image.getSizeX()-current_crop_x))
        crop_height=min(maximum_crop_size,int(image.getSizeY()-current_crop_y))

        #display current crop for potential debugging
        print("The current crop is crop_width,crop_height,current_crop_x,current_crop_y:",
              str(crop_width),str(crop_height),str(current_crop_x),str(current_crop_y))

        #load Dapi Channel and Fluorescence channel
        tile_dapi = pixels.getTile(0, theC=c_dapi, theT=0,
                                   tile=[current_crop_x, current_crop_y, crop_width, crop_height])
        tile_fluorescence = pixels.getTile(0, theC=c_fluorescence, theT=0,
                                           tile=[current_crop_x, current_crop_y, crop_width, crop_height])

        # Apply cellpose to extract coordinates, potentially one could also save masks and boundaries
        [list_nuclei_x_coords, list_nuclei_y_coords] = get_coordinates(tile_dapi, False)

        # cast list to numpy
        x_coords_nuclei = np.array(list_nuclei_x_coords)
        y_coords_nuclei = np.array(list_nuclei_y_coords)

        # take only nuclei within border of overlap/2
        # also take into account that cropped regions might be smaller than maximum_crop_size

        too_close_to_x0 = x_coords_nuclei<overlap/2.
        too_close_to_xmax = (x_coords_nuclei>maximum_crop_size-overlap/2.)+\
                            (x_coords_nuclei>image.getSizeX()-overlap/2.)

        too_close_to_y0 = y_coords_nuclei < overlap / 2.
        too_close_to_ymax = (y_coords_nuclei > maximum_crop_size - overlap / 2.) +\
                            (y_coords_nuclei > image.getSizeY() - overlap / 2.)

        x_coords_nuclei = np.delete(x_coords_nuclei,np.argwhere(too_close_to_x0+too_close_to_xmax+too_close_to_y0+too_close_to_ymax))
        y_coords_nuclei = np.delete(y_coords_nuclei, np.argwhere(too_close_to_x0 + too_close_to_xmax + too_close_to_y0 + too_close_to_ymax))


        tmp_array = np.zeros((width, height))
        for id_x,x in enumerate(x_coords_nuclei):
            temp_crop_x =(x - half_width)
            temp_crop_y =(y_coords_nuclei[id_x] - half_height)
            tmp_array = tile_fluorescence[temp_crop_x:(temp_crop_x+width), temp_crop_y:(temp_crop_y+height)]
            list_all_mean.append(np.mean(tmp_array.flatten()))
            list_all_median.append(np.median(tmp_array.flatten()))
            list_all_var.append(np.var(tmp_array.flatten()))
            list_all_skew.append(skew(tmp_array.flatten()))

            # transform from local coordinates of tile to coordinates of wsi

        x_coords_nuclei = x_coords_nuclei + current_crop_x
        y_coords_nuclei = y_coords_nuclei + current_crop_y

        list_all_x.append(x_coords_nuclei)
        list_all_y.append(y_coords_nuclei)

        '''    
        tmp_array = np.zeros((width, height))
        tmp_array[:, :] = pixels.getTile(0, theC=1, theT=0,
                                             tile=[x - half_width, y_coords_nuclei[id_x] - half_height, width,
                                                   height])'''

        stop_time = time.time()
        print("Processing of this tile took: " +str(stop_time-start_time))
print(n_run)
all_x = np.concatenate(list_all_x)
all_y = np.concatenate(list_all_y)
all_mean = list_all_mean
all_median = list_all_median
all_var =  list_all_var
all_skew = list_all_skew

all = np.array((all_x,all_y, all_mean, all_median, all_var, all_skew),dtype=float)
if local:
    path = ""
else:
    path="/scratch/jfranz/Analysis/Cellpose/"+group_name+"/"
    try:
        os.mkdir(path)
    except:
        print("Results are saved to existing directory: " + path)

fname = "CellposeAllNucleiCentroids.txt"
np.savetxt(path + fname,all.T, delimiter=',',fmt='%f')

## Refresh connection and get group name of image
conn.close()
conn = refresh_omero_session(conn, user, pw)
conn.SERVICE_OPTS.setOmeroGroup('-1')
image = conn.getObject("Image", imageId)
group_name = image.getDetails().getGroup().getName()

# Upload file via omero client in bash system steered by python to the omero server and link to the image

login_command = "omero login "+user+"@134.76.18.202 -w "+pw+" -g \"" + group_name + "\""

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
command = "omero obj new ImageAnnotationLink parent="+"Image:"+str(imageId)+ " child=" + output
stream = os.popen(command)
output = stream.read()
print(output)



stream.close()
conn.close()


