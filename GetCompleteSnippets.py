import numpy as np
from omero_tools import refresh_omero_session
import sys
import os
import omero
from skimage import filters

user = "Franz"
pw = sys.argv[1]
local = (sys.argv[2]=="True")
imageId = sys.argv[3]
threshold_method = sys.argv[4]

os.system("echo %s" %local)


if local:
    path = "/share/Work/Neuropathologie/MicrogliaDetection/Snippets/"
    try:
        os.mkdir(path)
    except:
        print("Results are saved to existing directory: " + path)
else:
    path="/scratch/users/jfranz/Analysis/Cellpose/MicrogliaDepletion/Snippets/"
    try:
        os.mkdir(path)
    except:
        print("Results are saved to existing directory: " + path)


print(imageId)
## Load image for the first time
try:##establish connection
    conn = refresh_omero_session(None,user,pw)
    conn.SERVICE_OPTS.setOmeroGroup('-1')
    image = conn.getObject("Image", imageId)
    group_id = image.getDetails().getGroup().getId()
    group_name = image.getDetails().getGroup().getName()
    print("Switched Group to Group of image: ", group_id, " with name: ", group_name)
except:
    print("Unable to load image for the first time.")
    sys.exit(1)


conn = refresh_omero_session(conn,user,pw)
conn.SERVICE_OPTS.setOmeroGroup('-1')
image = conn.getObject("Image", imageId)
group_id = image.getDetails().getGroup().getId()
conn.setGroupForSession(group_id)
pixels = image.getPrimaryPixels()

for ann in image.listAnnotations():
    if isinstance(ann, omero.gateway.FileAnnotationWrapper):
        if ann.getFile().getName() == str(imageId)+"_CellposeAllNucleiCentroidsV2_c3.txt":
            print("File ID:", ann.getFile().getId(), ann.getFile().getName(), \
                "Size:", ann.getFile().getSize())
            file_path = os.path.join(path, ann.getFile().getName())

            with open(str(file_path), 'wb') as f:
                print("\nDownloading file to", file_path, "...")
                for chunk in ann.getFileInChunks():
                    f.write(chunk)
            print("File downloaded!")
cods = np.loadtxt(file_path,delimiter=",")
x_centroids = []
y_centroids= []
mean = []
median = []
var = []
skew = []
for p in cods:
    x_centroids.append(p[0])
    y_centroids.append(p[1])
    mean.append(p[2])
    median.append(p[3])
    var.append(p[4])
    skew.append(p[5])
x_centroids = np.array(x_centroids)
y_centroids = np.array(y_centroids)
mean = np.array(mean)
median = np.array(median)
var = np.array(var)
skew = np.array(skew)

width=101
height=101
half_width=int(width/2.)
half_height=int(height/2.)
c_max = 4


def load_data(x, y, pixels):
    width = 101
    height = 101
    half_width = int(width / 2.)
    half_height = int(height / 2.)
    c_max = 4
    image = np.zeros((width, height, c_max))
    for c in range(0, c_max):
        image[:, :, c] = pixels.getTile(0, theC=c, theT=0, tile=[x - half_width, y - half_height, width, height])
    return image


all_centroids = np.arange(len(x_centroids))

if threshold_method == "triangle":
    nan_median = np.isnan(median)
    # Images are assumed here not not depend on floating scale.
    threshold = int(filters.threshold_otsu(median[False==nan_median]))
    centroid_id_to_load = all_centroids[median>threshold]
    centroid_id_nan_median = all_centroids[nan_median]
else:
    print("No threshold method defined or no threshold method provided.")
    sys.exit(1)

all_images = np.zeros((len(centroid_id_to_load),width,height,c_max))
all_images_nan_median = np.zeros((len(centroid_id_nan_median),width,height,c_max))
image_id_list = np.ones((len(centroid_id_to_load),3))*np.nan
print("We have to load " + str(len(centroid_id_to_load))+ " images.")

for id_n, id_ in enumerate(centroid_id_to_load[0:20]):
    x = x_centroids[int(id_)]
    y = y_centroids[int(id_)]
    image = load_data(x,y,pixels)
    all_images[id_n,:,:,:] = image
    image_id_list[id_n,0]  = imageId
    image_id_list[id_n,1]  = x
    image_id_list[id_n,2]  = y


nan_median_x_coords = x_centroids[nan_median]
nan_median_y_coords = y_centroids[nan_median]

for id_n, id_ in enumerate(centroid_id_nan_median[0:20]):
    x = x_centroids[int(id_)]
    y = y_centroids[int(id_)]
    image = load_data(x,y,pixels)
    all_images_nan_median[id_n,:,:,:] = image


fname = str(imageId)+"_full_dataset_threshold_"+str(threshold)+".npy"
np.save(path+fname, all_images)
print("We just saved " + str(len(centroid_id_to_load))+ " images as a snipped image set under "+ path+fname)
fname = str(imageId)+"_nan_median.npy"
np.save(path+fname, all_images_nan_median)
fname = str(imageId)+"_images_with_nan_median.npy"
np.save(path+fname, np.array([nan_median_x_coords,nan_median_y_coords]))
print("We just saved a list of " + str(len(centroid_id_nan_median))+ " image ids with nan_median under "+ path+fname)
conn.close()
    #fname = str(imageId)+"_TrainingData.txt"