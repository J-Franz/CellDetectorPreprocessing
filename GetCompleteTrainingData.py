import numpy as np
from omero_tools import refresh_omero_session
import sys
import os
import omero
from tqdm import tqdm

user = "Franz"
pw = sys.argv[1]
local = (sys.argv[2]=="True")


os.system("echo %s" %local)


if local:
    path = "/share/Work/Neuropathologie/MicrogliaDetection/MihaelaTrainingsData/"
else:
    path="/scratch/users/jfranz/Analysis/Cellpose/MicrogliaDepletion/MihaelaTrainingsData/"
    try:
        os.mkdir(path)
    except:
        print("Results are saved to existing directory: " + path)


fname_list = os.listdir(path)
for fname in fname_list:
    if (fname[-3:]=="npy") and (fname.split("_")[1]=="training"):

        imageId = fname.split("_")[0]
        threshold = fname.split("_")[4][0:3]
        fname = str(imageId)+"_training_results_Threshold_"+str(threshold)+".npy"
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

        results = np.load(path+fname)


        conn.close()
        conn = refresh_omero_session(conn,user,pw)
        conn.SERVICE_OPTS.setOmeroGroup('-1')
        image = conn.getObject("Image", imageId)
        group_id = image.getDetails().getGroup().getId()
        conn.setGroupForSession(group_id)
        pixels = image.getPrimaryPixels()

        print(fname)
        labeled_data = np.isnan(results[1])==False
        centroid_id_to_load = results[0][labeled_data]
        labeled_data_list = results[1][labeled_data]

        for ann in image.listAnnotations():
            if isinstance(ann, omero.gateway.FileAnnotationWrapper):
                if ann.getFile().getName() == str(imageId)+"_CellposeAllNucleiCentroids.txt":
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

        all_images = np.zeros((len(centroid_id_to_load),width,height,c_max))
        image_id_list = np.ones((len(centroid_id_to_load),3))*np.nan
        print("We have to load " + str(len(centroid_id_to_load))+ " images.")
        if local:
            for id_n, id_ in tqdm(enumerate(centroid_id_to_load)):
                x = x_centroids[int(id_)]
                y = y_centroids[int(id_)]
                image = load_data(x,y,pixels)
                all_images[id_n,:,:,:] = image
                image_id_list[id_n,0]  = imageId
                image_id_list[id_n,1]  = x
                image_id_list[id_n,2]  = y
        else:
            for id_n, id_ in enumerate(centroid_id_to_load):
                x = x_centroids[int(id_)]
                y = y_centroids[int(id_)]
                image = load_data(x,y,pixels)
                all_images[id_n,:,:,:] = image
                image_id_list[id_n,0]  = imageId
                image_id_list[id_n,1]  = x
                image_id_list[id_n,2]  = y


        fname = str(imageId)+"labels_list_threshold_"+str(threshold)+".npy"
        np.save(path+fname, labeled_data_list)

        fname = str(imageId)+"image_id_list_threshold_"+str(threshold)+".npy"
        np.save(path+fname, image_id_list)

        fname = str(imageId)+"_full_dataset_threshold_"+str(threshold)+".npy"
        np.save(path+fname, all_images)
        print("We just saved " + str(len(centroid_id_to_load))+ " images as a trainings image set under "+ path+fname)
        conn.close()
    #fname = str(imageId)+"_TrainingData.txt"