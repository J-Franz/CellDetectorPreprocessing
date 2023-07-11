from torch.utils.data import Dataset
import numpy as np
import os
'''import omero
from omero_tools import refresh_omero_session'''

class Image101Dataloader_1(Dataset):
    def __init__(self, path, training_image_ids, credentials, local = False,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.path = path
        labeled_data_list = []
        image_id_list_non_flat = []
        self.all_images_list = []
        self.training_image_ids = []
        self.Cellpose_results = []
        for image_id in training_image_ids:
            label_list_file = str(image_id) + "labels_list.npy"
            image_id_list_file = str(image_id) + "image_id_list.npy"
            if local == True:
                try:
                    data_cellpose = np.array(np.loadtxt(path+str(image_id) + "_CellposeAllNucleiCentroids.txt", delimiter=","))
                    self.Cellpose_results.append([np.nanmax(data_cellpose,0)[2],np.nanmedian(data_cellpose,0)[4]])
                    for fname in os.listdir(path):
                        if fname[0:13+len(str(image_id))] ==("%d_full_dataset"%image_id):
                            self.all_images_list.append(np.load(path+fname))
                            self.training_image_ids.append(image_id)
                        if (fname[0:len(label_list_file)-4]==label_list_file[:-4]):
                          labeled_data_list.append(np.load(path+fname))
                        if (fname[0:len(image_id_list_file)-4]==image_id_list_file[:-4]):
                          image_id_list_non_flat.append(np.load(path+fname))

                except:
                    print("Couldn't find data of image: "+ str(image_id))
                    continue
            else:
                print("Non Local Data Loading not yet supported.")
        self.label_list = np.array([])
        for data in labeled_data_list:
            self.label_list = np.concatenate((self.label_list, data), 0)
        self.image_id_list = np.array([[], [],[]])
        self.image_id_list = self.image_id_list.T
        for data in image_id_list_non_flat:
            self.image_id_list = np.concatenate((self.image_id_list, data), 0)
        self.transform = transform
        self.width = 101
        self.height = 101
        self.half_width = int(self.width / 2.)
        self.half_height = int(self.height / 2.)
        self.c_max = 4
        self.local = local
        if local==False:
            #TODO implement :-)
            self.conn = refresh_omero_session(None,self.user,self.pw)
            self.user = credentials["user"]
            self.pw = credentials["pw"]

    def __len__(self):
        return len(self.image_id_list)

    def __getitem__(self, idx):
        image_id = self.image_id_list[idx][0]
        x_pos = self.image_id_list[idx][1]
        y_pos = self.image_id_list[idx][2]
        if not self.local:
            # TODO: Rework for loading from image list!!
            self.conn.close()
            self.conn = refresh_omero_session(self.conn, self.user, self.pw)
            self.conn.SERVICE_OPTS.setOmeroGroup('-1')
            image = self.conn.getObject("Image", image_id)
            group_id = image.getDetails().getGroup().getId()
            self.conn.setGroupForSession(group_id)
            pixels = image.getPrimaryPixels()
            X = self.load_data_omero(x_pos,y_pos,pixels)
        else:
            image = np.zeros((self.c_max,self.width, self.height))
            list_id = np.squeeze(np.argwhere(image_id == self.training_image_ids)[0])
            first_in_image_id_list = next(x_id for x_id,x in enumerate(self.image_id_list) if image_id==x[0])
            try:
                for c in range(0, self.c_max):
                    image[c,:,:] = self.all_images_list[list_id][idx-first_in_image_id_list,:,:,c]
            except:
                print(idx)
            #image[1,:,:] = rescale_intensity(image[1,:,:], (0, 400), (-1, 1))
            #image[0,:,:] = rescale_intensity(image[0,:,:],(0,1000), (-1, 1))
            #image[3,:,:] = rescale_intensity(image[3,:,:],(0,1000),  (-1, 1))
            #image[2,:,:] = rescale_intensity(image[2,:,:],(0,1000),  (-1, 1))
            #print(np.nanstd(image[1,:,:]))
            image[1, :, :] = (image[1,:,:] -self.Cellpose_results[list_id][0])/max(self.Cellpose_results[list_id][1],0.0001)

            image[0, :, :] = (image[0,:,:] -np.nanmean(image[0,:,:] ))/np.nanstd(image[0,:,:] )
            image[2, :, :] = (image[2,:,:] -np.nanmean(image[2,:,:] ))/np.nanstd(image[2,:,:] )
            image[3, :, :] = (image[3,:,:] -np.nanmean(image[3,:,:] ))/np.nanstd(image[3,:,:] )
            X = image
        Y = self.label_list[idx]

        return [X,Y]

    def load_data_omero(self, x, y, pixels, local = False):
        image = np.zeros((self.c_max,self.width, self.height))
        for c in range(0, self.c_max):
            image[c,:, :] = pixels.getTile(0, theC=c, theT=0, tile=[x - self.half_width,
                                                                    y - self.half_height,
                                                                    self.width, self.height])
        return image
    def close(self):
        self.conn.close()