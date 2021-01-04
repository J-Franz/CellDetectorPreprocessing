from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch
import torch.utils.data as TData
import torchvision.models as models
import numpy as np
from omero_tools import refresh_omero_session
import os
import omero
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt

class Image101Dataset(Dataset):
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
        self.training_image_ids = training_image_ids
        for image_id in training_image_ids:
            label_list_file = str(image_id) + "labels_list.npy"
            image_id_list_file = str(image_id) + "image_id_list.npy"
            if local == True:
                try:
                    for fname in os.listdir(path):
                        if fname[0:13+len(str(image_id))] ==("%d_full_dataset"%image_id):
                          self.all_images_list.append(np.load(path+fname))
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
        self.user = credentials["user"]
        self.pw = credentials["pw"]
        self.conn = refresh_omero_session(None,self.user,self.pw)
        self.local = local

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
            for c in range(0, self.c_max):
                image[c,:,:] = self.all_images_list[list_id][idx-first_in_image_id_list,:,:,c]
            #image[1,:,:] = rescale_intensity(image[1,:,:], (0, 400), (-1, 1))
            #image[0,:,:] = rescale_intensity(image[0,:,:],(0,1000), (-1, 1))
            #image[3,:,:] = rescale_intensity(image[3,:,:],(0,1000),  (-1, 1))
            #image[2,:,:] = rescale_intensity(image[2,:,:],(0,1000),  (-1, 1))
            #print(np.nanstd(image[1,:,:]))
            image[1, :, :] = (image[1,:,:] -np.nanmean(image[1,:,:] ))/max(np.nanstd(image[1,:,:] ),0.0001)

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

from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

from torch.optim import Adam
from torch.optim import SGD
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss

# model definition
class CNN(Module):
    # define model elements
    def __init__(self, n_channels):
        super(CNN, self).__init__()
        # input to first hidden layer
        self.hidden1 = Conv2d(n_channels, 4, (16, 16))
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # first pooling layer
        self.pool1 = MaxPool2d((2, 2), stride=(2, 2))
        # second hidden layer
        self.hidden2 = Conv2d(4, 4, (3, 3))
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # second pooling layer
        self.pool2 = MaxPool2d((2, 2), stride=(2, 2))
        # fully connected layer
        self.hidden3 = Linear(8000, 100)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        # output layer
        self.hidden4 = Linear(100, batch_size)
        xavier_uniform_(self.hidden4.weight)
        self.act4 = Softmax(dim=1)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.pool1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.pool2(X)
        # flatten
        X = X.view(-1, 8000)
        # third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        # output layer
        X = self.hidden4(X)
        X = self.act4(X)
        X = X.view(-1)
        return X

from torchvision import transforms
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5,0.5), (0.5, 0.5, 0.5,0.5))])

credentials = {}
credentials["user"] = "Franz"
credentials["pw"]  ="ome"
image_id_list = []
local = True

if local:
    path = "/share/Work/Neuropathologie/MicrogliaDetection/MihaelaTrainingsData/"
else:
    path="/scratch/jfranz/Analysis/Cellpose/MicrogliaDepletion/"
    try:
        os.mkdir(path)
    except:
        print("Results are saved to existing directory: " + path)

for file in os.listdir(path):
    if (file[-3:]=="npy")and (file.split("_")[1]=="training"):
        image_id = file.split("_")[0]
        image_id_list.append(int(image_id))
## TODO: get Trainingsdata from several images
Test_Dataset = Image101Dataset(path, image_id_list,credentials, local)

batch_size = 5
len_train = int(0.6*len(Test_Dataset))
len_train = len_train-len_train%batch_size
len_test = len(Test_Dataset)-len_train

class_sample_count = np.array(
    [len(np.where(Test_Dataset.label_list[0:len_train] == t)[0]) for t in np.unique(Test_Dataset.label_list[0:len_train])])
weight = 1. / class_sample_count
print(weight)
samples_weight = np.array([weight[int(t)] for t in Test_Dataset.label_list[0:len_train]])
samples_weight = torch.from_numpy(samples_weight)
samples_weigth = samples_weight.double()
sampler = TData.WeightedRandomSampler(samples_weight, len_train)

train,test = TData.random_split(Test_Dataset,[len_train,len_test])

train_dl = TData.DataLoader(train, batch_size=batch_size, sampler=sampler)
test_dl = TData.DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
print(len(train))
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

from torch import nn
from torch.optim import lr_scheduler
# define the optimization
model = CNN(4)
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


criterion = BCEWithLogitsLoss() #MSELoss()#
optimizer = SGD(model.parameters(), lr=0.001,momentum=0.9)

for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(train_dl):

        inputs, labels = data

        plt.imshow(inputs.numpy()[0,1,:,:])
        labels = labels#-0.5
        optimizer.zero_grad()
        # compute the model output
        #print(labels)
        yhat = model(inputs.float())
        # calculate loss
        loss = criterion(yhat,labels.float())
        # credit assignment
        loss.backward()
        # update model weights
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    print(running_loss)



correct = 0
total = 0
positive= 0
positiv_preditions = 0
positive_correct = 0
for data in test_dl:
    images, labels = data
    labels = labels#-0.5
    outputs = model(images.float())
    outputs = outputs.round()
    total += labels.size(0)
    positive += (labels).sum().item()
    correct += (outputs == labels).sum().item()
    positiv_preditions += outputs.sum().item()
    positive_correct += (labels[outputs == labels]>0).sum().item()

print("Total samples: %d" %total)
print("Positive samples: %d" %positive)
print("Correct: %d" %correct)
print("Positive Correct: %d" %positive_correct)
print("Accuracy: %f" %(correct/total))
print("Total number - positive samples: %d" %(total-positive))
print("Total number positive predicted: %d" %(positiv_preditions))
Test_Dataset.close()