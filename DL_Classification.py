import numpy as np
import os
import sys
import time

# Torch Classes
import torch
import torch.utils.data as TData

## Optimizer Classes
from torch.optim import Adam
from torch.optim import SGD

# Loss function classes
from torch.nn import BCEWithLogitsLoss
from torch.nn import BCELoss
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss

## not used classes
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt

## Own Classes
from model_CNN_1 import model_CNN_1
from model_CNN_2 import model_CNN_2
from Image101Dataloader_1 import Image101Dataloader_1


credentials = {}
credentials["user"] = "Franz"
credentials["pw"]  ="ome"
image_id_list = []
local = True
print(sys.argv)
hpc = (sys.argv[1]=="True")


## Define where to look for images and load
if not hpc:
    path = "/share/Work/Neuropathologie/MicrogliaDetection/MihaelaTrainingsData/"
else:
    path="/usr/users/jfranz/MicrogliaDepletion/"
    try:
        os.mkdir(path)
    except:
        print("Results are saved to existing directory: " + path)

for file in os.listdir(path):
    if (file[-3:]=="npy")and (file.split("_")[1]=="training"):
        image_id = file.split("_")[0]
        image_id_list.append(int(image_id))

Test_Dataset = Image101Dataloader_1(path, image_id_list,credentials, local)

batch_size = 8
proportion_split= 0.6
len_train = int(proportion_split*len(Test_Dataset))
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# define the optimization
model = model_CNN_2(batch_size,4)
model.to(device)
criterion = BCELoss() #MSELoss()#
optimizer = SGD(model.parameters(), lr=0.001,momentum=0.9)



for epoch in range(20):
    if epoch>5:
        optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.9)
    if epoch>100:
        optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9)
    running_loss = 0.0
    start_time = time.time()
    for i, data in enumerate(train_dl):

        inputs, labels = data[0].to(device), data[1].to(device)

        #plt.imshow(inputs.numpy()[0,1,:,:])
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
    os.system("echo \"current running loss %d\""%running_loss)
    fname = "model_after_epoch_"+str(epoch)
    #torch.save(model.state_dict(),path+fname)



    correct = 0
    total = 0
    positive= 0
    positiv_preditions = 0
    positive_correct = 0
    validation_loss = 0
    for data in test_dl:
        images, labels = data[0].to(device), data[1].to(device)
        labels = labels#-0.5
        outputs = model(images.float())
        loss = criterion(outputs,labels.float())
        outputs = (outputs>0.1)*1.#outputs.round()
        total += labels.size(0)
        positive += (labels).sum().item()
        correct += (outputs == labels).sum().item()
        positiv_preditions += outputs.round().sum().item()
        positive_correct += (labels[outputs == labels]>0).sum().item()
        validation_loss += loss.item() * images.size(0)

    os.system("echo \"Total samples: %d\"" %total)
    os.system("echo \"Positive samples: %d\"" %positive)
    os.system("echo \"Correct: %d\"" %correct)
    os.system("echo \"Positive Correct: %d\"" %positive_correct)
    os.system("echo \"Accuracy: %f\"" %(correct/total))
    os.system("echo \"Total number - positive samples: %d\"" %(total-positive))
    os.system("echo \"Total number positive predicted: %d\"" %(positiv_preditions))

    stop_time = time.time()
    os.system("echo \"One Epoch took %d\"" %(stop_time-start_time))

correct = 0
total = 0
positive= 0
positiv_preditions = 0
positive_correct = 0
validation_loss = 0
False_negative_images = []
False_positive_images = []
for data in test_dl:
    images, labels = data[0].to(device), data[1].to(device)
    labels = labels#-0.5
    outputs = model(images.float())
    loss = criterion(outputs,labels.float())
    outputs = outputs.round()
    total += labels.size(0)
    positive += (labels).sum().item()
    correct += (outputs == labels).sum().item()
    if (outputs<labels).detach().cpu().numpy().sum()>0:
        False_negative_images.append(images[outputs<labels].detach().cpu().numpy())
    if (outputs>labels).detach().cpu().numpy().sum()>0:
        False_positive_images.append(images[outputs>labels].detach().cpu().numpy())
    positiv_preditions += outputs.round().sum().item()
    positive_correct += (labels[outputs == labels]>0).sum().item()
    validation_loss += loss.item() * images.size(0)
print(validation_loss)
print("Ratio Validation over Running Loss: %f" %((validation_loss/(1-proportion_split))/(running_loss/proportion_split)))
print("Total samples: %d" %total)
print("Positive samples: %d" %positive)
print("Correct: %d" %correct)
print("Positive Correct: %d" %positive_correct)
print("Accuracy: %f" %(correct/total))
print("Total number - positive samples: %d" %(total-positive))
print("Total number positive predicted: %d" %(positiv_preditions))

os.system("echo \"Total samples: %d\"" %total)
os.system("echo \"Positive samples: %d\"" %positive)
os.system("echo \"Correct: %d\"" %correct)
os.system("echo \"Positive Correct: %d\"" %positive_correct)
os.system("echo \"Accuracy: %f\"" %(correct/total))
os.system("echo \"Total number - positive samples: %d\"" %(total-positive))
os.system("echo \"Total number positive predicted: %d\"" %(positiv_preditions))

stop_time = time.time()
os.system("echo \"One Epoch took %d\"" %(stop_time-start_time))
False_negative_images = np.array(False_negative_images, dtype=object)

False_positive_images = np.array(False_positive_images, dtype=object)

np.save(path+"False_negative_images_2.npy",False_negative_images)
np.save(path+"False_positive_images_2.npy",False_positive_images)