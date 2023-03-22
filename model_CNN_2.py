from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.nn import BatchNorm2d
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_


# model definition
class model_CNN_2(Module):
    # define model elements
    def __init__(self, batchsize, n_channels):
        super(model_CNN_2, self).__init__()
        self.batchsize =batchsize
        self.n_channels = n_channels
        # input to first hidden layer
        self.hidden1 = Conv2d(n_channels, 4, (3, 3))
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
        self.hidden3 = BatchNorm2d(4)

        self.hidden4 = Linear(int(batchsize*2116), 1000)
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        self.act4 = ReLU()
        # fully connected layer
        self.hidden5 = Linear(1000, batchsize)
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        self.act5 = Sigmoid()

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
        # third hidden layer
        X = self.hidden3(X)
        X = X.view(-1, 2116*self.batchsize)

        X = self.hidden4(X)
        X = self.act4(X)
        # third hidden layer
        X = self.hidden5(X)
        X = self.act5(X)
        X = X.view(-1)
        return X
