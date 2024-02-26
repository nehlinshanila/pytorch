# conda run -n myenv1 python neural_network.py


# imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader  # Fixed import statement
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# create fully connected network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)  # Fixed linear layer definition
        self.fc2 = nn.Linear(50, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = NN(784, 10)
x = torch.randn(64, 784)
print(model(x).shape)



# set device
# hyperparameters
# load data
# initialize network
# loss and optimizer
# train network
# check accuracy on training and test to see how good our model is