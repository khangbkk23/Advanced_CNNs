import torch
import torch.nn.functional as F
import torch.nn as nn
class CNN(nn.Module):
   def __init__(self, in_channels, num_classes):

       """
       Building blocks of convolutional neural network.

       Parameters:
           * in_channels: Number of channels in the input image (for grayscale images, 1)
           * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
       """
       super(CNN, self).__init__()

       # 1st convolutional layer
       self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
       # Max pooling layer
       self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
       # 2nd convolutional layer
       self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
       # Fully connected layer
       self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

   def forward(self, x):
       x = F.relu(self.conv1(x))
       x = self.pool(x)
       x = F.relu(self.conv2(x))
       x = self.pool(x)
       x = x.reshape(x.shape[0], -1)
       x = self.fc1(x)
       return x
       x = x.reshape(x.shape[0], -1)
       x = self.fc1(x)
       return x