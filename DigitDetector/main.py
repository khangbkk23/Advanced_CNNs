from data.data import get_data_loaders
from utils.set_seed import seed_all
from models.CNN import CNN
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision

seed_all(seed=42)
# Load data
train_loader, test_loader = get_data_loaders(config_path="./config.yaml")

# Show some samples
def imshow(img):
   npimg = img.numpy()
   plt.imshow(np.transpose(npimg, (1, 2, 0)))
   plt.show()

dataiter = iter(train_loader)
images, labels = next(dataiter)
labels
imshow(torchvision.utils.make_grid(images))

# CNN model
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN(in_channels=1, num_classes=10).to(device)
print(model)

