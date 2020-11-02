import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.utils import save_image
import torchvision.transforms as transforms
import helper


# Class to define NN
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # same convolution
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))  # same convolution
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


# Set device settings if Nvidia use cuda core else use cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""
Full Connected NN

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer
criterior = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        data = data.reshape(data.shape[0], -1)

        # forward
        scores = model(data)
        loss = criterior(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gredient descend
        optimizer.step()
        
        
        check_accuracy(train_loader, model)
        check_accuracy(test_loader, model)
        # save_model(model, './model/mymodel')
        img = image_loader(data_transforms, './Immagine.bmp')
        img = img.reshape(img.shape[0], -1)
        scores = model(img)
        _, predictions = scores.max(1)
        print(predictions)
        """



# Method for checking accuracy on TR and TS
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            """img = x[1]
            save_image(img, 'img2.bmp')"""
            # x = x.reshape(x.shape[0], -1) # faccio il reshape se sto in NN
            scores = model(x)
            _, predictions = scores.max(1)
            # print("Previsione: "+ str(predictions.numpy()[0])+" -  Valore reale: "+str(y.numpy()[0]))
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
        )
    model.train()


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_NNmodel(PATH, input_size, num_classes):
    mod = NN(input_size=input_size, num_classes=num_classes).to(device)
    mod.load_state_dict(torch.load(PATH))
    mod.eval()
    return mod


def load_CNNmodel(PATH):
    mod = CNN().to(device)
    mod.load_state_dict(torch.load(PATH))
    mod.eval()
    return mod


def image_loader2(path, transform):
    dataset = datasets.ImageFolder(path, transform=transform)
    return dataset


data_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()])

"""
# Prediction on NN
def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image


model = load_model('./model/mymodel')
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
check_accuracy(test_loader, model)
# leggo l'immagini create da me
img = image_loader2('./input', transform=data_transforms)
dataloader = torch.utils.data.DataLoader(img, batch_size=1, shuffle=False)
for x in dataloader:
    x = x[0]
    x = x.reshape(x.shape[0], -1)
    scores = model(x)
    _, predictions = scores.max(1)
    print("Previsione: " + str(predictions.numpy()[0]))
"""

"""
Train CNN
# Set device settings if Nvidia use cuda core else use cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channel = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

# Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
model = CNN().to(device)

# Loss and optimizer
criterior = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        # forward
        scores = model(data)
        loss = criterior(scores, targets)
        # backward
        optimizer.zero_grad()
        loss.backward()
        # gredient descend
        optimizer.step()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
save_model(model, './model/mycnn')
"""
# Previsione su CNN
batch_size=64
model = load_CNNmodel('./model/mycnn')
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
check_accuracy(test_loader, model)
# leggo l'immagini create da me
img = image_loader2('./input', transform=data_transforms)
dataloader = torch.utils.data.DataLoader(img, batch_size=1, shuffle=False)
for x in dataloader:
    x = x[0]
    scores = model(x)
    _, predictions = scores.max(1)
    print("Previsione: " + str(predictions.numpy()[0]))