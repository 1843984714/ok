import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

batch_size = 30
learning_rate = 0.01
momentum = 0.5
epochs = 20

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)  # 6000
test_datatast = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)  # 1000

# print(len(train_dataset))
# print(len(test_datatast))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # self.pooling = nn.MaxPool2d(kernel_size=2)
        # self.fc1 = nn.Linear(260, 120)
        # self.fc2 = nn.Linear(120, 10)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        # x = F.relu(self.pooling(self.conv1(x)))
        # x = F.relu(self.pooling(self.conv2(x)))
        # x = x.view(batch_size, -1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


model = Net()
# print(model)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

model = model.to(device)


def train(epoch):
    running_loss = 0.0
    # running_total = 0
    # running_correct = 0
    for batch_index, (inputs, targets) in enumerate(train_loader, 0):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_index % 300 == 299:
            print(
                "[%d/%d]:loss:%.3f" % (epoch + 1, batch_index + 1, running_loss / 300)
            )
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    print(100 * correct / total)


for epoch in range(epochs):
    train(epoch)
    test()
