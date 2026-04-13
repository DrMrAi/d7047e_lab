# Imports
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from comet_ml import Experiment

# -------------------------
# Comet setup (IMPORTANT)
# -------------------------
experiment = Experiment(
    api_key="lZ6VM5qo8Rz8JKYXjXVS3zWzs",  # set this in your environment
    project_name="D7047E"
)

experiment.set_name("SimpleCNN_CIFAR10")
experiment.log_parameters({
    "optimizer": "SGD",
    "lr": 0.01,
    "batch_size": 64,
    "epochs": 10,
    "architecture": "SimpleCNN"
})

# -------------------------
# Load dataset
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# -------------------------
# CNN
# -------------------------
class SimpleCNN(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------
# Training with Comet logging
# -------------------------
def train_model(model, trainloader, optimizer, criterion, epochs=10):
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)

        print(f"Epoch {epoch+1}, Loss: {avg_loss:.3f}")

        # 🔥 Log to Comet
        experiment.log_metric("loss", avg_loss, step=epoch + 1)

# -------------------------
# Evaluation
# -------------------------
def evaluate_model(model, testloader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    return acc

# -------------------------
# Run
# -------------------------
model = SimpleCNN(F.leaky_relu)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

train_model(model, trainloader, optimizer, criterion)

acc = evaluate_model(model, testloader)

print("Accuracy:", acc)

# 🔥 Log final accuracy
experiment.log_metric("test_accuracy", acc)

experiment.end()