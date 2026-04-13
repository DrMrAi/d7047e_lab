# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Load dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# CNN setup
class SimpleCNN(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

        self.conv1 = nn.Conv2d(3, 32, 3) # (Convolution layer 1) Input: 3 channels (RGB), Output: 32 feature maps, Kernel size: 3x3
        self.conv2 = nn.Conv2d(32, 64, 3) # (Convolution layer 2) Takes 32 feature maps and produces 64
        self.pool = nn.MaxPool2d(2, 2) # (Max pooling) Reduces computation and extracts important features

        self.fc1 = nn.Linear(64 * 6 * 6, 128) # (Fully connected layer) Input: 64 feature maps that are 6x6, Output: 128 neurons
        self.fc2 = nn.Linear(128, 10) # (Final layer) Input: 128 neurons, Output: 10 classes because CIFAR-10 has 10 classes

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x))) # First conv layer → activation → pooling
        x = self.pool(self.activation(self.conv2(x))) # Second conv layer → activation → pooling
        x = x.view(-1, 64 * 6 * 6) # Flattens feature maps into a vector
        x = self.activation(self.fc1(x)) # Fully connected layer → activation
        x = self.fc2(x) # Final layer
        return x

# Training
def train_model(model, trainloader, optimizer, criterion, epochs=10):
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for inputs, labels in trainloader:
            optimizer.zero_grad() # Resets gradients
            outputs = model(inputs) # Forward pass to get predictions
            loss = criterion(outputs, labels) # Computes loss
            loss.backward() # Backpropogation to compute gradients
            optimizer.step() # Update weights using optimizer

            running_loss += loss.item() # Adds batch loss to total loss

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.3f}")

# Evaluation
def evaluate_model(model, testloader):
    model.eval()

    correct = 0
    total = 0

    # Disables gradient computation
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images) # Forward pass
            _, predicted = torch.max(outputs, 1) # Get predicted class
            total += labels.size(0) # Count total samples
            correct += (predicted == labels).sum().item() # Counts total correct predictions

    return 100 * correct / total

model = SimpleCNN(F.leaky_relu)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

train_model(model, trainloader, optimizer, criterion)
acc = evaluate_model(model, testloader)

# Expected accuracy 50-60%
print("Accuracy:", acc)
# Adam CNN
model = SimpleCNN(F.leaky_relu)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_model(model, trainloader, optimizer, criterion)
acc = evaluate_model(model, testloader)

# Expected accuracy 60-70%
print("Accuracy:", acc)
#Task 0.2.1 fine-tuning
print("starting")
from torchvision.models import alexnet, AlexNet_Weights

weights = AlexNet_Weights.DEFAULT
model_ft = alexnet(weights = weights)

# Replace final layer (alexnet has 4096 classes, we have 10)
num_ftrs = model_ft.classifier[6].in_features
model_ft.classifier[6] = nn.Linear(num_ftrs, 10)

# Loss function
criterion_ft = nn.CrossEntropyLoss()

# Optimizer (not frozen here)
optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=0.0001)

#Transforms, for resizing pictures to fit alexnet
transforms_ft = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

trainset_ft = torchvision.datasets.CIFAR10(
    root='./cifar_data',
    train=True,
    download=True,
    transform=transforms_ft
)

testset_ft = torchvision.datasets.CIFAR10(
    root='./cifar_data',
    train=False,
    download=True,
    transform=transforms_ft
)

#Dataloaders
trainloader_ft = torch.utils.data.DataLoader(trainset_ft, batch_size=64, shuffle=True)
testloader_ft = torch.utils.data.DataLoader(testset_ft, batch_size=64, shuffle=False)

#Train the model
print("training")
train_model(model_ft, trainloader_ft, optimizer_ft, criterion_ft, epochs=3)

#Test the model
print("testing")
acc = evaluate_model(model_ft, testloader_ft)

print("Accuracy:", acc)
#task 0.2.1 feature extraction
model_fe = alexnet(weights = weights)

# Freeze all pretrained layers
for param in model_fe.parameters():
    param.requires_grad = False

# Replace final layer
num_ftrs = model_fe.classifier[6].in_features
model_fe.classifier[6] = nn.Linear(num_ftrs, 10)

# Loss function
criterion_fe = nn.CrossEntropyLoss()

# Optimizer, trains only final classification layer
optimizer_fe = torch.optim.Adam(model_fe.classifier[6].parameters(), lr=0.0001)


trainloader_fe = trainloader_ft
testloader_fe = testloader_ft

# Train the model
train_model(model_fe, trainloader_fe, optimizer_fe, criterion_fe, epochs=3)

#Test the model
acc = evaluate_model(model_fe, testloader_fe)

print("Accuracy:", acc)




# =========================
# Task 0.2.2 MNIST → SVHN
# =========================

print("\n--- Task 0.2.2: MNIST → SVHN ---")

# -------------------------
# 1. Load MNIST
# -------------------------
transform_mnist = transforms.Compose([
    transforms.Resize((32, 32)),                  # match SVHN size
    transforms.Grayscale(num_output_channels=3),  # convert to 3 channels
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset_mnist = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform_mnist
)

testset_mnist = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform_mnist
)

trainloader_mnist = torch.utils.data.DataLoader(trainset_mnist, batch_size=64, shuffle=True)
testloader_mnist = torch.utils.data.DataLoader(testset_mnist, batch_size=64, shuffle=False)

# -------------------------
# 2. Train CNN on MNIST
# -------------------------
model_mnist = SimpleCNN(F.leaky_relu)
criterion_mnist = nn.CrossEntropyLoss()
optimizer_mnist = torch.optim.Adam(model_mnist.parameters(), lr=0.001)

print("Training on MNIST...")
train_model(model_mnist, trainloader_mnist, optimizer_mnist, criterion_mnist, epochs=5)

acc_mnist = evaluate_model(model_mnist, testloader_mnist)
print("MNIST Accuracy:", acc_mnist)


# -------------------------
# 3. Load SVHN
# -------------------------
transform_svhn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset_svhn = torchvision.datasets.SVHN(
    root='./data',
    split='train',
    download=True,
    transform=transform_svhn
)

testset_svhn = torchvision.datasets.SVHN(
    root='./data',
    split='test',
    download=True,
    transform=transform_svhn
)

trainloader_svhn = torch.utils.data.DataLoader(trainset_svhn, batch_size=64, shuffle=True)
testloader_svhn = torch.utils.data.DataLoader(testset_svhn, batch_size=64, shuffle=False)


# -------------------------
# 4. Transfer Learning (MNIST → SVHN)
# -------------------------

# Copy pretrained weights
model_svhn = SimpleCNN(F.leaky_relu)
model_svhn.load_state_dict(model_mnist.state_dict())

# OPTIONAL: Freeze convolution layers (true transfer learning)
for param in model_svhn.conv1.parameters():
    param.requires_grad = False
for param in model_svhn.conv2.parameters():
    param.requires_grad = False

# New optimizer (only trains unfrozen layers)
optimizer_svhn = torch.optim.Adam(filter(lambda p: p.requires_grad, model_svhn.parameters()), lr=0.001)
criterion_svhn = nn.CrossEntropyLoss()

print("\nTraining on SVHN (transfer learning)...")
train_model(model_svhn, trainloader_svhn, optimizer_svhn, criterion_svhn, epochs=5)

acc_svhn = evaluate_model(model_svhn, testloader_svhn)
print("SVHN Accuracy (Transfer):", acc_svhn)