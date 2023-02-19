###  here's my  PyTorch code for transfer learning using a pre-trained model:



import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

# Define the pre-trained model
model = torchvision.models.resnet18(pretrained=True)

# Freeze all the layers of the pre-trained model
for param in model.parameters():
    param.requires_grad = False

# Modify the last layer to fit our problem
num_classes = 10  # Change this to the number of classes in your problem
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# Define the data transformations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the data
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_loader = data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)

val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
val_loader = data.DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2)

# Train the model
num_epochs = 10  # Change this to the number of epochs you want to train for

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# Evaluate the model on the validation set
correct = 0
total = 0

with torch.no_grad():
    for data in val_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on the validation set: {correct / total}")
