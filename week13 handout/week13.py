import torch
import torchvision
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# Set seed for reproducibility
torch.manual_seed(0)
# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) # Normalize the image data
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform) # Load training set
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True) # Create a dataloader for training set

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform) # Load test set
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False) # Create a dataloader for test set

# Define CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) # 1 input channel, 32 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 32 input channels, 64 output channels, 3x3 kernel
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 2x2 pooling
        self.fc1 = nn.Linear(64 * 7 * 7, 128) # 64 * 7 * 7 input features, 128 output
        self.fc2 = nn.Linear(128, 10) # 128 input, 10 output
        self.relu = nn.ReLU()

    def forward(self, x): 
        x = self.pool(self.relu(self.conv1(x))) # Convolutional layer 1 -> ReLU -> Pooling
        x = self.pool(self.relu(self.conv2(x))) # Convolutional layer 2 -> ReLU -> Pooling
        x = x.view(-1, 64 * 7 * 7) # Flatten
        x = self.relu(self.fc1(x)) # Fully connected layer 1 -> ReLU
        x = self.fc2(x) # Fully connected layer 2
        return x 
 
# Instantiate model, define loss function and optimizer
model = CNNModel() 
criterion = nn.CrossEntropyLoss() # loss function
optimizer = optim.Adam(model.parameters(), lr=0.001) # optimizer




# Training the model
num_epochs = 5
train_losses = []
test_losses = []

for epoch in range(num_epochs): # Loop through the dataset multiple times
    model.train()
    running_loss = 0.0
    for images, labels in trainloader: # Loop through the training set
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(trainloader))

    model.eval()
    test_loss = 0.0
    with torch.no_grad(): 
        for images, labels in testloader: 
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    test_losses.append(test_loss / len(testloader))

# Save the model
torch.save(model.state_dict(), 'cnn_model.pth')

# Draw the learning curve
import matplotlib.pyplot as plt

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Test loss')
plt.legend()
plt.show()

# Evaluate the model
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

# Print first 10 test results
# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get some random test images
dataiter = iter(testloader)
images, labels = next(dataiter)

# Print images
imshow(torchvision.utils.make_grid(images[:10]))
model.eval()
with torch.no_grad():
    for i, (images, labels) in enumerate(testloader):
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        print(f'Predicted: {predicted[:10]}, Actual: {labels[:10]}')
        if i == 0:
            break