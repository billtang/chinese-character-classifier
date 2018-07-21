'''
Author: William Hu (william.hu@yale.edu)
Acknowledgements: Much of the code is adapted from the official Pytorch tutorials,
along with yunjey's machine learning tutorials.
'''

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from random import randint
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading the dictionary")
character_dict = torch.load('character_dict.pt')
torch.backends.cudnn.enabled = False

# Parameters
batch_size = 32
learning_rate = 1e-4
num_classes = len(character_dict)

# Dataset Class
class ChineseCharacterDataset(Dataset):
    def __init__(self, data, transform=None):
        self.dataset = torch.load(data)
        self.length = len(self.dataset)
        while self.length % batch_size != 0:
            self.dataset += [self.dataset[randint(0, self.length - 1)]]
            self.length += 1
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            return (self.transform(self.dataset[index][0]), self.dataset[index][1])
        else:
            return self.dataset[index]

    def __len__(self):
        return self.length

# Two-Layer Convolutional Neural Network
class ConvNN(torch.nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
                nn.Conv2d(64, 1024, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2))
        self.dp = nn.Dropout(p=0.25)
        self.fc = nn.Linear(1024 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 1024 * 7 * 7)
        x = self.fc(x)
        return x

# Load the training and testing data
print("Starting to load the data")
training_dataset = ChineseCharacterDataset('training_data.pt', transform=transforms.Compose([
                                                                        transforms.ToPILImage(),
                                                                        transforms.RandomCrop(128),
                                                                        transforms.ToTensor()]))
training_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
print("Done loading the training data")
testing_dataset = ChineseCharacterDataset('testing_data.pt')
testing_loader = DataLoader(dataset=testing_dataset, batch_size=batch_size)
print("Done loading the testing data")

# Build the model
print("Building the model...")
model = ConvNN().to(device)

# Cross-entropy loss function and SGD optimization
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Learning rate decay
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training
num_epochs = 25
best_accuracy = 0.0
best_epoch = 0

for epoch in range(1, num_epochs + 1):
    print('Epoch {}/{}'.format(epoch, num_epochs))
    print('-' * 10)

    final_loss = 0
    num_correct = 0
    num_total = 0

    model.train()
    for i, (images, labels) in enumerate(tqdm(training_loader)):
        images = images.to(device)
        labels = labels.to(device)

        # Compute the output, calculate the loss, and readjust the weights
        predictions = model(images)
        labels = labels.long()
        loss = criterion(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute the final loss and training accuracy
        final_loss = loss.item()
        _, prediction = torch.max(predictions.data, 1)
        num_correct += (prediction == labels).sum().item()
        num_total += labels.size(0)

    acc = num_correct / num_total
    print('Training Accuracy: {:.4f}'.format(acc))
    print('Final Loss: {:.4f}'.format(final_loss))

    # Testing
    model.eval()
    with torch.no_grad():
        num_correct = 0
        num_total = 0

        for images, labels in tqdm(testing_loader):
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            _, prediction = torch.max(predictions.data, 1)
            num_correct += (prediction == labels).sum().item()
            num_total += labels.size(0)

        acc = num_correct / num_total
        print('Testing Accuracy: {:.4f}'.format(acc) + '\n')
        if acc > best_accuracy:
            best_accuracy = acc
            best_epoch = epoch
            torch.save(model.state_dict(), 'model.pt')

# Save the model
print('-' * 10)
print('Best Accuracy of {:.4f} at Epoch {}'.format(best_accuracy, best_epoch))
