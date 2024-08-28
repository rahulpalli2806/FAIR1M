import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # Output embedding dimension of 64
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pairwise_distance = nn.PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative):
        d_ap = self.pairwise_distance(anchor, positive)
        d_an = self.pairwise_distance(anchor, negative)
        loss = torch.clamp(d_ap - d_an + self.margin, min=0.0)
        return loss.mean()

class TripletImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.label_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.samples):
            self.label_to_indices[label].append(idx)

    def __getitem__(self, index):
        anchor_image, anchor_label = super().__getitem__(index)
        positive_index = self._get_positive_index(anchor_label)
        positive_image, _ = super().__getitem__(positive_index)
        
        negative_label = self._get_negative_label(anchor_label)
        negative_index = random.choice(self.label_to_indices[negative_label])
        negative_image, _ = super().__getitem__(negative_index)
        
        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)
            
        return anchor_image, positive_image, negative_image

    def _get_positive_index(self, anchor_label):
        positive_indices = self.label_to_indices[anchor_label]
        return random.choice(positive_indices)

    def _get_negative_label(self, anchor_label):
        negative_labels = list(self.label_to_indices.keys())
        negative_labels.remove(anchor_label)
        return random.choice(negative_labels)

# Set up transforms and dataset
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# Dummy data
num_samples = 100
images = [torch.randn(1, 28, 28) for _ in range(num_samples)]  # Random images with 1 channel (e.g., grayscale)
labels = [i % 10 for i in range(num_samples)]  # Random labels (10 classes)

# Create dataset and dataloader
dataset = TripletImageFolder(root='path/to/your/data', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize model, loss function, and optimizer
model = SimpleCNN()
criterion = TripletLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for anchor, positive, negative in dataloader:
        # Forward pass
        anchor_embed = model(anchor)
        positive_embed = model(positive)
        negative_embed = model(negative)
        
        # Compute loss
        loss = criterion(anchor_embed, positive_embed, negative_embed)
        
        # Zero gradients, backward pass, and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)