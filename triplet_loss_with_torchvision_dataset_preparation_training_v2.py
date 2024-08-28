import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random

class SimpleCNN(nn.Module):
    """
    A simple convolutional neural network for generating embeddings.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),  # Adjusted input size to match output of the last conv layer
            nn.ReLU(),
            nn.Linear(128, 64)  # Output embedding dimension of 64
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x

class TripletLoss(nn.Module):
    """
    Triplet loss function with a margin.
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pairwise_distance = nn.PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative):
        """
        Compute the triplet loss.

        Parameters:
        - anchor (Tensor): Embeddings of anchor images.
        - positive (Tensor): Embeddings of positive images.
        - negative (Tensor): Embeddings of negative images.

        Returns:
        - Tensor: Computed loss value.
        """
        d_ap = self.pairwise_distance(anchor, positive)
        d_an = self.pairwise_distance(anchor, negative)
        loss = torch.clamp(d_ap - d_an + self.margin, min=0.0)
        return loss.mean()

class TripletImageFolder(datasets.ImageFolder):
    """
    Custom dataset for triplet sampling using ImageFolder.
    """
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.label_to_indices = self._create_label_to_indices()

    def _create_label_to_indices(self):
        """
        Create a dictionary mapping labels to indices.

        Returns:
        - dict: Mapping from labels to list of indices.
        """
        label_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.samples):
            label_to_indices[label].append(idx)
        return label_to_indices

    def __getitem__(self, index):
        """
        Get a triplet (anchor, positive, negative) image.

        Parameters:
        - index (int): Index of the anchor image.

        Returns:
        - tuple: (anchor_image, positive_image, negative_image)
        """
        anchor_image, anchor_label = super().__getitem__(index)
        positive_image = self._get_positive_image(anchor_label)
        negative_image = self._get_negative_image(anchor_label)
        
        return anchor_image, positive_image, negative_image

    def _get_positive_image(self, anchor_label):
        """
        Get a positive image with the same label as the anchor image.

        Parameters:
        - anchor_label (int): The label of the anchor image.

        Returns:
        - PIL.Image or Tensor: The positive image.
        """
        positive_indices = self.label_to_indices[anchor_label]
        positive_index = random.choice(positive_indices)
        positive_image, _ = super().__getitem__(positive_index)
        return positive_image

    def _get_negative_image(self, anchor_label):
        """
        Get a negative image with a different label from the anchor image.

        Parameters:
        - anchor_label (int): The label of the anchor image.

        Returns:
        - PIL.Image or Tensor: The negative image.
        """
        negative_labels = list(self.label_to_indices.keys())
        negative_labels.remove(anchor_label)
        negative_label = random.choice(negative_labels)
        negative_index = random.choice(self.label_to_indices[negative_label])
        negative_image, _ = super().__getitem__(negative_index)
        return negative_image

def train(model, dataloader, criterion, optimizer, num_epochs=5):
    """
    Training loop for the triplet network.

    Parameters:
    - model (nn.Module): The neural network model to train.
    - dataloader (DataLoader): DataLoader providing the triplet batches.
    - criterion (nn.Module): Loss function.
    - optimizer (optim.Optimizer): Optimizer for updating model parameters.
    - num_epochs (int): Number of epochs to train.
    """
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
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

if __name__ == "__main__":
    # Set up transforms and dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Adjusted to 128x128 for demonstration
        transforms.Grayscale(num_output_channels=1),  # Ensure images are grayscale
        transforms.ToTensor()
    ])

    dataset = TripletImageFolder(root='path/to/your/data', transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    # Initialize model, loss function, and optimizer
    model = SimpleCNN()
    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train(model, dataloader, criterion, optimizer, num_epochs=5)
