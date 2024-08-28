import os
import random
from collections import defaultdict
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class TripletImageFolder(Dataset):
    """
    A Dataset class that generates triplets of (anchor, positive, negative) images
    for training triplet networks.
    """
    def __init__(self, root_dir, transform=None):
        """
        Initialize the TripletImageFolder dataset.

        Parameters:
        - root_dir (str): Path to the root directory containing the dataset.
        - transform (callable, optional): A function/transform to apply to each image.
        """
        self.image_folder = datasets.ImageFolder(root=root_dir, transform=transform)
        self.transform = transform
        self.label_to_indices = self._create_label_to_indices()

    def _create_label_to_indices(self):
        """
        Create a mapping from labels to indices in the dataset.

        Returns:
        - dict: A dictionary mapping labels to lists of indices.
        """
        label_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.image_folder.samples):
            label_to_indices[label].append(idx)
        return label_to_indices

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
        - int: The number of samples in the dataset.
        """
        return len(self.image_folder)

    def __getitem__(self, index):
        """
        Get a triplet (anchor, positive, negative) image.

        Parameters:
        - index (int): The index of the anchor image.

        Returns:
        - tuple: A tuple (anchor_image, positive_image, negative_image).
        """
        anchor_image, anchor_label = self.image_folder[index]
        positive_image = self._get_positive_image(anchor_label)
        negative_image = self._get_negative_image(anchor_label)

        return anchor_image, positive_image, negative_image

    def _get_positive_image(self, anchor_label):
        """
        Get a positive image that shares the same label as the anchor image.

        Parameters:
        - anchor_label (int): The label of the anchor image.

        Returns:
        - PIL.Image or Tensor: The positive image.
        """
        positive_indices = self.label_to_indices[anchor_label]
        positive_index = random.choice(positive_indices)
        positive_image, _ = self.image_folder[positive_index]
        if self.transform:
            positive_image = self.transform(positive_image)
        return positive_image

    def _get_negative_image(self, anchor_label):
        """
        Get a negative image that has a different label than the anchor image.

        Parameters:
        - anchor_label (int): The label of the anchor image.

        Returns:
        - PIL.Image or Tensor: The negative image.
        """
        negative_labels = list(self.label_to_indices.keys())
        negative_labels.remove(anchor_label)
        negative_label = random.choice(negative_labels)
        negative_index = random.choice(self.label_to_indices[negative_label])
        negative_image, _ = self.image_folder[negative_index]
        if self.transform:
            negative_image = self.transform(negative_image)
        return negative_image

# Define transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# Initialize dataset and dataloader
dataset = TripletImageFolder(root_dir='path/to/your/data', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

if __name__ == "__main__":
    # Example usage
    for anchor, positive, negative in dataloader:
        # Your training loop here
        pass
