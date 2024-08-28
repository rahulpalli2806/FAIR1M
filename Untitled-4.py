'''
To modify the code to use Triplet Loss instead of Cross-Entropy Loss, you need to make several changes to both `training.py` and the importing script. Here's how you can adapt your code:

### Changes in `training.py`

1. **Add Triplet Loss Function**:
   Define a function for Triplet Loss. PyTorch’s `torch.nn.TripletMarginLoss` can be used for this purpose.

2. **Modify the `pass_epoch` Function**:
   Since Triplet Loss requires triplets of anchor, positive, and negative samples, you need to adjust the `pass_epoch` function to handle triplets.

Here’s how to incorporate these changes:

'''
import torch
import numpy as np
import time

class Logger(object):
    # (no changes required for this class)


class BatchTimer(object):
    # (no changes required for this class)


def accuracy(logits, y):
    _, preds = torch.max(logits, 1)
    return (preds == y).float().mean()


def triplet_loss(anchor, positive, negative, margin=0.2):
    """
    Compute the triplet loss given anchor, positive, and negative embeddings.
    """
    criterion = torch.nn.TripletMarginLoss(margin=margin, p=2)
    return criterion(anchor, positive, negative)


def pass_epoch(
    model, loss_fn, loader, optimizer=None, scheduler=None,
    batch_metrics={'time': BatchTimer()}, show_running=True,
    device='cpu', writer=None
):
    """Train or evaluate over a data epoch.
    
    Arguments:
        model {torch.nn.Module} -- Pytorch model.
        loss_fn {callable} -- A function to compute (scalar) loss.
        loader {torch.utils.data.DataLoader} -- A pytorch data loader.
    
    Keyword Arguments:
        optimizer {torch.optim.Optimizer} -- A pytorch optimizer.
        scheduler {torch.optim.lr_scheduler._LRScheduler} -- LR scheduler (default: {None})
        batch_metrics {dict} -- Dictionary of metric functions to call on each batch. The default
            is a simple timer. A progressive average of these metrics, along with the average
            loss, is printed every batch. (default: {{'time': iter_timer()}})
        show_running {bool} -- Whether or not to print losses and metrics for the current batch
            or rolling averages. (default: {False})
        device {str or torch.device} -- Device for pytorch to use. (default: {'cpu'})
        writer {torch.utils.tensorboard.SummaryWriter} -- Tensorboard SummaryWriter. (default: {None})
    
    Returns:
        tuple(torch.Tensor, dict) -- A tuple of the average loss and a dictionary of average
            metric values across the epoch.
    """
    
    mode = 'Train' if model.training else 'Valid'
    logger = Logger(mode, length=len(loader), calculate_mean=show_running)
    loss = 0
    metrics = {}

    for i_batch, (anchor, positive, negative) in enumerate(loader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        y_pred_anchor = model(anchor)
        y_pred_positive = model(positive)
        y_pred_negative = model(negative)
        loss_batch = loss_fn(y_pred_anchor, y_pred_positive, y_pred_negative)

        if model.training:
            loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()

        metrics_batch = {}
        for metric_name, metric_fn in batch_metrics.items():
            metrics_batch[metric_name] = metric_fn(y_pred_anchor, positive).detach().cpu()
            metrics[metric_name] = metrics.get(metric_name, 0) + metrics_batch[metric_name]
            
        if writer is not None and model.training:
            if writer.iteration % writer.interval == 0:
                writer.add_scalars('loss', {mode: loss_batch.detach().cpu()}, writer.iteration)
                for metric_name, metric_batch in metrics_batch.items():
                    writer.add_scalars(metric_name, {mode: metric_batch}, writer.iteration)
            writer.iteration += 1
        
        loss_batch = loss_batch.detach().cpu()
        loss += loss_batch
        if show_running:
            logger(loss, metrics, i_batch)
        else:
            logger(loss_batch, metrics_batch, i_batch)
    
    if model.training and scheduler is not None:
        scheduler.step()

    loss = loss / (i_batch + 1)
    metrics = {k: v / (i_batch + 1) for k, v in metrics.items()}
            
    if writer is not None and not model.training:
        writer.add_scalars('loss', {mode: loss.detach()}, writer.iteration)
        for metric_name, metric in metrics.items():
            writer.add_scalars(metric_name, {mode: metric})

    return loss, metrics


def collate_pil(x): 
    out_x, out_y = [], [] 
    for xx, yy in x: 
        out_x.append(xx) 
        out_y.append(yy) 
    return out_x, out_y 
```

### Changes in the Importing Script

1. **Adjust Dataset**:
   You need to prepare your dataset to provide triplets. This requires a custom dataset that yields triplets of anchor, positive, and negative images.

2. **Change Loss Function**:
   Update `loss_fn` to use the triplet loss function.

Here’s how to adapt the importing script:

```python
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os

# Define a custom dataset class that yields triplets
class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.class_to_idx = dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        anchor_img, anchor_label = self.dataset[index]
        positive_img, positive_label = self.find_positive(anchor_label)
        negative_img, negative_label = self.find_negative(anchor_label)
        return anchor_img, positive_img, negative_img

    def find_positive(self, anchor_label):
        # Method to find a positive example
        while True:
            idx = np.random.choice(len(self.dataset))
            img, label = self.dataset[idx]
            if label == anchor_label:
                return img, label

    def find_negative(self, anchor_label):
        # Method to find a negative example
        while True:
            idx = np.random.choice(len(self.dataset))
            img, label = self.dataset[idx]
            if label != anchor_label:
                return img, label

data_dir = '../data/test_images'

batch_size = 32
epochs = 8
workers = 0 if os.name == 'nt' else 8

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Initialize the model
resnet = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)  # Changed classify=False

# Define Triplet Loss
def triplet_loss(anchor, positive, negative):
    criterion = torch.nn.TripletMarginLoss(margin=0.2, p=2)
    return criterion(anchor, positive, negative)

optimizer = optim.Adam(resnet.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, milestones=[5, 10])

trans = transforms.Compose([
    transforms.ToTensor(),
    fixed_image_standardization
])

dataset = datasets.ImageFolder(data_dir + '_cropped', transform=trans)
triplet_dataset = TripletDataset(dataset)

img_inds = np.arange(len(triplet_dataset))
np.random.shuffle(img_inds)
train_inds = img_inds[:int(0.8 * len(img_inds))]
val_inds = img_inds[int(0.8 * len(img_inds)):]

train_loader = DataLoader(
    triplet_dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_inds)
)
val_loader = DataLoader(
    triplet_dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(val_inds)
)

metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
}

writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10

for epoch in range(epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    resnet.train()
    training.pass_epoch(
        resnet, triplet_loss, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    resnet.eval()
    training.pass_epoch(
        resnet, triplet_loss, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

writer.close()
```

### Key Points

1. **Dataset**:
   - A custom dataset `TripletDataset` is defined to yield triplets (anchor, positive, negative).

2. **Model**:
   - `InceptionResnetV1` is set with `classify=False` since we are not performing classification anymore.

3. **Loss Function**:
   - The triplet loss function is defined and used in place of cross-entropy loss.

By following these changes, you will be able to run training with Triplet Loss effectively.