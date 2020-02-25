'''
This example code shows how to use the PyTorch integration for online training
(example data loaders and training loop).

This code is only provided as a reference. In a real application, you would
need to load in actual image paths to train over.
'''

from flyingsquid.label_model import LabelModel
from flyingsquid.pytorch_loss import FSLoss
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader

# Load in the L matrices
L_train = np.load('tutorials/L_train_video.npy')
L_dev = np.load('tutorials/L_dev_video.npy')
Y_dev = np.load('tutorials/Y_dev_video.npy')

# This is where you would load in the images corresponding to the rows
X_paths_train = np.load('....')
X_paths_dev = np.load('....')

# Example dataloader for FSLoss
class ImageFSDataset(Dataset):
    def __init__(self, paths, weak_labels, T, gt_labels=None,
                 transform=None):
        self.T = T
        self.paths = self.paths[:len(self.paths) - (len(self.paths) % T)]
        self.weak_labels = self.weak_labels[:self.weak_labels.shape[0] - 
                                            (self.weak_labels.shape[0] % T)]
        m_per_task = self.weak_labels.shape[1]

        self.transform = transform
        
        n_frames = self.weak_labels.shape[0]
        n_seqs = n_frames // T
        
        v = T
        m = m_per_task * T
        
        self.data_temporal = {
            'paths': np.reshape(self.paths, (n_seqs, v)),
            'weak_labels': np.reshape(self.weak_labels, (n_seqs, m))
        }
        
        self.gt_labels = gt_labels
        if gt_labels is not None:
            self.gt_labels = self.gt_labels[:len(self.gt_labels) -
                                            (len(self.gt_labels) % T)]
            self.data_temporal['gt_labels'] = np.reshape(self.gt_labels, (n_seqs, v))

    def __len__(self):
        return self.data_temporal['paths'].shape[0]

    def __getitem__(self, idx):
        paths_seq = self.data_temporal['paths'][idx]
        
        img_tensors = [
            torch.unsqueeze(
                self.transform(Image.open(path).convert('RGB')), 
                dim = 0)
            for path in paths_seq
        ]
        
        weak_labels = self.data_temporal['weak_labels'][idx]
        
        if self.gt_labels is not None:
            return (torch.cat(img_tensors),
                    torch.unsqueeze(torch.tensor(weak_labels), dim=0), 
                    torch.unsqueeze(torch.tensor(self.data_temporal['gt_labels'][idx]), dim = 0))
        else:
            return torch.cat(img_tensors), torch.unsqueeze(torch.tensor(weak_labels), dim = 0)
        
# Example training loop
def train_model_online(model, T, criterion, optimizer, dataset):
    model.train()
    dataset_size = len(dataset) * T

    for item in dataset:
        image_tensor = item[0]
        weak_labels = item[1]
        labels = None if dataset.gt_labels is None else item[2]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        with torch.set_grad_enabled(True):
            outputs = model(inputs)

            loss = criterion(torch.unsqueeze(outputs, dim = 0), weak_labels)

            # backward + optimize
            loss.backward()
            optimizer.step()

    return model

# Model three frames at a time
v = 3

# Set up the dataset
train_dataset = ImageFSDataset(X_paths_train, L_train, v)

# Set up the loss function
fs_criterion = FSLoss(
    m,
    v = v, 
    y_edges = [ (i, i + 1) for i in range(v - 1) ],
    lambda_y_edges = [ (i, i // m_per_frame) for i in range(m) ]
)

# Train up a model online
model = models.resnet50(pretrained=True)
num_ftrs = model_online.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model = train_model_online(model, v, fs_criterion, optimizer, dataset)
