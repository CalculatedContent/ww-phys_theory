import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from torchvision import datasets
from torchvision import transforms

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def get_data(batch_size=100, train_range=None, test_range=None):
  normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
  transform_train = transforms.Compose([
          transforms.ToTensor(),
          normalize])
  transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize])
  train_dataset = datasets.CIFAR10(root='data', 
                                train=True, 
                                transform=transform_train,
                                download=True)
  test_dataset = datasets.CIFAR10(root='data', 
                                train=False, 
                                transform=transform_test,
                                download=True)  
  
  if train_range:
      train_dataset = Subset(train_dataset, train_range)

  if test_range:
      teat_dataset = Subset(test_dataset, test_range)


  train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size,
                          num_workers=4,
                          shuffle=False)
  test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size,
                         num_workers=4,
                         shuffle=False)
  return train_dataset, test_dataset, train_loader, test_loader




def get_esd_plot(eigenvalues, weights):
  density, grids = density_generate(eigenvalues, weights)
  plt.semilogy(grids, density + 1.0e-7)
  plt.ylabel('Density (Log Scale)', fontsize=14, labelpad=10)
  plt.xlabel('Eigenvlaue', fontsize=14, labelpad=10)
  plt.xticks(fontsize=12)
  plt.yticks(fontsize=12)
  plt.axis([np.min(eigenvalues) - 1, np.max(eigenvalues) + 1, None, None])
  plt.tight_layout()
  plt.savefig('example.pdf')


def density_generate(eigenvalues,
                     weights,
                     num_bins=10000,
                     sigma_squared=1e-5,
                     overhead=0.01):

  eigenvalues = np.array(eigenvalues)
  weights = np.array(weights)

  lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + overhead
  lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - overhead

  grids = np.linspace(lambda_min, lambda_max, num=num_bins)
  sigma = sigma_squared * max(1, (lambda_max - lambda_min))

  num_runs = eigenvalues.shape[0]
  density_output = np.zeros((num_runs, num_bins))

  for i in range(num_runs):
    for j in range(num_bins):
      x = grids[j]
      tmp_result = gaussian(eigenvalues[i, :], x, sigma)
      density_output[i, j] = np.sum(tmp_result * weights[i, :])
  density = np.mean(density_output, axis=0)
  normalization = np.sum(density) * (grids[1] - grids[0])
  density = density / normalization
  return density, grids


def gaussian(x, x0, sigma_squared):
  return np.exp(-(x0 - x)**2 /
(2.0 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared)