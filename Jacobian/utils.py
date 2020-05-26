import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from torchvision import datasets
from torchvision import transforms

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