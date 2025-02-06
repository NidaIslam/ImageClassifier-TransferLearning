from torchvision import datasets, transforms
from PIL import Image
import numpy as np

def load_dataset(data_dir):
    #data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
    'train': transforms.Compose([
             transforms.RandomResizedCrop(size=(224, 224), antialias=True),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
             transforms.RandomResizedCrop(size=(224, 224), antialias=True),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
            transforms.RandomResizedCrop(size=(224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    }

     image_datasets = { 
    'train': torchvision.datasets.ImageFolder(train_dir,transform=data_transforms['train']),
    'valid': torchvision.datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    'test': torchvision.datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    } 

# TODO: Using the image datasets and the trainforms, define the dataloaders
     dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32,     shuffle=True),
    'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=False),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=False),
    }
    
    return dataloaders

