# TODO: Save the checkpoint
import torch

def save_checkpoint(model, optimizer, classifier, epochs, train_dataset, save_path='flower_classifier_checkpoint.pth'):
    """
    Save the trained model checkpoint.

    Parameters:
        model (torch.nn.Module): The trained model to save.
        optimizer (torch.optim.Optimizer): The optimizer used during training.
        classifier (torch.nn.Module): The classifier attached to the model.
        epochs (int): Number of training epochs.
        train_dataset (torchvision.datasets.ImageFolder): The training dataset to retrieve class indices.
        save_path (str): Path to save the checkpoint file.
    """
    # Save class_to_idx mapping
    model.class_to_idx = train_dataset.class_to_idx

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'classifier': classifier,
        'epochs': epochs,
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx
    }

    torch.save(checkpoint, save_path)
    print(f'Checkpoint saved to {save_path}')



# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_checkpoint(filepath, arch):
    """
    Load a trained model checkpoint for inference only.
    Args: Path to the saved checkpoint file.
    Returns: Trained model ready for inference.
    """
    
    # Load the checkpoint
    checkpoint = torch.load(filepath, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
     # Load the pre-trained model architecture and Adjust to your model if different
    if arch == 'vgg16':
        model = models.vgg16(pretrained = True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained = True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        raise ValueError(f'Unsupported architecture: {arch}')
    
    # Load the custom classifier from the checkpoint
    model.classifier = checkpoint['classifier']
    
    # Load the model's state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load the class-to-index mapping
    model.class_to_idx = checkpoint['class_to_idx']
    
    # Set the model to evaluation mode for inference
    model.eval()
    
    return model
 
