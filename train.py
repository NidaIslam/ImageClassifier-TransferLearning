import argparse
import torch
import numpy as np
from PIL import Image

def train_model():
    # Check GPU availability
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Load data using the function from data_utils.py
    dataloaders = load_data(args.data_directory)

    if args.arch == 'vgg16':
        model = models.vgg16(pretrained = True)
    elif args.arch == 'densenet121':
        model = models.densenet121(pretrained = True)
    elif args.arch == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        raise ValueError("Unsupported architecture. Choose from 'vgg16', 'densenet121', or 'alexnet'.")
        

    for param in model.parameters():
        param.requires_grad= False
    
    input_size = model.classifier[0].in_features
    output_size = 102

    classifier = nn.Sequential(
        nn.Linear(input_size,512),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(512,256),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(256,output_size)
    )
 

    model.classifier = classifier
    model = model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()  # Use criterion instead of loss to avoid confusion
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Training parameters
    epochs = 5
    train_losses, valid_losses = [], []

    for epoch in range(epochs):
        training_loss = 0
    
        model.train()
        for images, labels in dataloaders['train']:
            images, labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
        
            loss.backward()
            optimizer.step()
        
            training_loss += loss.item()
    
        # Ensure the model is in evaluation mode
        model.eval()
    
        # Initialize counters for validation accuracy and loss
        validation_correct = 0
        validation_total = 0
        valid_loss = 0.0

        # Disable gradient tracking for validation
        with torch.no_grad():
            for images, labels in dataloaders['valid']:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                _, prediction = torch.max(outputs.data, 1)
                validation_total += labels.size(0)
                validation_correct += (prediction == labels).sum().item()

        # Calculate validation accuracy
        validation_accuracy = validation_correct / validation_total * 100

        # Record metrics
        train_losses.append(training_loss / len(dataloaders['train']))
        valid_losses.append(valid_loss / len(dataloaders['valid']))

        # Print metrics for each epoch
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {train_losses[-1]:.3f}.. "
              f"Validation loss: {valid_losses[-1]:.3f}.. "
              f"Validation accuracy: {validation_accuracy:.2f}%")

    # Save checkpoint using function from model_utils.py
    save_checkpoint(model, optimizer, classifier, args.epochs, dataloaders['train'].dataset, args.save_dir)


def main():
    parser = argparse.ArgumentParser(description="Neural Network Training Script")
    
    parser.add_argument("data_directory", type=str, help="Path to datasets")
    parser.add_argument("--save_dir", type=str, default='checkpoints/', help="Path to save checkpoint")
    parser.add_argument("--arch", type=str, default='vgg16', help="name of model architecture")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate for model training")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference if available")
    
    args = parser.parse_args()
    train_model(args)
    
if __name__ == "__main__":
    main()