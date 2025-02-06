import argparse
import torch
import numpy as np
from PIL import Image
from model_utils import load_checkpoint

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model    
    size = 224, 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Open the image using PIL
    img = Image.open(image_path) 
    
    # Resize the image while preserving aspect ratio
    img.thumbnail((256, 256))  
    
    # Calculate center crop coordinates
    width, height = img.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = left + 224
    bottom = top + 224

    # Crop the center portion of the image
    img = img.crop((left, top, right, bottom))  
    
    # Convert PIL image to NumPy array and normalize pixel values
    img = np.array(img) / 255.0 
    normalize_img = (img - mean) / std  

    # Reorder dimensions for PyTorch (channels first)
    normalize_img = normalize_img.transpose((2, 0, 1)) 
    
    # Convert to a PyTorch tensor and add a batch dimension
    image_tensor = torch.tensor(normalize_img, dtype=torch.float32)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    return image_tensor




def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    #img = Image.open(image_path)
    image = process_image(image_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image = image.to(device)
    
    model.eval()
    
    with torch.no_grad():
        outputs = model(image)
        #output_prob = torch.exp(output)
        
        # Apply the Softmax function to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
    # Extract topk probabilities and indices
    top_probs, top_indices = torch.topk(probabilities, topk, dim=1)
    
    # Convert these tensors into numpy array 
    top_probs = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()
    
    # invert the class_to_idx dictionery for mapping indices to the class labels
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    # Flatten the top_indices array and then iterate to get the class labels 
    top_classes = [idx_to_class[i] for i in top_indices.flatten()]  
    
    return top_probs.tolist(), top_classes

def main():
    parser = argparse.ArgumentParser(description="Image Classifier Prediction Script")
        
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top predictions to return")
    parser.add_argument("--category_names", help="Path to JSON file mapping categories to real names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference if available")
    
    args = parser.parse_args()

    # Load the trained model
    model = load_checkpoint(args.checkpoint)
    
     # Perform prediction
    top_probs, top_classes = predict(args.image_path, model, args.top_k)

    # Load category names if provided
    if args.category_names:
        import json
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_classes = [cat_to_name.get(str(cls), cls) for cls in top_classes]

    # Print results
    for i in range(len(top_probs)):
        print(f"Class: {top_classes[i]}, Probability: {top_probs[i]:.3f}")

if __name__ == "__main__":
    main()
        