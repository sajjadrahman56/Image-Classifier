import argparse
import torch
from torchvision import models
from PIL import Image
import json
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Predict flower name from an image")
    parser.add_argument('input', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to category names JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage, weights_only=True)
    model = models.vit_b_16(weights='DEFAULT')
    model.heads = torch.nn.Sequential(
        torch.nn.Linear(768, checkpoint['hidden_units']),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(checkpoint['hidden_units'], checkpoint['output_size']),
        torch.nn.LogSoftmax(dim=1)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image):
  
    width, height = image.size
    if width < height:
        new_width = 256
        new_height = int((256 / width) * height)
    else:
        new_height = 256
        new_width = int((256 / height) * width)
    
    # Crop
    left_margin = (image.width - 224) / 2
    bottom_margin = (image.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    image = image.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    # Normalize
    np_image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    np_image = np_image.transpose((2, 0, 1))
    
    return torch.from_numpy(np_image).type(torch.FloatTensor)

def predict(image_path, model, topk, device):
    model.to(device)
    model.eval()
    
    image = Image.open(image_path)
    image = process_image(image)
    image = image.unsqueeze_(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        probs, indices = torch.topk(torch.exp(output), topk)
        
    probs = probs.squeeze().tolist()
    indices = indices.squeeze().tolist()
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in indices]
    
    return probs, classes

def main():
    args = parse_args()
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    model = load_checkpoint(args.checkpoint)
    
    probs, classes = predict(args.input, model, args.top_k, device)
    
    if args.category_names:
        try:
            with open(args.category_names, 'r') as f:
                cat_to_name = json.load(f)
            class_names = [cat_to_name[cls] for cls in classes]
        except FileNotFoundError:
            print(f"Warning: Category names file '{args.category_names}' not found. Using class indices instead.")
            class_names = classes
    else:
        class_names = classes
    
    # Print results
    for i in range(len(probs)):
        print(f"{class_names[i]}: {probs[i]:.3f}")

if __name__ == "__main__":
    main()