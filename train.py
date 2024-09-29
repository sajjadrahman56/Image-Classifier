import torch
import argparse
from torch import nn, optim
from torchvision import datasets, transforms, models
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on a dataset")
    parser.add_argument('data_dir', type=str, help='Dataset directory')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vit_b_16', help='Model architecture (e.g., vit_b_16)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')
    return parser.parse_args()

def save_checkpoint(model, optimizer, save_dir, arch, hidden_units, output_size, class_to_idx):
    checkpoint = {
        'arch': arch,
        'input_size': 768, 
        'hidden_units': hidden_units,
        'output_size': output_size,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': class_to_idx,
    }
    torch.save(checkpoint, save_dir)

def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Data transforms
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dir = os.path.join(args.data_dir, 'train')
    valid_dir = os.path.join(args.data_dir, 'valid')
    
    train_data = datasets.ImageFolder(train_dir, transform=data_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32)

    # Load model architecture
    if args.arch == 'vit_b_16':
        model = models.vit_b_16(weights='DEFAULT')
        model.heads = nn.Sequential(
            nn.Linear(768, args.hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(args.hidden_units, len(train_data.classes)),
            nn.LogSoftmax(dim=1)
        )
    else:
        raise ValueError(f"Unsupported architecture: {args.arch} \n here using ''vit_b_16' ")
    
    model = model.to(device)
    

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.heads.parameters(), lr=args.learning_rate)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
        model.eval()
        valid_loss = 0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model(inputs)
                batch_loss = criterion(logps, labels)
                valid_loss += batch_loss.item()
                
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        print(f"Epoch {epoch+1}/{args.epochs}.. "
              f"Train loss: {running_loss/len(train_loader):.3f}.. "
              f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
              f"Validation accuracy: {accuracy/len(valid_loader):.3f}")

    # Save checkpoint
    save_checkpoint(model, optimizer, args.save_dir, args.arch, args.hidden_units, len(train_data.classes), train_data.class_to_idx)
    print(f"Model checkpoint saved to {args.save_dir}")

if __name__ == "__main__":
    main()