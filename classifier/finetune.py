import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import EfficientNet_V2_L_Weights
from datasets import load_dataset
import matplotlib.pyplot as plt
import os
import glob
import json
import pickle
from PIL import Image
from torch.amp import autocast, GradScaler

SOCK_CLASS_ID = 806
MAX_SAMPLES = 5000
SOCK_PERCENTAGE = 0.25
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
SOCK_DIR = "/path/to/your/sock/images"  # Replace with your sock images directory
DATASET_CACHE_FILE = "imagenet_subset_cache.pkl"  # File to save downloaded ImageNet samples

class ImageNetDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.Resize(480, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(480),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Handle both HuggingFace dataset items and custom sock items
        if isinstance(item, dict) and 'image' in item:
            # HuggingFace dataset item
            image = item['image']
            label = item['label']
        else:
            # Custom sock item (path, label)
            image_path, label = item
            image = Image.open(image_path).convert('RGB')
        
        image = self.transform(image)
        return image, label

def load_dataset_subset(max_samples, sock_percentage, sock_dir, cache_file):
    print(f"Loading {max_samples} samples ({sock_percentage*100}% socks)...")
    
    # Calculate how many sock images we need
    num_sock_samples = int(max_samples * sock_percentage)
    num_other_samples = max_samples - num_sock_samples
    
    # Load custom sock images from directory
    sock_files = glob.glob(os.path.join(sock_dir, "*.jpg")) + glob.glob(os.path.join(sock_dir, "*.jpeg")) + glob.glob(os.path.join(sock_dir, "*.png"))
    print(f"Found {len(sock_files)} sock images in {sock_dir}")
    
    # Create custom sock dataset items
    custom_sock_samples = [(path, SOCK_CLASS_ID) for path in sock_files[:num_sock_samples]]
    print(f"Using {len(custom_sock_samples)} custom sock images")
    
    # Calculate how many ImageNet samples we need (non-sock)
    remaining_sock_samples_needed = max(0, num_sock_samples - len(custom_sock_samples))
    
    # Check if we have cached ImageNet samples
    imagenet_samples = []
    if os.path.exists(cache_file):
        print(f"Loading cached ImageNet samples from {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                
            cached_sock_samples = cached_data.get('sock_samples', [])
            cached_other_samples = cached_data.get('other_samples', [])
            
            # Use cached samples up to the amount we need
            imagenet_sock_samples = cached_sock_samples[:remaining_sock_samples_needed]
            imagenet_other_samples = cached_other_samples[:num_other_samples]
            
            print(f"Using {len(imagenet_sock_samples)} cached sock samples and {len(imagenet_other_samples)} cached non-sock samples")
            
            # Check if we need more samples
            remaining_sock_samples_needed -= len(imagenet_sock_samples)
            num_other_samples -= len(imagenet_other_samples)
            
            # Add the cached samples we're using
            imagenet_samples = imagenet_sock_samples + imagenet_other_samples
            
        except Exception as e:
            print(f"Error loading cache: {e}")
            # Reset counters if cache loading fails
            remaining_sock_samples_needed = max(0, num_sock_samples - len(custom_sock_samples))
            num_other_samples = max_samples - num_sock_samples
            imagenet_samples = []
    
    # Download more samples if needed
    new_sock_samples = []
    new_other_samples = []
    
    if remaining_sock_samples_needed > 0 or num_other_samples > 0:
        print(f"Downloading {remaining_sock_samples_needed} additional sock samples and {num_other_samples} non-sock samples from ImageNet")
        
        # Stream ImageNet dataset for the rest of the samples
        dataset_stream = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
        
        # Collect non-sock samples and any additional sock samples needed
        imagenet_sock_count = 0
        imagenet_other_count = 0
        
        for sample in dataset_stream:
            # If we have enough samples, stop
            if imagenet_sock_count >= remaining_sock_samples_needed and imagenet_other_count >= num_other_samples:
                break
                
            if sample['label'] == SOCK_CLASS_ID and imagenet_sock_count < remaining_sock_samples_needed:
                new_sock_samples.append(sample)
                imagenet_samples.append(sample)
                imagenet_sock_count += 1
            elif sample['label'] != SOCK_CLASS_ID and imagenet_other_count < num_other_samples:
                new_other_samples.append(sample)
                imagenet_samples.append(sample)
                imagenet_other_count += 1
                    
            total_collected = imagenet_sock_count + imagenet_other_count
            if total_collected % 100 == 0 and total_collected > 0:
                print(f"Collected {total_collected} new ImageNet samples ({imagenet_sock_count} socks, {imagenet_other_count} others)")
        
        # Save newly downloaded samples to cache
        if cache_file and (len(new_sock_samples) > 0 or len(new_other_samples) > 0):
            try:
                # Load existing cache if it exists
                cached_data = {'sock_samples': [], 'other_samples': []}
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                
                # Append new samples
                cached_data['sock_samples'] = cached_data.get('sock_samples', []) + new_sock_samples
                cached_data['other_samples'] = cached_data.get('other_samples', []) + new_other_samples
                
                # Save updated cache
                with open(cache_file, 'wb') as f:
                    pickle.dump(cached_data, f)
                print(f"Cached {len(new_sock_samples)} new sock samples and {len(new_other_samples)} new non-sock samples")
            except Exception as e:
                print(f"Error saving cache: {e}")
    
    # Combine custom sock samples with ImageNet samples
    combined_samples = custom_sock_samples + imagenet_samples
    
    print(f"Final dataset composition:")
    print(f"  - {len(custom_sock_samples)} custom sock images")
    print(f"  - {len([s for s in imagenet_samples if isinstance(s, dict) and s.get('label') == SOCK_CLASS_ID])} ImageNet sock images")
    print(f"  - {len([s for s in imagenet_samples if isinstance(s, dict) and s.get('label') != SOCK_CLASS_ID])} ImageNet non-sock images")
    print(f"  = {len(combined_samples)} total images")
    
    return combined_samples

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize the GradScaler for mixed precision training
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Use autocast for mixed precision (fp16) training
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Scale the loss, perform backward pass, and update weights
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}')
        
        if val_loader:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            sock_correct = 0
            sock_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Use autocast for validation too
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    sock_mask = (labels == SOCK_CLASS_ID)
                    sock_total += sock_mask.sum().item()
                    sock_correct += (predicted[sock_mask] == labels[sock_mask]).sum().item()
            
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = correct / total
            sock_acc = sock_correct / sock_total if sock_total > 0 else 0
            
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Sock Acc: {sock_acc:.4f}')
    
    return model

def main():
    # Load your custom sock images + stream necessary ImageNet samples
    combined_dataset = load_dataset_subset(MAX_SAMPLES, SOCK_PERCENTAGE, SOCK_DIR, DATASET_CACHE_FILE)
    
    # Shuffle the dataset to mix sock and non-sock samples
    import random
    random.shuffle(combined_dataset)
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset = combined_dataset[:train_size]
    val_dataset = combined_dataset[train_size:]
    
    # Create data loaders
    train_loader = DataLoader(
        ImageNetDataset(train_dataset),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        ImageNetDataset(val_dataset),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    model = models.efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    model = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)
    
    torch.save(model.state_dict(), 'efficientnet_v2_finetuned.pt')
    print("Training complete! Model saved.")

if __name__ == "__main__":
    main()