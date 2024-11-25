import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data_loader import MelanomaDataset  # Import your dataset class
from src.evaluate import evaluate_model  # Import your evaluation function
from src.model import MultimodalCNN  # Import your model class
from src.train import train_model  # Import your training function

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load datasets
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Optional: Split train_data into train and validation sets
train_set, val_set = train_test_split(train_data, test_size=0.15, stratify=train_data['target'], random_state=42)
image_size = (260,260)
# Define transformations
train_transform = transforms.Compose([
    transforms.RandomVerticalFlip(p=0.5),  # Vertical Flip
    transforms.RandomHorizontalFlip(p=0.5),  # Horizontal Flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Random Brightness and Contrast
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=5),  # Gaussian Blur
        transforms.RandomErasing(scale=(0.375, 0.375), ratio=(0.3, 3.3))  # Cutout equivalent
    ], p=0.7),
    transforms.Resize(image_size),  # Resize to target size
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

val_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets and data loaders
train_dataset = MelanomaDataset(train_set['image_names'], train_set['metadata'], train_set['target'],
                                transform=train_transform)
val_dataset = MelanomaDataset(val_set['image_names'], val_set['metadata'], val_set['target'], transform=val_transform)
test_dataset = MelanomaDataset(test_data['image_names'], test_data['metadata'], test_data['target'],
                               transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model
model = MultimodalCNN().to(device)

# Train the model
best_model_path = 'best_CNN.pth'
train_model(model, train_loader, val_loader, best_model_path)

# Evaluate the model on the test set
evaluate_model(model, test_loader)

print("Training and evaluation completed.")
