import os
from PIL import Image
import torch
import torchvision
from torchvision import transforms

def load_image(image_path, image_size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def save_image(tensor, save_path):
    """
    Save a tensor as an image.
    """
    tensor = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    tensor = (tensor + 1) / 2  # Convert [-1, 1] range to [0, 1]
    Image.fromarray((tensor * 255).astype('uint8')).save(save_path)

def create_dataloader(image_dir, batch_size=16, image_size=(256, 256)):
    """
    Create a PyTorch DataLoader for training.
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = torchvision.datasets.ImageFolder(image_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
