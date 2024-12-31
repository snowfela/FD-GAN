import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys

# Add the backend module to the path
sys.path.append("C:/Users/sivan/Downloads/FinalYearProject/e-commerce-store/src/backend")
from fd_gan.generators import BlendGenerator, TransferGenerator
from fd_gan.discriminator import Discriminator
from fd_gan.losses import AdversarialLoss

# Preprocessing function for real images
def preprocess_image(image_path, image_size=(256, 256)):
    """
    Preprocess an image: load it, resize it, normalize it, and convert to a tensor.
    Args:
        image_path (str): Path to the input image.
        image_size (tuple): Target size for the image (width, height).
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB format
    return transform(image).unsqueeze(0)  # Add batch dimension

def show_image(tensor, title="Image"):
    """
    Utility function to display a tensor as an image.
    Converts the tensor from a range of [-1, 1] to [0, 1] for visualization.
    """
    tensor = tensor.squeeze(0)  # Remove batch dimension
    tensor = tensor.permute(1, 2, 0).detach().cpu().numpy()  # Convert to HWC format for visualization
    tensor = (tensor + 1) / 2  # Convert from [-1, 1] to [0, 1]
    plt.imshow(np.clip(tensor, 0, 1))  # Clip values for valid range
    plt.title(title)
    plt.axis('off')
    plt.show()

def test_discriminator(image_path, adversarial_loss):
    """
    Test the discriminator to classify whether the image is real or fake.
    If fake, use the original image to recover the original.
    """
    print("Testing Discriminator...")

    # Load and preprocess the forged image
    image = preprocess_image(image_path).to(device)
    discriminator = Discriminator().to(device)

    # Run through the discriminator to check if the image is fake or real
    spatial_pred, frequency_pred = discriminator(image)

    # Assuming the discriminator outputs a single score indicating real or fake
    is_fake = spatial_pred.mean() < 0  # Example: if the output is negative, consider it fake
    print(f"Discriminator prediction: {'Fake' if is_fake else 'Real'}")

    if is_fake:
        print("Image is fake. Recovering original image...")
        # Assuming you have the original image for recovery
        original_image_path = "C:/Users/sivan/Downloads/FinalYearProject/e-commerce-store/src/backend/fd_gan/org.jpg"  # Path to the original image
        original_image = preprocess_image(original_image_path).to(device)

        # Use a generator for recovery
        generator = BlendGenerator(num_masks=4, feature_channels=256).to(device)  # Example: Use the Blend Generator
        feature_map = torch.rand((1, 256, 64, 64)).to(device)  # Simulate feature map
        recovered_image, _ = generator(feature_map, image, original_image)

        # Display the recovered image
        show_image(recovered_image[0], title="Recovered Image")
    else:
        print("Image is real. No need for recovery.")

def test_blend_generator(forged_image_path, adversarial_loss):
    print("Testing Blend Generator...")

    # Use the generator for the forged image recovery if needed
    # Use the original image as the reference for recovery
    original_image_path = "C:/Users/sivan/Downloads/FinalYearProject/e-commerce-store/src/backend/fd_gan/org.jpg"  # Path to the original image

    # Load images
    forged_image = preprocess_image(forged_image_path).to(device)
    original_image = preprocess_image(original_image_path).to(device)

    # Use the BlendGenerator to recover the image
    generator = BlendGenerator(num_masks=4, feature_channels=256).to(device)
    feature_map = torch.rand((1, 256, 64, 64)).to(device)  # Simulate feature map
    recovered_image, _ = generator(feature_map, forged_image, original_image)

    # Show the recovered image
    show_image(recovered_image[0], title="Recovered Image")

    # Test adversarial loss for real and fake images
    discriminator = Discriminator().to(device)
    real_pred_spatial, real_pred_freq = discriminator(original_image)
    fake_pred_spatial, fake_pred_freq = discriminator(recovered_image)

    real_loss = adversarial_loss(real_pred_spatial, is_real=True) + adversarial_loss(real_pred_freq, is_real=True)
    fake_loss = adversarial_loss(fake_pred_spatial, is_real=False) + adversarial_loss(fake_pred_freq, is_real=False)

    print(f"Real Loss: {real_loss.item():.4f}")
    print(f"Fake Loss: {fake_loss.item():.4f}")

    # Verify shapes match
    assert recovered_image.shape == forged_image.shape, "Recovered image size mismatch"
    print("Blend Generator passed!")

if __name__ == "__main__":
    # Initialize the device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize adversarial loss (vanilla mode)
    adversarial_loss = AdversarialLoss(mode="vanilla").to(device)

    # Replace this path with your forged image
    forged_image_path = "C:/Users/sivan/Downloads/FinalYearProject/e-commerce-store/src/backend/fd_gan/forg.jpg"

    # Test the Discriminator
    test_discriminator(forged_image_path, adversarial_loss)

    # Test the Blend Generator for recovering the image if it's fake
    test_blend_generator(forged_image_path, adversarial_loss)
