from generators import BlendGenerator, TransferGenerator
from discriminator import Discriminator
from training import train_fd_gan
import torch.optim as optim

if __name__ == "__main__":
    # Initialize models
    generator_gb = BlendGenerator()
    generator_gt = TransferGenerator()
    discriminator = Discriminator()

    # Optimizers
    optimizer = optim.Adam(
        list(generator_gb.parameters()) + 
        list(generator_gt.parameters()) + 
        list(discriminator.parameters()),
        lr=0.0001
    )

    # Mock DataLoader (Replace with actual DataLoader)
    dataloader = [...]  

    # Train the model
    train_fd_gan(generator_gb, generator_gt, discriminator, dataloader, optimizer, epochs=10)
