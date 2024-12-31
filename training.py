import torch
import torch.optim as optim
from fd_gan.generators import BlendGenerator, TransferGenerator
from fd_gan.discriminator import Discriminator
from fd_gan.losses import AdversarialLoss

class FDGANTrainer:
    """
    Trainer class for the FD-GAN model, encompassing both generators and the discriminator.
    """
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.blend_generator = BlendGenerator().to(self.device)
        self.transfer_generator = TransferGenerator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.adv_loss = AdversarialLoss().to(self.device)

        # Optimizers
        self.gen_optimizer = optim.Adam(
            list(self.blend_generator.parameters()) + list(self.transfer_generator.parameters()),
            lr=0.0002, betas=(0.5, 0.999)
        )
        self.dis_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def train_step(self, source_img, reference_img):
        """
        A single training step for the FD-GAN model.
        Args:
            source_img (torch.Tensor): Source input image.
            reference_img (torch.Tensor): Reference input image.
        """
        source_img = source_img.to(self.device)
        reference_img = reference_img.to(self.device)
        # === Train Discriminator ===
        self.dis_optimizer.zero_grad()
        # Real images
        real_pred_spatial, real_pred_freq = self.discriminator(source_img)
        real_loss = self.adv_loss(real_pred_spatial, is_real=True) + self.adv_loss(real_pred_freq, is_real=True)
        # Fake images (generated by generators)
        fake_img, _ = self.blend_generator(torch.randn_like(source_img), source_img, reference_img)
        fake_pred_spatial, fake_pred_freq = self.discriminator(fake_img.detach())
        fake_loss = self.adv_loss(fake_pred_spatial, is_real=False) + self.adv_loss(fake_pred_freq, is_real=False)
        # Total discriminator loss
        dis_loss = real_loss + fake_loss
        dis_loss.backward()
        self.dis_optimizer.step()
        # === Train Generators ===
        self.gen_optimizer.zero_grad()
        # Adversarial loss for generators
        gen_pred_spatial, gen_pred_freq = self.discriminator(fake_img)
        gen_loss = self.adv_loss(gen_pred_spatial, is_real=True) + self.adv_loss(gen_pred_freq, is_real=True)
        gen_loss.backward()
        self.gen_optimizer.step()

        return dis_loss.item(), gen_loss.item()

    def train(self, dataloader, num_epochs=10):
        for epoch in range(num_epochs):
            for i, (source_img, reference_img) in enumerate(dataloader):
                dis_loss, gen_loss = self.train_step(source_img, reference_img)
                if i % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], " f"Dis Loss: {dis_loss:.4f}, Gen Loss: {gen_loss:.4f}")
