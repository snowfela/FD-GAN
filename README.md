# FD-GAN: Forgery Detection with Generative Adversarial Networks

## Overview
**FD-GAN** is a deep learning module designed for forgery detection and image authenticity recovery. Built using PyTorch, this project implements a generative adversarial network (GAN) that transforms forged images into their authentic counterparts, enabling robust and reliable detection of forgeries in digital content.

This project is ideal for applications such as:
- **E-commerce**: Verifying product authenticity.
- **Content Validation**: Ensuring originality in multimedia content.
- **Image Forensics**: Detecting image manipulation and recovering original content.

---

## Features
- **Forgery-to-Authenticity Transformation**: Recovers authentic images from manipulated (forged) inputs.
- **GAN Architecture**: Uses a two-generator or single-generator architecture depending on requirements.
- **Customizable Pipeline**: Modular design allows users to easily extend or modify the system.
- **Pretrained Models**: Includes pretrained weights for direct inference.
- **PyTorch-Based Implementation**: Leverages PyTorch for flexibility and scalability.

---

## Architecture
FD-GAN consists of:
1. **Generator**: Transforms forged images into authentic images.
2. **Discriminator**: Distinguishes between real (authentic) and generated (forged) images.
3. **Loss Functions**: Combines adversarial, cyclic, and reconstruction losses for robust performance.


