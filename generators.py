import torch
import torch.nn as nn
import torch.nn.functional as F

class BlendGenerator(nn.Module):
    def __init__(self, num_masks=8, feature_channels=256):
        super(BlendGenerator, self).__init__()
        self.num_masks = num_masks
        self.mask_filters = nn.ModuleList([
            nn.Conv2d(feature_channels, 1, kernel_size=1) for _ in range(num_masks)
        ])

    def forward(self, feature_map, source_img, reference_img):
        masks = [torch.sigmoid(f(feature_map)) for f in self.mask_filters]
        blended_mask = torch.clip(sum(masks) / self.num_masks, 0, 1)
        
        # Resize blended_mask to source_img dimensions
        if blended_mask.shape[2:] != source_img.shape[2:]:
            blended_mask = F.interpolate(
                blended_mask, size=source_img.shape[2:], mode='bilinear', align_corners=False
            )
        blended_mask = blended_mask.repeat(1, 3, 1, 1)
        blended_image = blended_mask * reference_img + (1 - blended_mask) * source_img
        return blended_image, blended_mask

class TransferGenerator(nn.Module):
    def __init__(self, style_channels=256, content_channels=256):
        super(TransferGenerator, self).__init__()
        self.style_encoder = nn.Sequential(
            nn.Conv2d(3, style_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.content_encoder = nn.Sequential(
            nn.Conv2d(3, content_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(content_channels, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, source_img, reference_img):
        style_features = self.style_encoder(reference_img)
        content_features = self.content_encoder(source_img)
        combined_features = content_features + style_features
        synthesized_image = self.decoder(combined_features)
        return synthesized_image
