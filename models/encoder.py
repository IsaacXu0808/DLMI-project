import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

class ViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        self.patch_embed = vit.conv_proj          # [B, 768, 14, 14]
        self.encoder = vit.encoder
        self.norm = self.encoder.ln

        self.cls_token = vit.class_token            # [1, 1, 768]
        self.pos_embedding = self.encoder.pos_embedding  # [1, 197, 768]

    def forward(self, x):
        """
        Args:
            x: input image [B, 3, 224, 224]
        Returns:
            cls_token: [B, 768]
            feature_map: [B, 768, 14, 14]
        """
        B = x.shape[0]
        x = self.patch_embed(x)                   # [B, 768, 14, 14]
        C, H, W = x.shape[1:]

        x = x.flatten(2).transpose(1, 2)          # [B, 196, 768]

        # Add CLS token and position embedding
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, 768]
        x = torch.cat((cls_token, x), dim=1)         # [B, 197, 768]
        x = x + self.pos_embedding

        x = self.encoder(x)                     # [B, 197, 768]
        x = self.norm(x)

        # Separate CLS token and patch tokens
        cls_token_out = x[:, 0]                 # [B, 768]
        patch_tokens = x[:, 1:]                 # [B, 196, 768]
        feature_map = patch_tokens.transpose(1, 2).reshape(B, C, H, W)  # [B, 768, 14, 14]

        return cls_token_out, feature_map
