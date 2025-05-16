import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconstructionDecoder(nn.Module):
    def __init__(self, in_channels=768, out_channels=3):
        super().__init__()

        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),  # 14 → 28

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),  # 28 → 56

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),  # 56 → 112

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),  # 112 → 224

            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output ∈ [0, 1]
        )

    def forward(self, x):
        return self.decode(x)

class ClassificationHead(nn.Module):
    def __init__(self, in_channels=768, hidden_dim=256, num_classes=3):
        super().__init__()

        # self.decoder = nn.Sequential(
        #     nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2),  # 14 → 28

        #     nn.Conv2d(256, 64, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        # )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.classifier(x)
    
class SegmentationDecoder(nn.Module):
    def __init__(self, in_channels=768, num_classes=1):
        super(SegmentationDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(32, num_classes, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x = self.decoder(x)
        return x
    

def model_info(model, input_size=(1, 768, 14, 14)):
    """
    Prints model parameter count and estimated model size in MB.
    
    Args:
        model (nn.Module): The model instance.
        input_size (tuple): Input size to the model (default is ViT's output).

    Returns:
        param_count (int): Total number of parameters.
        model_size (float): Model size in MB.
    """
    model = model.cpu()
    
    param_count = sum(p.numel() for p in model.parameters())
    
    model_size = param_count * 4 / (1024 ** 2)  # to MB
    
    print(f"Model Parameters: {param_count:,}")
    print(f"Estimated Model Size: {model_size:.2f} MB")
    
    with torch.no_grad():
        dummy_input = torch.randn(input_size)
        output = model(dummy_input)
        print(f"Output Shape: {output.shape}")
    
    return param_count, model_size