from models.encoder import ViTEncoder
from models.decoders import ReconstructionDecoder
from data.datasets import ReconstructionDataset
from torch.utils.data import DataLoader
import torch

DATA_PATH = "../ISIC2017_resized/"
BATCH_SIZE = 32
CHECKPOINT_PATH = "./checkpoints/"

rec_loader = DataLoader(
    ReconstructionDataset(
        image_dir=DATA_PATH+"train_images/",
        transform=None,
        std=0.3,
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

model = ViTEncoder()
decoder = ReconstructionDecoder()

for i, (images, labels) in enumerate(rec_loader):
    print(f"Batch {i+1}:")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    cls_toke, feature_map = model(images)
    print(f"Predictions shape: {feature_map.shape}")
    print(f"class token shape: {cls_toke.shape}")
    pred = decoder(feature_map)
    print(f"Decoded shape: {pred.shape}")
    break

torch.save(decoder.state_dict(), CHECKPOINT_PATH+"rcdecoder.pth")