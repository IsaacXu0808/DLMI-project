import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from models.encoder import ViTEncoder
from models.decoders import SegmentationDecoder
from data.datasets import SegmentationDataset
import torchvision.transforms as T
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import torchvision

# ----------------------------
# Command line arguments
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=128, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=40, help='batch size')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--data_path', type=str, default="../ISIC2017_resized/", help='path to dataset')
parser.add_argument('--save_dir', type=str, default="./checkpoints/", help='where to save models and results')
parser.add_argument('--json_dir', type=str, default="./checkpoints/train_json", help='directory of training info')
parser.add_argument('--image_dir', type=str, default="./checkpoints/train_images", help='directory of training images')
parser.add_argument('--mode', type=str, default="./checkpoints/train_images", choices=['e2e', 'frozen'])
parser.add_argument('--data_size', type=int, default=100, choices=[2000, 160, 80, 40, 20, 10], help='size of training data')
args = parser.parse_args()

# ----------------------------
# Setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(args.json_dir, exist_ok=True)

import random
import numpy as np

def set_seed(seed=66):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# Dataset & Dataloader
transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomVerticalFlip(),
    # torchvision.transforms.RandomRotation(45),
])


train_loader = DataLoader(
    dataset=SegmentationDataset(
        image_dir= args.data_path+"train_images/",
        mask_dir=args.data_path+"train_seg_masks/",
        size=args.data_size,
        transform=transform,
    ),
    batch_size=args.batch_size,
    shuffle=True,
)
print("Training Data Size:", len(train_loader.dataset))
val_loader = DataLoader(
    dataset=SegmentationDataset(
        image_dir= args.data_path+"val_images/",
        mask_dir=args.data_path+"val_seg_masks/",
        size=args.data_size // 2 if args.data_size != 2000 else None,
    ),
    batch_size=args.batch_size,
    shuffle=True,
)
print("Validation Data Size:", len(val_loader.dataset))

# Model
encoder = ViTEncoder().to(device)
if args.mode == "frozen":
    # Load the pre-trained encoder weights
    encoder.load_state_dict(torch.load(os.path.join(args.save_dir, "reconstruction.pth"))['encoder_state_dict'])
    # Freeze the encoder parameters
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
elif args.mode == "e2e":
    encoder.train()
decoder = SegmentationDecoder().to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
params = list(encoder.parameters()) + list(decoder.parameters()) if args.mode == "e2e" else list(decoder.parameters())
optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)

# ----------------------------
# Training Loop
# ----------------------------
epoch_losses = []
val_losses = []

# Start time
import time
start_time = time.time()
best_val_loss = float('inf')

for epoch in range(1, args.epochs + 1):
    if args.mode == "e2e": encoder.train()
    decoder.train()
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch", disable=bool(epoch % 50))

    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        _, features = encoder(images)  # get feature map
        output = decoder(features)

        loss = criterion(output, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * masks.size(0)
        pbar.set_postfix(loss=loss.item())

    # Calculate validation loss
    if args.mode == "e2e": encoder.eval()
    decoder.eval()
    val_loss = 0.0
    for val_images, val_masks in val_loader:
        val_images, val_masks = val_images.to(device), val_masks.to(device)
        with torch.no_grad():
            _, val_features = encoder(val_images)
            val_output = decoder(val_features)
            loss = criterion(val_output, val_masks)
            val_loss += loss.item() * val_images.size(0)            

    avg_loss = epoch_loss / len(train_loader.dataset)
    avg_val_loss = val_loss / len(val_loader.dataset)
    epoch_losses.append(avg_loss)
    val_losses.append(avg_val_loss)

    # Save checkpoint
    # if avg_val_loss <= min(val_losses, default=float('inf')):
    if avg_val_loss <= best_val_loss:
        print(f"Epoch {epoch}: Avg Loss = {avg_loss:.6f}", f"Val Loss = {avg_val_loss:.6f}")
        print(f"Validation loss improved, updating best model...")
        best_val_loss = avg_val_loss
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'epoch': epoch,
            'loss': avg_val_loss
        }, os.path.join(args.save_dir, "seg_"+args.mode+"_"+str(args.data_size)+".pth"))


# end time
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

# ----------------------------
# Save loss data
# ----------------------------
loss_data = {
    "epoch": args.epochs,
    "train_loss": epoch_losses,
    "val_loss": val_losses,
    "train_time": end_time - start_time,
    "data_size": args.data_size,
    "mode": args.mode,
}
with open(os.path.join(args.json_dir, "seg_"+args.mode+"_"+str(args.data_size)+".json"), "w") as f:
    json.dump(loss_data, f)
plt.figure()
plt.plot(range(1, args.epochs + 1), epoch_losses)
plt.plot(range(1, args.epochs + 1), val_losses)
plt.legend(["Train Loss", "Validation Loss"])
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.savefig(os.path.join(args.image_dir, "seg_"+args.mode+"_"+str(args.data_size)+".png"))