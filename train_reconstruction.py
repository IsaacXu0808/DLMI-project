import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import os
from models.encoder import ViTEncoder
from models.decoders import ReconstructionDecoder
from data.datasets import ReconstructionDataset
import torchvision.transforms as T
from tqdm import tqdm
import json

# ----------------------------
# Command line arguments
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=32, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=40, help='batch size')
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
parser.add_argument('--data_path', type=str, default="../ISIC2017_resized/", help='path to dataset')
parser.add_argument('--save_dir', type=str, default="./checkpoints/", help='where to save models and results')
parser.add_argument('--image_dir', type=str, default="./checkpoints/train_images", help='directory of training images')
parser.add_argument('--json_dir', type=str, default="./checkpoints/train_json", help='directory of training images')
args = parser.parse_args()

# ----------------------------
# Setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(args.image_dir, exist_ok=True)
os.makedirs(args.json_dir, exist_ok=True)

import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()


# Dataset & Dataloader
train_loader = DataLoader(
    ReconstructionDataset(
        image_dir=args.data_path+"train_images/",
        transform=None,
        std=0.1,
    ),
    batch_size=args.batch_size,
    shuffle=True,
)
val_loader = DataLoader(
    ReconstructionDataset(
        image_dir=args.data_path+"val_images/",
        transform=None,
        std=0.1,
    ),
    batch_size=args.batch_size,
    shuffle=True,
)

# Model
encoder = ViTEncoder().to(device)
decoder = ReconstructionDecoder().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=args.lr)

# ----------------------------
# Training Loop
# ----------------------------
epoch_losses = []
val_losses = []

for epoch in range(1, args.epochs + 1):
    encoder.train()
    decoder.train()
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")

    for noisy, clean in pbar:
        noisy, clean = noisy.to(device), clean.to(device)

        _, features = encoder(noisy)  # get feature map
        output = decoder(features)

        loss = criterion(output, clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * noisy.size(0)
        pbar.set_postfix(loss=loss.item())


    # Calculate validation loss
    encoder.eval()
    decoder.eval()
    val_loss = 0.0
    for val_noisy, val_clean in val_loader:
        val_noisy, val_clean = val_noisy.to(device), val_clean.to(device)
        with torch.no_grad():
            _, val_features = encoder(val_noisy)
            val_output = decoder(val_features)
            loss = criterion(val_output, val_clean)
            val_loss += loss.item() * val_noisy.size(0)
    encoder.train()
    decoder.train()    
            

    avg_loss = epoch_loss / len(train_loader.dataset)
    avg_val_loss = val_loss / len(val_loader.dataset)
    epoch_losses.append(avg_loss)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch}: Avg Loss = {avg_loss:.6f}", f"Val Loss = {avg_val_loss:.6f}")

    # Save checkpoint
    if avg_val_loss <= min(val_losses, default=float('inf')):
        print(f"Validation loss improved, saving model...")
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': avg_loss
        }, os.path.join(args.save_dir, f"reconstruction.pth"))

# ----------------------------
# Save loss data and plot
# ----------------------------
loss_data = {
    "epoch": list(range(1, args.epochs + 1)),
    "train_loss": epoch_losses,
    "val_loss": val_losses
}
with open(os.path.join(args.json_dir, "reconstruction_loss_data.json"), "w") as f:
    json.dump(loss_data, f)

plt.figure()
plt.plot(range(1, args.epochs + 1), epoch_losses, marker='o')
plt.plot(range(1, args.epochs + 1), val_losses, marker='o')
plt.legend(["Train Loss", "Validation Loss"])
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.savefig(os.path.join(args.image_dir, "loss_vs_epoch.png"))