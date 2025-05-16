import torch
import os
import json
from torch.utils.data import DataLoader
from models.encoder import ViTEncoder
from models.decoders import SegmentationDecoder
from data.datasets import SegmentationDataset
from eval_utils import compute_metrics

TP, TN, FP, FN = 0, 1, 2, 3
IOU, DICE, ACC, SEN = 0, 1, 2, 3

MODE = ['e2e', 'frozen']
DATA_SIZE = ['2000', '160', '80', '40', '20', '10']

MODEL_PATH = './checkpoints'
DATA_PATH  = '../ISIC2017_resized/'
JSON_PATH  = './checkpoints/eval_json'
BATCH_SIZE = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(JSON_PATH, exist_ok=True)

print("Loading Test Data...")

test_data = DataLoader(
    dataset=SegmentationDataset(
        image_dir= DATA_PATH+"test_images/",
        mask_dir=DATA_PATH+"test_seg_masks/",
    ),
    batch_size=BATCH_SIZE,
)

for ds in DATA_SIZE:
    for m in MODE:
        print()
        print('----------------------------------------')
        model_name = 'seg_' + m + '_' + ds
        model_dict = torch.load(os.path.join(MODEL_PATH, model_name+".pth"))

        print("Loading Model Paramters for {s} ...".format(s=model_name))
        decoder = SegmentationDecoder().to(DEVICE)
        decoder.load_state_dict(model_dict['decoder_state_dict'])
        encoder = ViTEncoder().to(DEVICE)
        encoder.load_state_dict(model_dict['encoder_state_dict'])

        print("Calculating Metrics for {s} ...".format(s=model_name))
        res = []
        for images, masks in test_data:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            _, features = encoder(images)  # get feature map
            output = decoder(features)
            metrics = compute_metrics(output, masks, threshold=0.5, eps=1e-6)
            res.append(metrics)
        res = torch.cat(res, dim=0)

        to_save = {
            'model': model_name,
            'epoch': int(model_dict['epoch']),
            'IoU'  : res[:, IOU].tolist(),
            'Dice' : res[:, DICE].tolist(),
            'Acc'  : res[:, ACC].tolist(),
            'Sen'  : res[:, SEN].tolist(),
        }
        print("Saving results...")
        with open(os.path.join(JSON_PATH, model_name+".json"), "w") as f:
            json.dump(to_save, f)

        print("Mean Jaccard Idx:\t", round(res[:, IOU].mean().item(), 3))
        print("Mean Dice Score :\t", round(res[:, DICE].mean().item(), 3))
        print("Mean Accuracy   :\t", round(res[:, ACC].mean().item(), 3))
        print("Mean Sensitivity:\t", round(res[:, SEN].mean().item(), 3))
        print('----------------------------------------')