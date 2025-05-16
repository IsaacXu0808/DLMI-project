import argparse
from utils import resize_img as resize
import os
import shutil

# Costimizable
DEFAULT_PATH = "../../ISIC2017"
DEFAULT_OUT_DIR= "../../ISIC2017_resized"
DEFAULT_SHAPE = (224, 224)
# Given by ISIC
DEFAULT_PREFIX="ISIC-2017_"
PAR_NAME = {"train": "Training", "test": "Test_v2", "val": "Validation"}
DATA_SIZE = 2000

def process_dir(input_dir, output_dir, size=DEFAULT_SHAPE, is_mask=False):
    os.makedirs(output_dir, exist_ok=True)
    file_extension = ".png" if is_mask else ".jpg"
    for filename in os.listdir(input_dir):
        # Images end with ".jpg", Superpixel masks & Seg masks end with ".png"
        if filename.lower().endswith(file_extension):
            data_id = filename.split("_")[1].split('.')[0]
            in_path = os.path.join(input_dir, filename)
            out_path = os.path.join(output_dir, data_id+file_extension)
            resize(in_path, out_path, size=size, is_mask=is_mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ISIC2017 Data")
    parser.add_argument("--subset", type=str, choices=["train", "val", "test"], required=True)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--input_dir", type=str, default=DEFAULT_PATH)

    args = parser.parse_args()
    input_dir_img = args.input_dir + '/' + DEFAULT_PREFIX + PAR_NAME[args.subset] + "_DATA/"
    input_dir_mask = args.input_dir + '/' + DEFAULT_PREFIX + PAR_NAME[args.subset] + "_Part1_GroundTruth/"

    output_dir_img = args.output_dir + '/' + args.subset + "_images"
    output_dir_mask = args.output_dir + '/' + args.subset + "_seg_masks"
    
    cls_GT_in = args.input_dir + '/' + DEFAULT_PREFIX + PAR_NAME[args.subset] + "_Part3_GroundTruth.csv"
    cls_GT_out = args.output_dir + '/' + args.subset + "_cls_GT.csv"


    print()
    print(f"----------Processing dataset: {PAR_NAME[args.subset]}------------")
    process_dir(input_dir=input_dir_img, output_dir=output_dir_img)
    print(f"Resized image files saved to: {output_dir_img}")
    print()
    process_dir(input_dir=input_dir_mask, output_dir=output_dir_mask, is_mask=True)
    print(f"Resized mask files saved to: {output_dir_mask}")
    print()
    print(f"Classification Ground Truth copied to: {cls_GT_out}")
    shutil.copy(cls_GT_in, cls_GT_out)
    print()