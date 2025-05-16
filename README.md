# Few-Shot Dermoscopic Image Segmentation with a Frozen Self-Supervised Pretrained Encoder ─── Instructions for Replication

## Step 1. Directory Preparation
(1) Create the following structure of directories.
You can run:
``` 
mkdir {Your Home Folder}/src
mkdir {Your Home Folder}/ISIC2017
mkdir {Your Home Folder}/ISIC2017_resized
```
**{Your Home Folder}**\
├──── src\
├\
├──── ISIC2017\
├\
├──── ISIC2017_resized\
\
(2) Clone this repository to `{Your Home Folder}/src`.


## Step 2. Download Data
(1) Download all the data relaetd to **Task-1** from  [ISIC](https://challenge.isic-archive.com/data/#2017).\
(2) Unzip the data to `{Your Home Folder}/ISIC2017` and **keep the original folder names**.

## Step 3. Data Preprocessing
(1) Diriect the terminal to `{Your Home Folder}/src/data`:
```
cd {Your Home Folder}/src/data
```
(2) Run the following commands to preprocess the data:
```
python ./preprocess.py --subset train --input_dir ../../ISIC2017 --output_dir ../../ISIC2017_resized
python ./preprocess.py --subset val --input_dir ../../ISIC2017 --output_dir ../../ISIC2017_resized
python ./preprocess.py --subset test --input_dir ../../ISIC2017 --output_dir ../../ISIC2017_resized
```

**Note:** \
You may use `{Your Home Folder}/src/data/preprocessing_visualization.ipynb` to check whether the previous steps are completed.

## Step 4. Pre-Training
(1) Diriect the terminal to `{Your Home Folder}/src`:
```
cd {Your Home Folder}/src
```
(2) Run the following commands to pre-train a ViT-based Encoder
```
python .\train_reconstruction.py
```
**Note 1:**\
You may use `{Your Home Folder}/src/reconstruction_eval.ipynb` to check whether the pretraining is successfully completed.

**Note 2:** \
Customizable main function arguments include:
```
'--epochs'     : 'number of training epochs'
'--batch_size' : 'batch size'
'--lr'         : 'learning rate'
...
```
For the other data path arguments, it is **highly recommended** to use the default values.

## Step 5. Training 
(1) Diriect the terminal to `{Your Home Folder}/src`:
```
cd {Your Home Folder}/src
```
(2) Run the following commands to train the segmentation models:
```
python .\train_segmentation.py --mode=e2e --data_size=2000 --epochs=100 --batch_size=50

python .\train_segmentation.py --mode=frozen --data_size=2000 --epochs=100 --batch_size=50

python .\train_segmentation.py --mode=e2e --data_size=160 --epochs=200 --batch_size=10

python .\train_segmentation.py --mode=frozen --data_size=160 --epochs=200 --batch_size=10

python .\train_segmentation.py --mode=e2e --data_size=80 --epochs=300 --batch_size=10

python .\train_segmentation.py --mode=frozen --data_size=80 --epochs=300 --batch_size=10

python .\train_segmentation.py --mode=e2e --data_size=40 --epochs=500 --batch_size=10

python .\train_segmentation.py --mode=frozen --data_size=40 --epochs=500 --batch_size=10

python .\train_segmentation.py --mode=e2e --data_size=20 --epochs=800 --batch_size=10

python .\train_segmentation.py --mode=frozen --data_size=20 --epochs=800 --batch_size=10

python .\train_segmentation.py --mode=e2e --data_size=10 --epochs=1200 --batch_size=5

python .\train_segmentation.py --mode=frozen --data_size=10 --epochs=1200 --batch_size=5
```

**Note:** \
Customizable main function arguments include:
```
'--epochs'     : number of training epochs
'--batch_size' : batch size
'--lr'         : learning rate
'--mode'       : training mode of the encoder ('e2e' or 'frozen')
'--data_size'  : number of training images
...
```
For the other data path arguments, it is **highly recommended** to use the default values.

## Step 6. Evaluation
(1) Diriect the terminal to `{Your Home Folder}/src`:
```bash
cd {Your Home Folder}/src
```
(2) Run the following commands to compute the metrics for each model using the test dataset:
```
python .\eval_metrics.py
```
(3) Run the following commands to generate all loss curve plots comparing the two modes of training:
```
python .\loss_curve.py
```

(4) Use `{Your Home Folder}/src/metrics_visualization.ipynb` to generate histograms of metrics for all models.

(5) Use `{Your Home Folder}/src/seg_visualization.ipynb` to generate visualization of segmentation models' outputs on the same input image for comparison and case study.

**Note:**\
All the evaluation programs are hard-coded for only **DEFAULT**  arguments in the training. You may need to change some values to accommodate them to your needs.
