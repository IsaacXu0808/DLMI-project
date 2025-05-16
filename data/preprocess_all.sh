#!/bin/bash

python ./preprocess.py --subset train --input_dir ../../ISIC2017 --output_dir ../../ISIC2017_resized
python ./preprocess.py --subset val --input_dir ../../ISIC2017 --output_dir ../../ISIC2017_resized
python ./preprocess.py --subset test --input_dir ../../ISIC2017 --output_dir ../../ISIC2017_resized

echo "All preprocessing complete. Press Enter to exit."
read
