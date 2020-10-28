#!/bin/bash
echo "Download dataset."
wget https://www.dropbox.com/s/59fx0r5fuqmuo3j/class3_images.zip
unzip class3_images.zip
rm class3_images.zip

echo "Generate training dataset."
python gen_training_data.py --pathfile list.txt --dataset class3 --size 48 --keepaspect yes


