# Description
Simple classification example using pytorch_toolbox for Kaggle's cat vs dog dataset

# Dataset
Download Dataset from : https://www.kaggle.com/c/dogs-vs-cats/download/train.zip
Extract train in a folder (example CatVsDog)
Create valid in CatVsDog and move a small percentage of train images to valid
Copy train_config.yml.dst to train_config.yml and set the proper paths/parameters
run : python train.py train_config.yml