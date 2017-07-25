# Description
Simple classification example using pytorch_toolbox for Kaggle's cat vs dog dataset
This simple example is there to show how to use the train loop, how to define a data loader and how
to define a network architecture.

# Dataset
- Download the [Dataset](https://www.kaggle.com/c/dogs-vs-cats/download/train.zip)
- Create a folder called `CatVsDog`
- Extract train.zip inside `CatVsDog`
- Create a folder called `valid` inside `CatVsDog` and move a small percentage of the images inside `train` to `valid`. 
- Copy `train_config.yml.dst` to `train_config.yml` and set the proper paths/parameters.
- run ``` python train.py train_config.yml ```
