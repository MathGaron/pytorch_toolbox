# Description
Simple classification example using pytorch_toolbox for Kaggle's cat vs dog dataset
This simple example is there to show how to use the train loop, how to define a data loader and how
to define a network architecture.

## Typical files by a user:
#### [train.py](train.py)
- Read the configurations
- Instantiate the network, data loaders, optimizer, transformations
- Instantiate the Trainloop object
- Set the callbacks and run

#### [cat_dog_net.py](cat_dog_net.py)
- Define architecure (init + forward)
- Define loss

#### [cat_dog_loader.py](cat_dog_loader.py)
- Define the dataset indexer
- Define the sample loading function

#### [cat_dog_callback.py](cat_dog_callback.py)
- Define batch callback (visualization, compute scores, extra data to log)
- Define epoch callback (log, visualization)

# Setup
- Download the [Dataset](https://www.kaggle.com/c/dogs-vs-cats/download/train.zip)
- Create a folder called `CatVsDog`
- Extract train.zip inside `CatVsDog`
- Create a folder called `valid` inside `CatVsDog` and move a small percentage of the images inside `train` to `valid`. 
 - You can move a specific amount of random files with the followinf command: 
 
   ``` shuf -n [Number of files to move] -e [PATH to the files to be moved] | xargs -i mv {} [PATH to the dest] ``` 
- run ``` python train.py --output /your/output/path --dataset /your/dataset/path ```
