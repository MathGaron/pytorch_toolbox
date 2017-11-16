# Description
Sometimes, the train/valid curve is not enough to debug. The following scripts provide example on how to probe already trained networks for debugging.
The following scripts provide easy way to obtain and visualize and compare activation maps.



## Examples
#### [show_activations.py](show_activations.py)
Takes an image and a network checkpoint and show all convolution activation maps with matplotlib.
Note how the activation gets important around the eyes and nose.

![activation](images/activations.png?raw=true "cat's activation")


#### [compare_activations.py](compare_activations.py)
Takes an image and copy it by adding the following occluders (we try to remove the important parts seen in the previous example):

<img src="images/compare_inputs.png?raw=true" width="600" height="300">

Load a pretrained model and compare the activations at each layers : in this case we use the absolute difference between the features,
 the framework will accept any user defined error functions.

<img src="images/compare_activations.png?raw=true">

Note how the last layers have more channels with low error signal. We can see that the output probability is lightly affected even with large occlusion:

<img src="images/compare_predictions.png?raw=true" width="900" height="600" align="center">

#### compare_networks.py
TODO

#### show_tsne_vectors.py
TODO
