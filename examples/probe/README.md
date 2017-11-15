# Description
The following scripts provide example on how to probe already trained networks for debugging.


## Examples
#### show_activations.py
Takes an image and a model checkpoint and show all convolution activation maps with matplotlib.
Note how the activation gets important around the eyes and nose.

![activation](images/activations.png?raw=true "cat's activation")


#### compare_activations.py
Takes an image and copy it by adding the following occluders:

<img src="images/compare_inputs.png?raw=true" width="600" height="300" align="center">

Load a pretrained model and compare the activations at each layers

![activation](images/compare_activations.png?raw=true "cat's activation")

Note how the last layers have a low error signal. We can see that the output probability is lightly affected even with large occlusion.

<img src="images/compare_predictions.png?raw=true" width="900" height="600" align="center">

