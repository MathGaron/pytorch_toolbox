# Description
Sometimes, the train/valid curve is not enough to debug. The following scripts provide
example on how to probe an already trained network for debugging.



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

#### [compare_networks.py](compare_networks.py)
Compare activations between two networks.

#### [show_embedding.py](show_embedding.py)
Compare embeddings between the dog and cat class.

#### show_grad.py
TODO : similar to https://github.com/experiencor/deep-viz-keras?

#### [show_runtime.py](show_runtime.py)
Compute the network's runtime w.r.t minibatch size and plot.
