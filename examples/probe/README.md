# Description
The following scripts provide example on how to probe already trained networks for debugging.


## Examples
#### show_activations.py
Takes an image and a model checkpoint and show all convolution activation maps with matplotlib.

![activation](https://github.com/MathGaron/pytorch_toolbox/raw/develop/examples/probe/images/activations "cat's activation")


#### compare_activations.py
- Define architecure (init + forward)
- Define loss
