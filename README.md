# pytorch_toolbox

For a full documentation, check the [wiki page](https://github.com/MathGaron/pytorch_toolbox/wiki)

The base usage of pytorch_toolbox is to provide minimalist code to run deep neural network training experiments. To do so, one must **instantiate a Network** architecture, define **how the data is loaded**, and **interact** with the training process. Usually, the user will have to define the main script that will instantiate those three aspects.
The following pseudocode list the main file structure:
```python
# e.g. command line arguments or from a file
params = load_user_params()

# The network architecture and loading code are defined as a class
network = UserNetwork()
dataset = UserDataset()
callbacks = UserCallbacks()

# pytorch specific
data_train_loader = DataLoader(dataset, params)
data_valid_loader = DataLoader(dataset, params)
optimizer = optim.Adam(params)

# setup training
handler = TrainLoop(network, data_train_loader, data_valid_loader, optimizer, params)
handler.add_callback([callbacks])
handler.loop(params)
```

Cat vs Dog training [example](examples/classification).

Network probing [example](examples/probe).
-   Example code to compare activations or latent vectors
