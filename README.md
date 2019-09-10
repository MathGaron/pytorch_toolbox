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

For a full examples see the [Cat vs Dog](https://github.com/MathGaron/pytorch_toolbox/blob/develop/examples/classification/train.py) example.

The main motivation for this package is research, it is thus necessary that the user can:
* Setup an experiment as quickly as possible;
* Manage easily a large amount of experiments;
* Have flexibility while implementing ideas.

Implementation speed is partly provided with the boilerplate code removal : a new experiment is basically created by writing 3 files : Network, Data, Callback. Also a set of parameters is usually needed (e.g. batch size, number of epoch etc.). This also help to manage experimentation more easily, e.g. if an experiment is defined by its 3 files, keeping them will ensure that the experiment can be rerun anytime (as long as pytorch_toolbox has the same version). Finally, the callback system is there to make it possible for the user to execute code during the training (e.g. logging information, processing some metrics, etc.)
