# pytorch_toolbox

Code to minimize the need to rewrite boilerplate code.

Research tricks : 

- To be able to reproduce my experiments easily, I save (together with the checkpoints) the file containing the network, callback and data loader classes. With this single file and the toolbox installed I can reproduce my experiments anytime without adding thousands of conditions etc.

- I rely heavily on the callback classes to probe information (will be more "natural" in next version) during the training. With python being dynamic, it is easy to "hack" stuff without having to change the toolbox code...

See the Cat vs Dog [example](https://github.com/MathGaron/pytorch_toolbox/tree/develop/examples/classification).

## Code example:

```python
    #
    #   Instantiate models/loaders/etc.
    #
    model = YourNetwork()
    loader_class = YourDataloader

    # Transformation to apply to each inputs
    transformations = [Compose([Resize((128, 128)),
                                ToFloat(),
                                NumpyImage2Tensor(),
                                Normalize(mean=[123, 116, 103], std=[58, 57, 57])])]

    # Optimizer (a list of optimizer can be used)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Training and validation loaders
    train_dataset = loader_class(os.path.join(data_path, "train"), transformations)
    valid_dataset = loader_class(os.path.join(data_path, "valid"), transformations)

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=number_of_core,
                                   pin_memory=use_shared_memory,
                                   drop_last=True,
                                   )

    val_loader = data.DataLoader(valid_dataset,
                                 batch_size=batch_size,
                                 num_workers=number_of_core,
                                 pin_memory=use_shared_memory,
                                 )

    # Instantiate the train loop and train the model.
    train_loop_handler = TrainLoop(model, train_loader, val_loader, optimizer, backend, gradient_clip,
                                   use_tensorboard=use_tensorboard, tensorboard_log_path=tensorboard_path)

    # We can add any number of callbacks to handle data during training methods of the callback are called at
    # different moment in the loop : each batch, each epoch
    train_loop_handler.add_callback([YourCallbacks()])

    # The training loop will optimize the network on the train dataset and run the validation
    train_loop_handler.loop(epochs, output_path, load_best_checkpoint=load_best)

    print("Training Complete")
```

## Todo
- Add tools to probe gradient information
- Right now there is no simple way to change the gradient descent process. e.g. If you want to train a GAN by steping the generator and then the discriminator, the backprop step is hard coded in the loop...
