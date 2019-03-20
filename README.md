# pytorch_toolbox

General tools for using pytorch :
- Reduce the need for boilerplate code
- Visualization tools

The main goal if this simple package is to remove the boilerplate code needed for training the network and leave flexibility to the user.

Cat vs Dog training [example](examples/classification).

Network probing [example](examples/probe).

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