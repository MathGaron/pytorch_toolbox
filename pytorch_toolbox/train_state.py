class TrainingState:
    def __init__(self):
        """
        A Structure containing states to be passed to callbacks
        """
        # Will increment after each epochs
        self.current_epoch = 0
        # Will increment after each batch
        self.current_batch = 0
        # last execution was train or eval
        self.training_mode = True
        # The tensorboard logger
        self.tensorboard_logger = None
        # Average time to retrieve the next minibatch
        self.average_data_loading_time = 0
        # Average time to retrieve the next minibatch + process it
        self.average_batch_processing_time = 0

        self.training_data_size = 0
        self.validation_data_size = 0

        # Dictionary containing all average losses returned
        # by the network on the validation
        self.validation_average_losses = {}
        # Dictionary containing all average losses returned
        # by the network on the training (average on full epoch)
        self.training_average_losses = {}

        # Sum of average losses returned by the network on the validation
        self.validation_average_loss = float('Inf')
        # Sum of average losses returned by the network on the training
        self.training_average_loss = float('Inf')

        # Last prediction done by the network
        self.last_prediction = None
        # Last input given to the network
        self.last_network_input = None
        # Last target given to the network
        self.last_target = None

        # The network being optimized
        self.model = None