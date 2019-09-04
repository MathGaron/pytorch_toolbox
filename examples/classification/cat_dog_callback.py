import os

from pytorch_toolbox.loop_callback_base import LoopCallbackBase
from pytorch_toolbox.train_state import TrainingState
from pytorch_toolbox.utils import classification_accuracy


class CatDogCallback(LoopCallbackBase):
    """
    Here we define how we handle data during training : visualize, save, etc...
    """
    def __init__(self, file_output_path, reset_files=True):
        """
        In init we can keep persistent data and pass information from the user.
        For example, we handle how we rewrite the log file if needed.
        """
        super().__init__()
        self.file_output_path = file_output_path
        if reset_files:
            self.output_path = file_output_path
            train_path = os.path.join(self.file_output_path, "training_data.csv")
            valid_path = os.path.join(self.file_output_path, "validation_data.csv")
            if os.path.exists(train_path):
                os.remove(train_path)
            if os.path.exists(valid_path):
                os.remove(valid_path)

    def batch(self, state: TrainingState):
        """
            We have access to the network input/output and the ground truth for each minibatches, in the cat vs dog
            case we compute and keep the classification accuracy on the batch. (will use it in epoch callback)
        """
        # At every 10 minibatch, log the gradient going in the fc1 layer
        if state.current_batch % 10:
            grad_log = state.model.grad_data["fc1"]

        score, _ = classification_accuracy(state.last_prediction[0].data, state.last_target[0], top_k=(1, 1))

        # Log information about batch, an history will be kept of each score and the average is computed for each epoch
        self.batch_logger["Accuracy"] = score

    def epoch(self, state: TrainingState):
        """
            At every epoch we log the loss, times and accuracy :
            show in console
            show with visdom
            log in file
        """
        self.print_batch_data(state)
        self.save_epoch_data(self.output_path, state)

