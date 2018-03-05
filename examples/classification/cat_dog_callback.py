import numpy as np
import os
from pytorch_toolbox.loop_callback_base import LoopCallbackBase
from pytorch_toolbox.utils import classification_accuracy


class CatDogCallback(LoopCallbackBase):
    """
    Here we define how we handle data during training : visualize, save, etc...
    """
    def __init__(self, update_rate, idx_to_class, file_output_path, reset_files=True):
        """
        In init we can keep persistent data and pass information from the user.
        For example, we handle how we rewrite the log file if needed.
        """
        self.batch_scores = []
        self.count = 0
        self.update_rate = update_rate
        self.idx_to_class = idx_to_class
        self.file_output_path = file_output_path
        if reset_files:
            train_path = os.path.join(self.file_output_path, "training_data.csv")
            valid_path = os.path.join(self.file_output_path, "validation_data.csv")
            if os.path.exists(train_path):
                os.remove(train_path)
            if os.path.exists(valid_path):
                os.remove(valid_path)

    def batch(self, predictions, network_inputs, targets, isvalid=True):
        """
            We have access to the network input/output and the ground truth for each minibatches, in the cat vs dog
            case we compute and keep the classification accuracy on the batch. (will use it in epoch callback)

            show_example we send a picture/label to visdom every x iteration
        """
        score, _ = classification_accuracy(predictions[0].data, targets[0], top_k=(1, 1))
        self.batch_scores.append(score[0])
        self.show_example(network_inputs, predictions)

    def epoch(self, loss, data_time, batch_time, isvalid=True):
        """
            At every epoch we log the loss, times and accuracy :
            show in console
            show with visdom
            log in file
        """
        average_score = sum(self.batch_scores)/len(self.batch_scores)
        self.batch_scores = []
        self.console_print(loss, data_time, batch_time, [average_score], isvalid)
        filename = "validation_data.csv" if isvalid else "training_data.csv"
        self.file_print(os.path.join(self.file_output_path, filename),
                        loss, data_time, batch_time, [average_score])

    def show_example(self, network_input, predictions):
        if self.count % self.update_rate == 0:
            # Unormalize an image and convert it to uint8
            img = network_input[0][0].cpu().numpy()
            std = np.array([58, 57, 57], dtype=np.float32)
            mean = np.array([123, 116, 103], dtype=np.float32)
            std = std[:, np.newaxis, np.newaxis]
            mean = mean[:, np.newaxis, np.newaxis]
            img = img * std + mean
            img = img.astype(np.uint8)

            # log softmax output to class string
            prediction_index = np.argmax(predictions[0][0].data.cpu().numpy())
            prediction_class = self.idx_to_class[prediction_index]


        self.count += 1

