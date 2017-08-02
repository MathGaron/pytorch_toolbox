""" Train loop boilerplate code

    Uses preinstantiated data loaders, network, loss and optimizer to train a model.

    - Supports multiple inputs
    - Supports multiple outputs

"""
import shutil

from pytorch_toolbox.utils import AverageMeter
import time
import torch
from tqdm import tqdm
import numpy as np
import os


class TrainLoop:
    def __init__(self, model, train_data_loader, valid_data_loader, criterions, optimizer, backend):
        """
        See examples/classification/train.py for usage

        :param model:               Any nn.module (it is the network model to train)
        :param train_data_loader:   Any torch dataloader for training data
        :param valid_data_loader:   Any torch dataloader for validation data
        :param criterions:          List of Criterions (nn.module) depending on the number of outputs of
                                    the network [criterion1, criterion2, ...]
                                    #TODO Does not handle n criterions per output...
        :param optimizer:           Any torch optimizer
        :param backend:             cuda | cpu
        """
        self.train_data = train_data_loader
        self.valid_data = valid_data_loader
        self.criterions = criterions
        self.optim = optimizer
        self.backend = backend
        self.model = model

        self.prediction_callbacks = []

        if backend == "cuda":
            self.model = self.model.cuda()
            for i in range(len(self.criterions)):
                self.criterions[i] = self.criterions[i].cuda()

    @staticmethod
    def setup_loaded_data(data, target, backend):
        """
        Will make sure that the targets are formated as list in the right backend
        :param data:
        :param target:
        :param backend: cuda | cpu
        :return:
        """
        if not isinstance(data, list):
            data = [data]

        if not isinstance(target, list):
            target = [target]

        if backend == "cuda":
            for i in range(len(data)):
                data[i] = data[i].cuda()
            for i in range(len(target)):
                target[i] = target[i].cuda()
        else:
            for i in range(len(data)):
                data[i] = data[i].float()
            for i in range(len(target)):
                target[i] = target[i].long()
        return data, target

    @staticmethod
    def to_autograd(data, target):
        """
        Converts data and target to autograd Variable
        :param data:
        :param target:
        :return:
        """
        target_var = []
        data_var = []
        for i in range(len(data)):
            data_var.append(torch.autograd.Variable(data[i]))
        for i in range(len(target)):
            target_var.append(torch.autograd.Variable(target[i]))
        if len(data_var) == 1:
            data_var = data_var[0]
        return data_var, target_var

    def predict(self, data_variable):
        """
        compute prediction
        :param data_variable: tuple containing the network's input data
        :return:
        """
        y_pred = self.model(data_variable)
        if not isinstance(y_pred, tuple):
            y_pred = (y_pred,)
        return y_pred

    def compute_loss(self, predictions, target_variable):
        """
        Compute the criterion w.r.t output and target list

        :param predictions:     list of network outputs
        :param target_variable: list of targets
        :return:
        """
        loss = None
        for i, crit in enumerate(self.criterions):
            if loss is None:
                loss = crit(predictions[i], target_variable[i])
            else:
                loss += crit(predictions[i], target_variable[i])
        return loss

    def add_prediction_callback(self, func):
        """
        add a prediction callback that takes as input the predictions and targets and return
        a score that will be displayed

        GOTCHA: There is a gotcha here, the callback will get the list of prediction and the list of target for every
                minibatch iterations.

        :param func:
        :return:
        """
        if isinstance(func, list):
            for cb in func:
                self.prediction_callbacks.append(cb)
        else:
            self.prediction_callbacks.append(func)

    def train(self):
        """
        Minibatch loop for training. Will iterate through the whole dataset and backprop for every minibatch

        It will keep an average of the computed losses and every score obtained with the user's callbacks.

        The information is displayed on the console

        :return: averageloss, [averagescore1, averagescore2, ...]
        """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        self.model.train()

        scores = []
        for i in range(len(self.prediction_callbacks)):
            scores.append(AverageMeter())

        for i, (data, target) in tqdm(enumerate(self.train_data), total=len(self.train_data)):
            data_time.update(time.time() - end)
            if not isinstance(target, list):
                target = [target.view(-1)]
            data, target = self.setup_loaded_data(data, target, self.backend)
            data_var, target_var = self.to_autograd(data, target)
            y_pred = self.predict(data_var)
            loss = self.compute_loss(y_pred, target_var)
            losses.update(loss.data[0], data[0].size(0))

            for callback, acc in zip(self.prediction_callbacks, scores):
                score = callback(y_pred, target)
                acc.update(score, data[0].size(0))

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            batch_time.update(time.time() - end)
            end = time.time()
        print(' Train\t || Loss: {:.3f} | Load Time {:.3f}s | Batch Time {:.3f}s'.format(losses.avg,
                                                                                         data_time.avg,
                                                                                         batch_time.avg))
        for i, acc in enumerate(scores):
            print('\t || Acc {}: {:.3f}'.format(i, acc.avg))
        return losses, scores

    def validate(self):
        """
        Validation loop (refer to train())

        Only difference is that there is no backpropagation..

        #TODO: It repeats mostly train()'s code...

        :return:
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        scores = []
        for i in range(len(self.prediction_callbacks)):
            scores.append(AverageMeter())

        self.model.eval()

        end = time.time()
        for i, (data, target) in enumerate(self.valid_data):
            if not isinstance(target, list):
                target = [target.view(-1)]
            data, target = self.setup_loaded_data(data, target, self.backend)
            data_var, target_var = self.to_autograd(data, target)
            y_pred = self.predict(data_var)
            loss = self.compute_loss(y_pred, target_var)
            losses.update(loss.data[0], data[0].size(0))

            for callback, acc in zip(self.prediction_callbacks, scores):
                score = callback(y_pred, target)
                acc.update(score, data[0].size(0))

            batch_time.update(time.time() - end)
            end = time.time()
        print(' Valid\t || Loss: {:.3f} | Batch Test Time {:.3f}s'.format(losses.avg, batch_time.avg))
        for i, acc in enumerate(scores):
            print('\t || Acc {}: {:.3f}'.format(i, acc.avg))

        return losses, scores

    @staticmethod
    def save_checkpoint(state, is_best, path="", filename='checkpoint.pth.tar'):
        """
        Helper function to save models's parameters
        :param state:   dict with metadata and models's weight
        :param is_best: bool
        :param path:    save path
        :param filename:
        :return:
        """
        file_path = os.path.join(path, filename)
        torch.save(state, file_path)
        if is_best:
            print("Save best checkpoint...")
            shutil.copyfile(file_path, os.path.join(path, 'model_best.pth.tar'))

    def loop(self, epochs_qty, output_path):
        """
        Training loop for n epoch.
        todo : Use callback instead of hardcoded savetxt to leave the user choise on results handling
        :param epochs_qty:
        :param output_path:
        :return:
        """
        best_prec1 = 65000
        train_plot_data = np.zeros((epochs_qty, len(self.criterions)))
        valid_plot_data = np.zeros((epochs_qty, len(self.criterions)))
        loss_plot_data = np.zeros((epochs_qty, 2))

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for epoch in range(epochs_qty):
            print("-" * 20)
            print(" * EPOCH : {}".format(epoch))

            train_loss, train_scores = self.train()
            val_loss, valid_scores = self.validate()

            loss_plot_data[epoch, 0] = train_loss.avg
            loss_plot_data[epoch, 1] = val_loss.avg
            validation_loss_average = val_loss.avg

            for i, score in enumerate(train_scores):
                train_plot_data[epoch, i] = score.avg

            for i, score in enumerate(valid_scores):
                valid_plot_data[epoch, i] = score.avg
            # remember best loss and save checkpoint
            is_best = validation_loss_average < best_prec1
            best_prec1 = min(validation_loss_average, best_prec1)
            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, output_path, "checkpoint.pth.tar")
            np.savetxt(os.path.join(output_path, "loss.csv"), loss_plot_data, delimiter=",")
            np.savetxt(os.path.join(output_path, "train_scores.csv"), train_plot_data, delimiter=",")
            np.savetxt(os.path.join(output_path, "valid_scores.csv"), valid_plot_data, delimiter=",")
