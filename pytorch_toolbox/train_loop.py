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
    def __init__(self, model, train_data_loader, valid_data_loader, optimizer, backend):
        """
        See examples/classification/train.py for usage

        :param model:               Any NetworkBase (in pytorch_toolbox.network) (it is the network model to train)
        :param train_data_loader:   Any torch dataloader for training data
        :param valid_data_loader:   Any torch dataloader for validation data
        :param optimizer:           Any torch optimizer
        :param backend:             cuda | cpu
        """
        self.train_data = train_data_loader
        self.valid_data = valid_data_loader
        self.optim = optimizer
        self.backend = backend
        self.model = model

        self.score_callbacks = []
        self.epoch_callbacks = []
        self.batch_callbacks = []

        if backend == "cuda":
            self.model = self.model.cuda()

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
    def to_autograd(data, target, istest=True):
        """
        Converts data and target to autograd Variable
        :param data:
        :param target:
        :return:
        """
        target_var = []
        data_var = []
        for i in range(len(data)):
            data_var.append(torch.autograd.Variable(data[i], volatile=istest))
        for i in range(len(target)):
            target_var.append(torch.autograd.Variable(target[i], volatile=istest))
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

    def add_score_callback(self, func):
        """
        add a prediction callback that takes as input the predictions and targets and return
        a *score* that will be displayed, the callback must return a float

        GOTCHA: There is a gotcha here, the callback will get the list of prediction and the list of target for every
                minibatch iterations.

        :param func:
        :return:
        """
        if isinstance(func, list):
            for cb in func:
                self.score_callbacks.append(cb)
        else:
            self.score_callbacks.append(func)

    def add_epoch_callback(self, func):
        """
        add a epoch callback that takes as input the average loss, data load time, batch load time,
        and a list of average scores computed by score_callbacks and a boolean to tell if it is called in the train
        or validation

        GOTCHA: There is a gotcha here, the callback will get the list of prediction and the list of target for every
                minibatch iterations.

        :param func:
        :return:
        """
        if isinstance(func, list):
            for cb in func:
                self.epoch_callbacks.append(cb)
        else:
            self.epoch_callbacks.append(func)

    def add_batch_callback(self, func):
        """
        add a batch callback that takes as input the last prediction, target and a boolean to tell if it is called from
         train or validation loop for each minibatch

        GOTCHA: There is a gotcha here, the callback will get the list of prediction and the list of target for every
                minibatch iterations.

        :param func:
        :return:
        """
        if isinstance(func, list):
            for cb in func:
                self.batch_callbacks.append(cb)
        else:
            self.batch_callbacks.append(func)

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
        for i in range(len(self.score_callbacks)):
            scores.append(AverageMeter())

        for i, (data, target) in tqdm(enumerate(self.train_data), total=len(self.train_data)):
            data_time.update(time.time() - end)
            if not isinstance(target, list):
                target = [target.view(-1)]
            data, target = self.setup_loaded_data(data, target, self.backend)
            data_var, target_var = self.to_autograd(data, target, istest=False)
            y_pred = self.predict(data_var)
            loss = self.model.loss(y_pred, target_var)
            losses.update(loss.data[0], data[0].size(0))

            for callback, acc in zip(self.score_callbacks, scores):
                score = callback(y_pred, target)
                acc.update(score, data[0].size(0))
            for callback in self.batch_callbacks:
                callback(y_pred, target, True)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            batch_time.update(time.time() - end)
            end = time.time()
        for callback in self.epoch_callbacks:
            scores_average = [x.avg for x in scores]
            callback(losses.avg, data_time.avg, batch_time.avg, scores_average, True)
        return losses, scores

    def validate(self):
        """
        Validation loop (refer to train())

        Only difference is that there is no backpropagation..

        #TODO: It repeats mostly train()'s code...

        :return:
        """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        scores = []
        for i in range(len(self.score_callbacks)):
            scores.append(AverageMeter())

        self.model.eval()

        end = time.time()
        for i, (data, target) in enumerate(self.valid_data):
            data_time.update(time.time() - end)
            if not isinstance(target, list):
                target = [target.view(-1)]
            data, target = self.setup_loaded_data(data, target, self.backend)
            data_var, target_var = self.to_autograd(data, target, istest=True)
            y_pred = self.predict(data_var)
            loss = self.model.loss(y_pred, target_var)
            losses.update(loss.data[0], data[0].size(0))

            for callback, acc in zip(self.score_callbacks, scores):
                score = callback(y_pred, target)
                acc.update(score, data[0].size(0))
            for callback in self.batch_callbacks:
                callback(y_pred, target, False)

            batch_time.update(time.time() - end)
            end = time.time()
        for callback in self.epoch_callbacks:
            scores_average = [x.avg for x in scores]
            callback(losses.avg, data_time.avg, batch_time.avg, scores_average, False)

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

    @staticmethod
    def load_checkpoint(path="", filename='model_best.pth.tar'):
        """
        Helper function to load models's parameters
        :param state:   dict with metadata and models's weight
        :param path:    load path
        :param filename:
        :return:
        """
        file_path = os.path.join(path, filename)
        print("Loading best model...")
        state = torch.load(file_path)

        return state

    def loop(self, epochs_qty, output_path, load_checkpoint=False):
        """
        Training loop for n epoch.
        todo : Use callback instead of hardcoded savetxt to leave the user choise on results handling
        :param load_checkpoint:  If true, will check for model_best.pth.tar in output path and load it.
        :param epochs_qty:       Number of epoch to train
        :param output_path:      Path to save the model and log data
        :return:
        """
        best_prec1 = 65000
        train_plot_data = None
        valid_plot_data = None
        loss_plot_data = np.zeros((epochs_qty, 2))


        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if load_checkpoint:
            model_name = 'model_best.pth.tar'
            if os.path.exists(os.path.join(output_path, model_name)):
                self.model.load_state_dict(self.load_checkpoint(output_path, model_name)['state_dict'])
            else:
                raise RuntimeError("Can't load model {}".format(os.path.join(output_path, model_name)))

        for epoch in range(epochs_qty):
            print("-" * 20)
            print(" * EPOCH : {}".format(epoch))

            train_loss, train_scores = self.train()
            val_loss, valid_scores = self.validate()

            loss_plot_data[epoch, 0] = train_loss.avg
            loss_plot_data[epoch, 1] = val_loss.avg
            validation_loss_average = val_loss.avg

            if train_plot_data is None or valid_plot_data is None:
                train_plot_data = np.zeros((epochs_qty, len(train_scores)))
                valid_plot_data = np.zeros((epochs_qty, len(valid_scores)))

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
            }, is_best, output_path, "checkpoint{}.pth.tar".format(epoch))
            np.savetxt(os.path.join(output_path, "loss.csv"), loss_plot_data, delimiter=",")
            np.savetxt(os.path.join(output_path, "train_scores.csv"), train_plot_data, delimiter=",")
            np.savetxt(os.path.join(output_path, "valid_scores.csv"), valid_plot_data, delimiter=",")
