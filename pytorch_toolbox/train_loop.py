""" Train loop boilerplate code

    Uses preinstantiated data loaders, network, loss and optimizer to train a model.

    - Supports multiple inputs
    - Supports multiple outputs

"""
import datetime

from pytorch_toolbox.train_state import TrainingState
from pytorch_toolbox.utils import AverageMeter
from importlib._bootstrap_external import SourceFileLoader
import time
import torch
from tqdm import tqdm
import numpy as np
import os
import inspect
from shutil import copyfile


class TrainLoop:

    def __init__(self, model, train_data_loader, valid_data_loader, optimizers, backend, gradient_clip=False,
                 use_tensorboard=False, tensorboard_log_path="./logs", scheduler=None):
        """
        See examples/classification/train.py for usage

        :param model:               Any NetworkBase (in pytorch_toolbox.network) (it is the network model to train)
        :param train_data_loader:   Any torch dataloader for training data
        :param valid_data_loader:   Any torch dataloader for validation data
        :param optimizers:          List of any torch optimizers, the order is important though
        :param backend:             cuda | cpu
        :param scheduler:           Adjust learning rate
        """
        self.train_data = train_data_loader
        self.valid_data = valid_data_loader
        self.optims = optimizers if type(optimizers) is list else [optimizers]
        self.backend = backend
        self.model = model
        self.gradient_clip = gradient_clip
        self.tensorboard_logger = None
        self.scheduler = scheduler
        self.training_state = TrainingState()
        if use_tensorboard:
            from pytorch_toolbox.visualization.tensorboard_logger import TensorboardLogger
            date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            self.training_state.tensorboard_logger = TensorboardLogger(os.path.join(tensorboard_log_path, date_str))

        self.callbacks = []
        if backend == "cuda":
            self.model = self.model.cuda()

        self.training_state.model = self.model
        self.training_state.training_data_size = len(self.train_data.dataset)
        self.training_state.validation_data_size = len(self.valid_data.dataset)

    @staticmethod
    def load_from_output_directory(output_path, checkpoint_name="model_best.pth.tar", **kwargs):
        """
        Will import the network and the loader module saved durint a previous training. The loading is dynamic so no
        need to know the name of the Classes
        :param output_path:
        :param checkpoint_name:
        :param kwargs: dict containing the model arguments
        :return:
        """

        model = TrainLoop.load_network_from_output_directory(output_path, checkpoint_name, kwargs)

        loader_class = checkpoint_state['loader_class']
        data_path = os.path.join(output_path, "loader.py")
        module = data_path.split("/")[-1].split(".")[0]
        loader_module = SourceFileLoader(module, data_path).load_module()
        loader_class = getattr(loader_module, loader_class)

        return model, loader_class

    @staticmethod
    def load_network_from_output_directory(output_path, checkpoint_name="model_best.pth.tar", **kwargs):
        checkpoint_state = torch.load(os.path.join(output_path, checkpoint_name))
        weights = checkpoint_state['state_dict']
        network_class = checkpoint_state['network_class']

        network_path = os.path.join(output_path, "network.py")
        module = network_path.split("/")[-1].split(".")[0]
        network_module = SourceFileLoader(module, network_path).load_module()
        model = getattr(network_module, network_class)(**kwargs)
        model.eval()

        model.load_state_dict(weights)
        return model

    @staticmethod
    def convert_list_backend(input, backend="cuda"):
        """
        Will recursively convert every tensor to the given backend
        :param input:
        :return:
        """
        for i in range(len(input)):
            if isinstance(input[i], list):
                TrainLoop.convert_list_backend(input[i], backend)
            else:
                if backend == "cuda":
                    input[i] = input[i].cuda()
                else:
                    input[i] = input[i].cpu()

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
        TrainLoop.convert_list_backend(data, backend)
        TrainLoop.convert_list_backend(target, backend)
        return data, target

    @staticmethod
    def list_to_autograd(input):
        """
        Convert a list of Tensor to autograd variables
        :param input:
        :return:
        """
        ret = []
        for i in range(len(input)):
            if isinstance(input[i], list):
                output = TrainLoop.list_to_autograd(input[i])
            else:
                output = torch.autograd.Variable(input[i])
            ret.append(output)
        return ret


    def predict(self, data_variable):
        """
        compute prediction
        :param data_variable: tuple containing the network's input data
        :return:
        """
        y_pred = self.model(*data_variable)
        if not isinstance(y_pred, tuple):
            y_pred = (y_pred,)
        return y_pred

    def add_callback(self, func):
        """
        take a callback that will be called during the training process. See pytorch_toolbox.loop_callback_base

        :param func:
        :return:
        """
        if isinstance(func, list):
            for cb in func:
                self.callbacks.append(cb)
        else:
            self.callbacks.append(func)

    @staticmethod
    def _update_losses(avg_losses, batch_losses):
        if isinstance(batch_losses, dict):
            total_loss = 0
            for k, v in batch_losses.items():
                avg_loss = avg_losses.get(k, AverageMeter())
                avg_loss.update(v.item())
                avg_losses[k] = avg_loss
                total_loss += v
        else:
            total_loss = batch_losses
        avg_losses["total_loss"].update(total_loss.item())

        return total_loss

    def train(self):
        """
        Minibatch loop for training. Will iterate through the whole dataset and backprop for every minibatch

        It will keep an average of the computed losses and every score obtained with the user's callbacks.

        The information is displayed on the console

        :return: averageloss, [averagescore1, averagescore2, ...]
        """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = {"total_loss" : AverageMeter()}
        end = time.time()

        self.model.train()

        for i, (data, target) in tqdm(enumerate(self.train_data), total=len(self.train_data)):
            data_time.update(time.time() - end)
            data, target = self.setup_loaded_data(data, target, self.backend)
            data_var = self.list_to_autograd(data)
            target_var = self.list_to_autograd(target)
            y_pred = self.predict(data_var)
            loss = self.model.loss(y_pred, target_var)
            loss = self._update_losses(losses, loss)

            self.training_state.last_prediction = y_pred
            self.training_state.last_target = target
            self.training_state.last_network_input = data
            self.training_state.training_mode = True
            self.training_state.current_batch += 1

            for i, callback in enumerate(self.callbacks):
                callback.batch_(self.training_state)

            [optim.zero_grad() for optim in self.optims]
            loss.backward()
            if self.gradient_clip:
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
            [optim.step() for optim in self.optims]

            batch_time.update(time.time() - end)
            end = time.time()

        avg_losses = {k : v.avg for k, v in losses.items()}
        self.training_state.training_average_losses = avg_losses

        self.training_state.training_average_loss = avg_losses['total_loss']
        self.training_state.average_data_loading_time = data_time.avg
        self.training_state.average_batch_processing_time = batch_time.avg
        for i, callback in enumerate(self.callbacks):
            callback.epoch_(self.training_state)

        return losses

    def validate(self):
        """
        Validation loop (refer to train())

        Only difference is that there is no backpropagation..

        #TODO: It repeats mostly train()'s code...

        :return:
        """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = {"total_loss" : AverageMeter()}
        self.model.eval()

        end = time.time()
        for i, (data, target) in enumerate(self.valid_data):
            data_time.update(time.time() - end)
            data, target = self.setup_loaded_data(data, target, self.backend)
            data_var = self.list_to_autograd(data)
            target_var = self.list_to_autograd(target)
            with torch.no_grad():
                y_pred = self.predict(data_var)
                loss = self.model.loss(y_pred, target_var)
            loss = self._update_losses(losses, loss)

            self.training_state.last_prediction = y_pred
            self.training_state.last_target = target
            self.training_state.last_network_input = data
            self.training_state.training_mode = False

            for i, callback in enumerate(self.callbacks):
                callback.batch_(self.training_state)

            batch_time.update(time.time() - end)
            end = time.time()

        avg_losses = {k : v.avg for k, v in losses.items()}
        self.training_state.validation_average_losses = avg_losses

        self.training_state.validation_average_loss = avg_losses['total_loss']
        self.training_state.average_data_loading_time = data_time.avg
        self.training_state.average_batch_processing_time = batch_time.avg
        for i, callback in enumerate(self.callbacks):
            callback.epoch_(self.training_state)

        return losses

    @staticmethod
    def load_checkpoint(path="", filename='checkpoint*.pth.tar'):
        """
        Helper function to load models's parameters
        :param state:   dict with metadata and models's weight
        :param path:    load path
        :param filename:string
        :return:
        """
        file_path = os.path.join(path, filename)
        print("Loading model...")
        state = torch.load(file_path)
        dict = state['state_dict']
        best_prec1 = state['best_prec1']
        epoch = state['epoch'] - 1
        return dict, best_prec1, epoch

    def loop(self, epochs_qty, output_path,
             load_best_checkpoint=False,
             load_last_checkpoint=False,
             save_best_checkpoint=True,
             save_last_checkpoint=True,
             save_all_checkpoints=True):
        """
        Training loop for n epoch.
        todo : Use callback instead of hardcoded savetxt to leave the user choise on results handling
        :param load_best_checkpoint:  If true, will check for model_best.pth.tar in output path and load it.
        :param save_best_checkpoint:  If true, will save model_best.pth.tar in output path.
        :param save_all_checkpoints:  If true, will save all checkpoints as checkpoint<epoch>.pth.tar in output path.
        :param epochs_qty:            Number of epoch to train
        :param output_path:           Path to save the model and log data
        :return:
        """
        best_prec1 = float('Inf')
        epoch_start = 0

        assert not(load_best_checkpoint and load_last_checkpoint), 'Choose to load only one model: last or best'
        if load_best_checkpoint or load_last_checkpoint:
            model_name = 'model_best.pth.tar' if load_best_checkpoint else 'model_last.pth.tar'
            if os.path.exists(os.path.join(output_path, model_name)):
                dict, best_prec1, epoch_best = self.load_checkpoint(output_path, model_name)
                self.model.load_state_dict(dict)
                # also get back the last i_epoch, won't start from 0 again
                epoch_start = epoch_best + 1
            else:
                raise RuntimeError("Can't load model {}".format(os.path.join(output_path, model_name)))

        for epoch in range(epoch_start, epochs_qty):
            print("-" * 20)
            print(" * EPOCH : {}".format(epoch))

            self.training_state.current_epoch = epoch

            train_loss = self.train()
            val_loss = self.validate()

            if self.scheduler is not None:
                self.scheduler.step() if type(self.scheduler) != torch.optim.lr_scheduler.ReduceLROnPlateau else self.scheduler.step(val_loss.val)

            # log gradients, loss in tensorboard
            if self.training_state.tensorboard_logger is not None:
                self.training_state.tensorboard_logger.scalar_summary('loss', self.training_state.training_average_loss, epoch + 1)
                self.training_state.tensorboard_logger.scalar_summary('loss', self.training_state.validation_average_loss, epoch + 1, is_train=False)
                for tag, value in self.training_state.model.named_parameters():
                    tag = tag.replace('.', '/')
                    self.training_state.tensorboard_logger.histo_summary(tag, self.to_np(value), epoch + 1)
                    try:
                        self.training_state.tensorboard_logger.histo_summary(tag + '/grad', self.to_np(value.grad), epoch + 1)
                    except AttributeError:
                        print('None grad:', tag, value.shape)
                        self.training_state.tensorboard_logger.histo_summary(tag + '/grad', np.asarray([0]), epoch + 1)

            # remember best loss and save checkpoint
            is_best = self.training_state.validation_average_loss < best_prec1
            best_prec1 = min(self.training_state.validation_average_loss, best_prec1)
            checkpoint_data = {'epoch': epoch, 'state_dict': self.model.state_dict(), 'best_prec1': best_prec1,
                               'network_class': self.model.__class__.__name__, 'loader_class': self.train_data.dataset.__class__.__name__}
            if save_all_checkpoints:
                torch.save(checkpoint_data, os.path.join(output_path, "checkpoint{}.pth.tar".format(epoch)))
            if save_best_checkpoint and is_best:
                torch.save(checkpoint_data, os.path.join(output_path, "model_best.pth.tar"))
            if save_last_checkpoint:
                torch.save(checkpoint_data, os.path.join(output_path, "model_last.pth.tar"))

            # save source file for evaluation
            network_file_path = inspect.getfile(self.model.__class__)
            copyfile(network_file_path, os.path.join(output_path, "network.py"))
            loader_file_path = inspect.getfile(self.train_data.dataset.__class__)
            copyfile(loader_file_path, os.path.join(output_path, "loader.py"))

    @staticmethod
    def to_np(x):
        return x.data.cpu().numpy()

