""" Train loop boilerplate code

    Uses preinstantiated data loaders, network, loss and optimizer to train a model.

    - Supports multiple inputs
    - Supports multiple outputs

"""
import datetime

from pytorch_toolbox.utils import AverageMeter
import time
import torch
from tqdm import tqdm
import numpy as np
import os


class TrainLoop:

    def __init__(self, model, train_data_loader, valid_data_loader, optimizer, backend, gradient_clip=False,
                 use_tensorboard=False, tensorboard_log_path="./logs"):
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
        self.gradient_clip = gradient_clip
        self.use_tensorboard = use_tensorboard
        self.tensorboard_logger = None
        self.epoch_start = 0
        self.best_prec1 = float('Inf')
        if self.use_tensorboard:
            from pytorch_toolbox.visualization.tensorboard_logger import TensorboardLogger
            date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            self.tensorboard_logger = TensorboardLogger(os.path.join(tensorboard_log_path, date_str))

        self.callbacks = []
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
            # for i in range(len(target)):
            #     target[i] = target[i].cuda()
        else:
            for i in range(len(data)):
                data[i] = data[i].cpu()
            # for i in range(len(target)):
            #     target[i] = target[i].cpu()
        return data, target

    @staticmethod
    def to_autograd(data, target, is_train=True):
        """
        Converts data and target to autograd Variable
        :param data:
        :param target:
        :return:
        """
        target_var = []
        data_var = []
        for i in range(len(data)):
            data_var.append(torch.autograd.Variable(data[i], volatile=not is_train))
        # for i in range(len(target)):
        #     target_var.append(torch.autograd.Variable(target[i], volatile=not is_train))
        return data_var, target

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

    def train(self, epoch):
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

        for i, (data, target) in tqdm(enumerate(self.train_data), total=len(self.train_data)):
            data_time.update(time.time() - end)
            data, target = self.setup_loaded_data(data, target, self.backend)
            data_var, target_var = self.to_autograd(data, target, is_train=True)
            y_pred = self.predict(data_var)
            loss = self.model.loss(y_pred, target_var)
            losses.update(loss.data[0], data[0].size(0))

            for i, callback in enumerate(self.callbacks):
                callback.batch(y_pred, data, target, is_train=True, tensorboard_logger=self.tensorboard_logger)

            self.optim.zero_grad()
            loss.backward()
            if self.gradient_clip:
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
            self.optim.step()

            batch_time.update(time.time() - end)
            end = time.time()

        for i, callback in enumerate(self.callbacks):
            callback.epoch(epoch, losses.avg, data_time.avg, batch_time.avg,
                           is_train=True, tensorboard_logger=self.tensorboard_logger)

        return losses

    def test(self):
        """
        Test loop (refer to train())
        """
        self.model.eval()

        for i, (data, target) in enumerate(self.valid_data):
            data, target = self.setup_loaded_data(data, target, self.backend)
            data_var, target_var = self.to_autograd(data, target, is_train=False)
            y_pred = self.predict(data_var)

            for i, callback in enumerate(self.callbacks):
                callback.batch(y_pred, data, target, is_train=False, tensorboard_logger=self.tensorboard_logger)


    def validate(self, epoch):
        """
        Validation loop (refer to train())

        Only difference is that there is no backpropagation..

        #TODO: It repeats mostly train()'s code...

        :return:
        """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        self.model.eval()

        end = time.time()
        for i, (data, target) in enumerate(self.valid_data):
            data_time.update(time.time() - end)
            data, target = self.setup_loaded_data(data, target, self.backend)
            data_var, target_var = self.to_autograd(data, target, is_train=False)
            y_pred = self.predict(data_var)
            loss = self.model.loss(y_pred, target_var)
            losses.update(loss.data[0], data[0].size(0))

            for i, callback in enumerate(self.callbacks):
                callback.batch(y_pred, data, target, is_train=False, tensorboard_logger=self.tensorboard_logger)

            batch_time.update(time.time() - end)
            end = time.time()

        for i, callback in enumerate(self.callbacks):
            callback.epoch(epoch, losses.avg, data_time.avg, batch_time.avg, is_train=False, tensorboard_logger=self.tensorboard_logger)

        return losses

    def load_checkpoint(self, path="", filename='checkpoint*.pth.tar'):
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
        self.best_prec1 = state['best_prec1']
        epoch = state['epoch'] - 1
        return dict, self.best_prec1, epoch


    def setup_checkpoint(self, load_best_checkpoint, load_last_checkpoint, output_path, forget_best_prec):
        """
        Function to setup the desired checkpoint to be loaded (if desired)
        """
        assert not (load_best_checkpoint and load_last_checkpoint), 'Choose to load only one model: last or best'
        if load_best_checkpoint or load_last_checkpoint:
            model_name = 'model_best.pth.tar' if load_best_checkpoint else 'model_last.pth.tar'
            if os.path.exists(os.path.join(output_path, model_name)):
                dict, self.best_prec1, epoch_best = self.load_checkpoint(output_path, model_name)
                if forget_best_prec:
                    self.best_prec1 = float('Inf')
                print('best_prec: {}'.format(self.best_prec1))
                self.model.load_state_dict(dict)
                # also get back the last i_epoch, won't start from 0 again
                self.epoch_start = epoch_best + 1
            else:
                raise RuntimeError("Can't load model {}".format(os.path.join(output_path, model_name)))

    def loop(self, epochs_qty, output_path,
             load_best_checkpoint=False,
             load_last_checkpoint=False,
             save_best_checkpoint=True,
             save_last_checkpoint=True,
             save_all_checkpoints=True,
             forget_best_prec=False):
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


        self.setup_checkpoint(load_best_checkpoint, load_last_checkpoint, output_path, forget_best_prec)

        for epoch in range(self.epoch_start, epochs_qty):
            print("-" * 20)
            print(" * EPOCH : {}".format(epoch))

            train_loss = self.train(epoch + 1)
            val_loss = self.validate(epoch + 1)

            validation_loss_average = val_loss.avg
            training_loss_average = train_loss.avg

            if self.tensorboard_logger is not None:
                self.tensorboard_logger.scalar_summary('loss', training_loss_average, epoch + 1)
                self.tensorboard_logger.scalar_summary('loss', validation_loss_average, epoch + 1, is_train=False)
                for tag, value in self.model.named_parameters():
                    tag = tag.replace('.', '/')
                    self.tensorboard_logger.histo_summary(tag, self.to_np(value), epoch + 1)
                    try:
                        self.tensorboard_logger.histo_summary(tag + '/grad', self.to_np(value.grad), epoch + 1)
                    except AttributeError:
                        print('None grad:', tag, value.shape)
                        self.tensorboard_logger.histo_summary(tag + '/grad', np.asarray([0]), epoch + 1)

            # remember best loss and save checkpoint
            is_best = validation_loss_average < self.best_prec1
            self.best_prec1 = min(validation_loss_average, self.best_prec1)
            checkpoint_data = {'epoch': epoch, 'state_dict': self.model.state_dict(), 'best_prec1': self.best_prec1}
            if save_all_checkpoints:
                torch.save(checkpoint_data, os.path.join(output_path, "checkpoint{}.pth.tar".format(epoch)))
            if save_best_checkpoint and is_best:
                torch.save(checkpoint_data, os.path.join(output_path, "model_best.pth.tar"))
            if save_last_checkpoint:
                torch.save(checkpoint_data, os.path.join(output_path, "model_last.pth.tar"))

    @staticmethod
    def to_np(x):
        return x.data.cpu().numpy()

