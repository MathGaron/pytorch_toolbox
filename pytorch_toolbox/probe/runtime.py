import torch
from torch.autograd import Variable
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def compute_test_time(network_class, input_size, max_batch_size, step_size=1, is_cuda=False):
    backend = "cpu"
    if is_cuda:
        backend = "cuda"
    model = network_class()
    if is_cuda:
        model = model.cuda()
    model.eval()
    time_log = []

    # make sure that everything is in memorybefore the actual tests
    batch = Variable(torch.FloatTensor(1, *input_size))
    if is_cuda:
        batch = batch.cuda()
    model(batch)

    print("Compute {} test time".format(backend))
    for i in tqdm(range(0, max_batch_size, step_size)):
        batch = Variable(torch.FloatTensor(i+1, *input_size))
        if is_cuda:
            batch = batch.cuda()
        time_start = time.time()
        model(batch)
        time_log.append(time.time() - time_start)
    plt.plot(np.arange(1, max_batch_size + 1, step_size), time_log)
    plt.title("{} test time w.r.t minibatch size".format(backend))
    plt.ylabel("Time (s)")
    plt.xlabel("Batch size")


def compute_train_time(network_class, input_size, max_batch_size, step_size=1, is_cuda=False, backward_only=False):
    backend = "cpu"
    if is_cuda:
        backend = "cuda"
    model = network_class()
    if is_cuda:
        model = model.cuda()
    model.train()
    time_log = []

    # make sure that everything is in memorybefore the actual tests
    batch = Variable(torch.FloatTensor(1, *input_size))
    if is_cuda:
        batch = batch.cuda()
    model(batch)

    print("Compute {} test time".format(backend))
    for i in tqdm(range(0, max_batch_size, step_size)):
        batch = Variable(torch.FloatTensor(i+1, *input_size))
        if is_cuda:
            batch = batch.cuda()
        time_start = time.time()
        prediction = model(batch)
        out = torch.sum(prediction)
        if backward_only:
            time_start = time.time()
        out.backward()
        time_log.append(time.time() - time_start)
    plt.plot(np.arange(1, max_batch_size + 1, step_size), time_log)
    plt.title("{} train time w.r.t minibatch size".format(backend))
    plt.ylabel("Time (s)")
    plt.xlabel("Batch size")