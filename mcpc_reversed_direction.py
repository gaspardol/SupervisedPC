##  non-linear generative learning task on MNIST bechmark dataset
import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import Bias, Bias_supervised, bernoulli_fn, get_marginal_likelihood, get_mcpc_trainer_optimised, sample_pc, sample_x_fn_normal, setup_fig, get_mnist_data, get_model, get_pc_trainer, get_mcpc_trainer, fe_fn, random_step
from utils import get_mse_rec, get_fid, get_acc

import wandb
import sys
import predictive_coding as pc


torch.manual_seed(0)


def test(model, testloader, config, use_cuda):
    correct_count, all_count = 0., 0.
    model = model.eval()
    for data, labels in tqdm(testloader):
        pseudo_input = torch.zeros(data.shape[0], 10)
        if use_cuda:
            data, labels, pseudo_input = data.cuda(), labels.cuda(), pseudo_input.cuda() 
        # MAP inference
        pred = model(data).max(1)
        correct = (pred.indices == labels).long()
        correct_count += correct.sum()
        all_count += correct.size(0)
    model = model.train()
    return correct_count / all_count


use_cuda = torch.cuda.is_available()
if use_cuda:
    print("Using GPU")
    

config = {
    "model_type": "mcpc",
    "dataset": "mnist",
    "model_name": "mcpc_supervised",
    "loss_fn": bernoulli_fn ,
    "EPOCHS":50,
    "batch_size_train":2048,
    "batch_size_val": 6000,
    "batch_size_test": 1024,
    "input_size": 784,
    "hidden_size": 256,
    "hidden2_size": 256,
    "output_size": 10,
    "input_var":0.3,
    "activation_fn": 'relu',
    "T_pc":250,
    "optimizer_x_fn_pc": optim.Adam,
    "optimizer_x_kwargs_pc":{"lr": 0.1},
    "mixing":50,
    "sampling":100,
    "K": 500,
    "optimizer_x_kwargs_mcpc":{"lr": 0.001},
    "optimizer_p_fn_mcpc": optim.Adam,
    "optimizer_p_kwargs_mcpc": {"lr": 0.1, "weight_decay":0.01, "betas":(0.5,0.999)}
}

# Load MNIST data
train_loader, val_loader, test_loader = get_mnist_data(config)

# get model
# create model
gen_pc = get_model(config, use_cuda, sample_x_fn=sample_x_fn_normal)

# create trainer for MAP inference
# create trainer
pc_trainer = get_pc_trainer(gen_pc, config, is_mcpc=True)
# create MCPC trainer
mcpc_trainer = get_mcpc_trainer(gen_pc, config, training=True)

for idx_epoch in range(config["EPOCHS"]):
    for data, labels in train_loader:
        # convert to onehot
        labels = torch.nn.functional.one_hot(labels, 10).to(data.dtype)
        if use_cuda:
            labels, data = labels.cuda(), data.cuda()
        # initialise sampling
        pc_results = pc_trainer.train_on_batch(inputs=data, loss_fn=config["loss_fn"],loss_fn_kwargs={'_target':labels,'_var':config["input_var"]},is_log_progress=False,is_return_results_every_t=False,is_checking_after_callback_after_t=False)
        # mc inference
        mc_results = mcpc_trainer.train_on_batch(inputs=data,loss_fn=config["loss_fn"],loss_fn_kwargs={'_target': labels,'_var':config["input_var"]}, callback_after_t=random_step, callback_after_t_kwargs={'_pc_trainer':mcpc_trainer},is_sample_x_at_batch_start=False,is_log_progress=False,is_return_results_every_t=False,is_checking_after_callback_after_t=False)
    # test classificaiton accuracy
    acc = test(gen_pc, val_loader, config, use_cuda)
    print("Classificaiton accuracy: ", acc)


