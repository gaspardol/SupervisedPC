import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import Bias, Bias_supervised, bernoulli_fn, get_marginal_likelihood, get_mcpc_trainer_optimised, sample_pc, sample_x_fn_normal, setup_fig, get_mnist_data, get_model, get_pc_trainer, get_mcpc_trainer, fe_fn, random_step
from utils import get_mse_rec, get_fid, get_acc

import predictive_coding as pc


torch.manual_seed(0)


def test(model, testloader, config, use_cuda):
    # add bias layer for inferece
    test_model = nn.Sequential(
        Bias_supervised(10, offset=0.1),
        pc.PCLayer(sample_x_fn=sample_x_fn_normal),
        model
    )
    test_model.train()
    if use_cuda:
        test_model.cuda()

    # make pc_trainer for test_model
    pc_trainer = get_pc_trainer(test_model, config, is_mcpc=True, training=False)

    correct_count, all_count = 0., 0.
    for data, labels in tqdm(testloader):
        pseudo_input = torch.zeros(data.shape[0], 10)
        if use_cuda:
            data, labels, pseudo_input = data.cuda(), labels.cuda(), pseudo_input.cuda() 
        # MAP inference
        pc_trainer.train_on_batch(inputs=pseudo_input, loss_fn=config["loss_fn"],loss_fn_kwargs={'_target':data,'_var':config["input_var"]},is_log_progress=False,is_return_results_every_t=False,is_checking_after_callback_after_t=False)
        pred = torch.max(test_model[1].get_x(), dim=1)
        correct = (pred.indices == labels).long()
        correct_count += correct.sum()
        all_count += correct.size(0)
    return correct_count / all_count


def test_multihead(model, testloader, config, use_cuda, pc_trainer):
    """
                    - This function is under development -
        This function finds the one-hot output with the lowest energy of each test data and compares it to the true label
        This should behave like a one-hot prior 
    """

    correct_count, all_count = 0., 0.
    # set model pc layer to keep track of element wise energy
    layers = pc_trainer.get_model_pc_layers()
    for l in layers:
        l.is_keep_energy_per_datapoint = True

    for data, labels in tqdm(testloader):
        batch_size = data.shape[0]
        
        linspace = torch.linspace(0,9,10).reshape(1,-1)
    
        pseudo = linspace.repeat(batch_size,1).T.reshape(-1).to(torch.int64)
        pseudo = torch.nn.functional.one_hot(pseudo, 10).to(data.dtype)
        
        data = data.repeat(10,1)
        if use_cuda:
            data, pseudo = data.cuda(), pseudo.cuda()
        # MAP inference
        results = pc_trainer.train_on_batch(inputs=pseudo, loss_fn=config["loss_fn"],loss_fn_kwargs={'_target':data,'_var':config["input_var"]},is_log_progress=False,is_return_results_every_t=False,is_checking_after_callback_after_t=False, is_return_batchelement_loss=True)
        pred = results["overall_elementwise"].reshape(10,batch_size).min(0)
        correct = (pred.indices.cpu() == labels).long()
        correct_count += correct.sum()
        all_count += correct.size(0)

    # reset each layer to not take element wise energies
    for l in layers:
        l.is_keep_energy_per_datapoint = True
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
    "input_size": 10,
    "hidden_size": 256,
    "hidden2_size": 256,
    "output_size": 784,
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
        pc_results = pc_trainer.train_on_batch(inputs=labels, loss_fn=config["loss_fn"],loss_fn_kwargs={'_target':data,'_var':config["input_var"]},is_log_progress=False,is_return_results_every_t=False,is_checking_after_callback_after_t=False)
        # mc inference
        mc_results = mcpc_trainer.train_on_batch(inputs=labels,loss_fn=config["loss_fn"],loss_fn_kwargs={'_target': data,'_var':config["input_var"]}, callback_after_t=random_step, callback_after_t_kwargs={'_pc_trainer':mcpc_trainer},is_sample_x_at_batch_start=False,is_log_progress=False,is_return_results_every_t=False,is_checking_after_callback_after_t=False)
        # test classificaiton accuracy
    acc = test_multihead(gen_pc, val_loader, config, use_cuda, pc_trainer)
    print("Classificaiton accuracy: ", acc)
    acc = test(gen_pc, val_loader, config, use_cuda)
    print("Classificaiton accuracy: ", acc)


