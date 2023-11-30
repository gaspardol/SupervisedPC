import copy
import sys
import numpy as np
import torch
import torch.optim as optim
from utils import Bias_supervised, get_mcpc_trainer_optimised, get_mnist_data, get_model, get_pc_trainer, get_mcpc_trainer, fe_fn, random_step, bernoulli_fn, sample_x_fn_normal
import torch.nn as nn
import wandb
import predictive_coding as pc


# Check if an argument is provided
if len(sys.argv) > 1:
    # Get the argument value (assuming it's an integer)
    try:
        gpu_id = int(sys.argv[1]) 
        assert gpu_id < torch.cuda.device_count()
        
    except ValueError:
        print("Invalid integer value!")
else:
    gpu_id=0
    
print("Using gpu ", gpu_id)

use_cuda = torch.cuda.is_available()
if use_cuda:
    print("Using GPU")

def launch_sweep(sweep_config):
    return wandb.sweep(sweep_config, project="pc")
    

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
    test_config = copy.deepcopy(config)
    test_config["T_pc"] = 1500
    test_config["optimizer_x_kwargs_pc"]["lr"] = 0.05
    pc_trainer = get_pc_trainer(test_model, test_config, is_mcpc=True, training=False)
    pc_trainer = get_pc_trainer(test_model, config, is_mcpc=True, training=False)

    correct_count, all_count = 0., 0.
    for data, labels in testloader:
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

def train_model(sweep_config=None):
    with wandb.init(config=sweep_config):
        sweep_config = wandb.config
    
        with torch.cuda.device(gpu_id) if use_cuda else None:
            config = {
                "model_type": "mcpc",
                "dataset": "mnist",
                "model_name": "mcpc_supervised",
                "EPOCHS":50,
                "batch_size_train":sweep_config.batch_size,
                "batch_size_val": 1024,
                "batch_size_test": 1024,
                "input_size": 10,
                "hidden_size": sweep_config.hidden_size,
                "hidden2_size": sweep_config.hidden_size,
                "output_size": 784,
                "input_var":0.3,
                "activation_fn": sweep_config.activation_fn,
                "T_pc":250,
                "optimizer_x_fn_pc": optim.Adam,
                "optimizer_x_kwargs_pc":{"lr": sweep_config.lr_x},
                "optimizer_p_fn": optim.Adam,
                "optimizer_p_kwargs": {"lr": sweep_config.lr_p, "weight_decay":sweep_config.decay},
                "loss_fn": bernoulli_fn if sweep_config.loss_fn =="bernoulli_fn" else fe_fn 
            }

            # Load MNIST data
            train_loader, val_loader, test_loader = get_mnist_data(config)

            # get model
            gen_pc = get_model(config, use_cuda, sample_x_fn=sample_x_fn_normal)

            # create trainer for MAP inference
            pc_trainer = get_pc_trainer(gen_pc, config, training=True)
            
            best_acc=0.
            acc = test(gen_pc, val_loader, config, use_cuda)
            if acc > best_acc:
                best_acc = acc
            wandb.log({"acc": best_acc})
            for idx_epoch in range(config["EPOCHS"]):
                for data, labels in train_loader:
                    # convert to onehot
                    labels = torch.nn.functional.one_hot(labels, 10).to(data.dtype)
                    if use_cuda:
                        labels, data = labels.cuda(), data.cuda()
                    # initialise sampling
                    pc_results = pc_trainer.train_on_batch(inputs=labels, loss_fn=config["loss_fn"],loss_fn_kwargs={'_target':data,'_var':config["input_var"]},is_log_progress=False,is_return_results_every_t=False,is_checking_after_callback_after_t=False)
                    wandb.log({"pc_fe": pc_results["overall"][-1]})
                # test classificaiton accuracy
                acc = test(gen_pc, val_loader, config, use_cuda)
                if acc > best_acc:
                    best_acc = acc
                wandb.log({"acc": best_acc})
                

sweep_config={
    'method': 'bayes', 
    'name': 'pc_supervised',
    'project': 'pc',
    'metric': {
        'goal': 'maximize', 
        'name': 'acc'
		},
    'parameters': {
        'batch_size': {'values': [64, 128, 256, 2048]},
        'lr_x': {'values': [0.7, 0.3, 0.1, 0.03]},
        'lr_p': {'values': [0.03, 0.01, 0.003, 0.001]},
        "adam_b0": {'values': [0.5, 0.9]},
        'decay': {'values': [1., 0.1, 0.01, 0.]},
        'input_size': {'values': [10,15,20,25]},
        'hidden_size':{'values': [128, 256, 360]},
        'activation_fn': {'values': ['relu', 'tanh']},
        'loss_fn': {'values': ['bernoulli_fn']},
        'sampling': {'values':[1, 50, 100, 1000]}
     }
}


wandb.login()

sweep_id = None
sweep_id = 'pc/qm1zqjn7' 
if sweep_id == None:
    sweep_id = launch_sweep(sweep_config)

print("Sweep ID----------: " + str(sweep_id))

wandb.agent(sweep_id, train_model, count=100)

