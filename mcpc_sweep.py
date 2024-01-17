import sys
import torch
import torch.optim as optim
from utils import Bias_supervised, get_mcpc_trainer_optimised, get_mnist_data, get_model, get_pc_trainer, get_mcpc_trainer, fe_fn, random_step, bernoulli_fn, sample_x_fn_normal
import torch.nn as nn
import wandb
import predictive_coding as pc
from copy import deepcopy

# Check if an argument is provided
if len(sys.argv) > 1:
    # Get the argument value (assuming it's an integer)
    try:
        gpu_id = int(sys.argv[1]) 
        assert gpu_id < torch.cuda.device_count()
        
    except ValueError:
        print("Invalid GPU ID!")
else:
    gpu_id=0
    
print("Using gpu ", gpu_id)

use_cuda = torch.cuda.is_available()
if use_cuda:
    print("Using GPU")

def launch_sweep(sweep_config):
    return wandb.sweep(sweep_config, project="mcpc")
    


def cov_energy_function(inputs):
        sharpness = 10
        reg = 1
        precision = sharpness * torch.ones(1,inputs['x'].shape[-1],inputs['x'].shape[-1], device=inputs['x'].device)
        energy = torch.matmul(inputs['x'] - inputs['mu'], precision) * (inputs['x'] - inputs['mu'])
        # add regularisation to prevent latent states to be biger that 1 or smaller than 1
        energy += reg*(inputs['x']<0).double()*(inputs['x'].abs()) + reg*(inputs['x']>1).double()*((inputs['x']-1).abs())
        return energy
    

def test(model,testloader,config, use_cuda, bias = 0.1):    
    # add bias layer for inferece
    test_model = nn.Sequential(
        Bias_supervised(10, offset=bias),
        pc.PCLayer(energy_fn=cov_energy_function),
        model
    )
    test_model.train()
    if use_cuda:
        test_model.cuda()

    test_config = deepcopy(config)
    test_config["T_pc"] = 1000

    # make pc_trainer for test_model
    pc_trainer = get_pc_trainer(test_model, test_config, is_mcpc=True, training=False)

    correct_count, all_count = 0., 0.
    for data, labels in (testloader):
        pseudo_input = torch.zeros(data.shape[0], 10)
        if use_cuda:
            data, labels, pseudo_input = data.cuda(), labels.cuda(), pseudo_input.cuda() 
        # MAP inference
        pc_trainer.train_on_batch(inputs=pseudo_input, loss_fn=config["loss_fn"],loss_fn_kwargs={'_target':data,'_var':config["input_var"]}, is_log_progress=False, is_return_results_every_t=False,is_checking_after_callback_after_t=False)
        pred = torch.max(test_model[1].get_x(), dim=1)
        correct = (pred.indices == labels).long()
        correct_count += correct.sum()
        all_count += correct.size(0)
    return (correct_count / all_count).cpu().item()


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
                "mixing":500,
                "sampling":sweep_config.sampling,
                "K": 1,
                "optimizer_x_kwargs_mcpc":{"lr": sweep_config.lr_x_mcpc},
                "optimizer_p_fn_mcpc": optim.Adam,
                "optimizer_p_kwargs_mcpc": {"lr": sweep_config.lr_p, "weight_decay":sweep_config.decay, "betas":(sweep_config.adam_b0,0.999)},
                "loss_fn": bernoulli_fn if sweep_config.loss_fn =="bernoulli_fn" else fe_fn 
            }

            # Load MNIST data
            train_loader, val_loader, test_loader = get_mnist_data(config)

            # get model
            gen_pc = get_model(config, use_cuda, sample_x_fn=sample_x_fn_normal)

            # create trainer for MAP inference
            # create trainer
            pc_trainer = get_pc_trainer(gen_pc, config, is_mcpc=True)
            # create MCPC trainer
            mcpc_trainer = get_mcpc_trainer(gen_pc, config, training=True)

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
                    # mc inference
                    mc_results = mcpc_trainer.train_on_batch(inputs=labels,loss_fn=config["loss_fn"],loss_fn_kwargs={'_target': data,'_var':config["input_var"]}, callback_after_t=random_step, callback_after_t_kwargs={'_pc_trainer':mcpc_trainer},is_sample_x_at_batch_start=False,is_log_progress=False,is_return_results_every_t=False,is_checking_after_callback_after_t=False)
                    wandb.log({"pc_fe": pc_results["overall"][-1],"mcpc_fe": mc_results["overall"][-1]})
                # test classificaiton accuracy
                acc = test(gen_pc, val_loader, config, use_cuda)
                if acc > best_acc:
                    best_acc = acc
                wandb.log({"acc": best_acc})
                

sweep_config={
    'method': 'bayes', 
    'name': 'mcpc_supervised',
    'project': 'mcpc',
    'metric': {
        'goal': 'maximize', 
        'name': 'acc'
		},
    'parameters': {
        'batch_size': {'values': [64, 128, 256, 512, 1048, 2048]},
        'lr_x': {'values': [0.7, 0.3, 0.1, 0.03, 0.01]},
        'lr_x_mcpc': {'values': [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]},
        'lr_p': {'values': [0.1, 0.03, 0.01, 0.003, 0.001]},
        "adam_b0": {'values': [0.5, 0.9]},
        'decay': {'values': [1., 0.1, 0.01, 0.001, 0.]},
        'hidden_size':{'values': [128, 256, 360]},
        'activation_fn': {'values': ['relu', 'tanh']},
        'loss_fn': {'values': ['bernoulli_fn']},
        'sampling': {'values':[1, 100, 1000]}
     }
}


wandb.login()

sweep_id = None
sweep_id = 'mcpc/b93ftnkq' 
if sweep_id == None:
    sweep_id = launch_sweep(sweep_config)

print("Sweep ID----------: " + str(sweep_id))

wandb.agent(sweep_id, train_model, count=100)

