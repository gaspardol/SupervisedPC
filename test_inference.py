import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import Bias, Bias_supervised, bernoulli_fn, sample_x_fn_normal, setup_fig, get_mnist_data, get_model, get_pc_trainer, get_mcpc_trainer, random_step
from copy import deepcopy
import predictive_coding as pc
import numpy as np 
import matplotlib.pyplot as plt


torch.manual_seed(0)

def cov_energy_function(inputs):
        sharpness = 10
        reg = 1
        precision = sharpness * torch.ones(1,inputs['x'].shape[-1],inputs['x'].shape[-1], device=inputs['x'].device)
        energy = torch.matmul(inputs['x'] - inputs['mu'], precision) * (inputs['x'] - inputs['mu'])
        # add regularisation to prevent latent states to be biger that 1 or smaller than 1
        energy += reg*(inputs['x']<0).double()*(inputs['x'].abs()) + reg*(inputs['x']>1).double()*((inputs['x']-1).abs())
        return energy
    

def test_cov(model,testloader,config, use_cuda, bias = 0.1):    
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
    test_config["T_pc"] = 500
    test_config["optimizer_x_kwargs_pc"]={"lr": 0.1}

    # make pc_trainer for test_model
    pc_trainer = get_pc_trainer(test_model, test_config, is_mcpc=True, training=False)

    correct_count, all_count = 0., 0.
    for data, labels in tqdm(testloader):
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


def test_cov_plot_latent(model,testloader,config, use_cuda, bias = 0.1):
        
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
    test_config["T_pc"] = 500
    test_config["optimizer_x_kwargs_pc"]={"lr": 0.1}

    # make pc_trainer for test_model
    pc_trainer = get_pc_trainer(test_model, test_config, is_mcpc=True, training=False)

    data, labels = list(testloader)[0]

    pseudo_input = torch.zeros(data.shape[0], 10)
    if use_cuda:
        data, labels, pseudo_input = data.cuda(), labels.cuda(), pseudo_input.cuda() 
    # MAP inference
    outputs = pc_trainer.train_on_batch(inputs=pseudo_input, loss_fn=config["loss_fn"],loss_fn_kwargs={'_target':data,'_var':config["input_var"]}, is_log_progress=True, is_return_results_every_t=True,is_checking_after_callback_after_t=False, is_return_representations=True)
    reps = torch.concatenate([rep.unsqueeze(0) for rep in outputs["representations"]], dim=0)
    totals = reps.sum(-1)

    plt.plot(reps[:,0,:])
    plt.plot(totals[:,0], 'k', label = "sum of latent activity", linewidth=2)
    plt.xlabel("inference steps")
    plt.ylabel("latent activity")
    plt.legend()
    plt.tight_layout()
    plt.show()


def test(model, testloader, config, use_cuda, bias = 0.1):
    # add bias layer for inferece
    test_model = nn.Sequential(
        Bias_supervised(10, offset=bias),
        pc.PCLayer(),
        model
    )
    test_model.train()
    if use_cuda:
        test_model.cuda()

    test_config = deepcopy(config)
    test_config["T_pc"] = 500
    test_config["optimizer_x_kwargs_pc"]={"lr": 0.1}

    # make pc_trainer for test_model
    pc_trainer = get_pc_trainer(test_model, test_config, is_mcpc=True, training=False)

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
    return (correct_count / all_count).cpu().item()


def test_multihead(model, testloader, config, use_cuda, pc_trainer):
    """
                    - This function is under development -
        This function finds the one-hot output with the lowest energy of each test data and compares it to the true label
        This should behave like a one-hot prior 
    """

    test_config = deepcopy(config)
    test_config["T_pc"] = 2000
    test_config["optimizer_x_kwargs_pc"]={"lr": 0.05}

    # make pc_trainer for test_model
    pc_trainer = get_pc_trainer(model, test_config, is_mcpc=True, training=False)


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
    return (correct_count / all_count).cpu().item()


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

model_path = "models/mcpc_supervised_68"
gen_pc.load_state_dict(torch.load(model_path, map_location='cuda:0'),strict=False)
gen_pc.train()

# create trainer for warm-up MAP inference
pc_trainer = get_pc_trainer(gen_pc, config, is_mcpc=True)
# create MCPC trainer
mcpc_trainer = get_mcpc_trainer(gen_pc, config, training=True)


test_cov_plot_latent(gen_pc, val_loader, config, use_cuda, bias=0.1)

# res=[]
# for i in range(10):
#     # acc = test(gen_pc, val_loader, config, use_cuda, bias=0.1)
#     acc = test_cov(gen_pc, val_loader, config, use_cuda, bias=0.1)
#     print("Classificaiton accuracy, 0: ", acc)
#     res+=[acc]

# print(np.mean(res))
# print(np.std(res))
