import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import predictive_coding as pc
import matplotlib.pyplot as plt
from matplotlib import pylab
from matplotlib.ticker import StrMethodFormatter
from torch.utils.data import TensorDataset, DataLoader
import os, subprocess, glob
import tempfile, shutil
from torchvision.utils import save_image
import os


def energy_sparse(inputs):
    sparse_reg = 0.001
    return (inputs['mu'] - inputs['x'])**2 + sparse_reg * inputs['x']**2

def sample_x_fn(inputs):
    return inputs['mu'].detach().clone().uniform_(-10.,10.)

def sample_x_fn_normal(inputs):
    return torch.randn_like(inputs['mu'])

def sample_x_fn_one_hot_mean(inputs):
    return torch.ones_like(inputs['mu'])/inputs['mu'].shape[-1]

def sample_x_fn_cte(inputs):
    return 3*torch.ones_like(inputs['mu'])

def fe_fn(output, _target, _var):
    return (1/_var)*0.5*(output - _target).pow(2).sum()

def bernoulli_fn(output, _target, _var=None, _reduction="sum"):
    loss = nn.BCEWithLogitsLoss(reduction=_reduction)
    return loss(output, _target)

def zero_fn(output):
    return torch.tensor(0.)

def bernoulli_fn_pc(inputs):
    output=inputs['mu']
    _target=inputs['x']
    loss = nn.BCEWithLogitsLoss(reduction='sum')
    return loss(output, _target)


def setup_fig(zero=False):
    params = {'legend.fontsize': 14,
              'figure.figsize': (4., 4.),
              'axes.labelsize': 16,
              'axes.titlesize': 18,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14}
    pylab.rcParams.update(params)
    if zero is False:
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))  # No decimal places
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))  # 2 decimal places

def train(model, x, y, optimizer, criterion):
    # model.zero_grad()
    output = model(x)
    optimizer.zero_grad()
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    return loss, output

def test(model, testloader, print_acc=False):
    correct_count, all_count = 0., 0.
    for data, labels in testloader:
        pred = torch.max(torch.exp(model(data)), 1)
        correct = (pred.indices == labels).long()
        correct_count += correct.sum()
        all_count += correct.size(0)
    acc =correct_count / all_count
    if print_acc:
        print("Model Accuracy =", acc)
    return acc


class BinaryMNIST(TensorDataset):
    def __init__(self, mnist_dataset):
        self.dataset = mnist_dataset

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        img, label = self.dataset.__getitem__(index)
        img = (img>0.5).type_as(img)
        return img, label
        

def get_mnist_data(config, binary=True):
    # Load MNIST data
    if config['loss_fn'] == fe_fn:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.Lambda(lambda x: torch.flatten(x))])
        train_dataset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
        temp_dataset = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
        val_dataset = torch.utils.data.Subset(temp_dataset, [i for i in range(6000)])
        test_dataset = torch.utils.data.Subset(temp_dataset, [i+6000 for i in range(4000)])    
    elif (config['loss_fn'] == bernoulli_fn) or (config['loss_fn'] == "vae"):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
        temp_dataset = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
        if binary:
            train_dataset = BinaryMNIST(datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform))
            val_dataset = BinaryMNIST(torch.utils.data.Subset(temp_dataset, [i for i in range(6000)]))
            test_dataset = BinaryMNIST(torch.utils.data.Subset(temp_dataset, [i+6000 for i in range(4000)]))    
        else:
            train_dataset = (datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform))
            val_dataset = (torch.utils.data.Subset(temp_dataset, [i for i in range(6000)]))
            test_dataset = (torch.utils.data.Subset(temp_dataset, [i+6000 for i in range(4000)]))    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size_train"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size_val"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size_test"], shuffle=False)
    return train_loader, val_loader, test_loader


def get_model(config, use_cuda, sample_x_fn = sample_x_fn, energy_fn=None):
    # create model
    if config['activation_fn'] == 'relu':
        activation_fn = nn.ReLU
    elif config['activation_fn'] == 'tanh':
        activation_fn = nn.Tanh

    pc_layer_kwargs = {
        "sample_x_fn":sample_x_fn,
    }
    if energy_fn != None:
        pc_layer_kwargs["energy_fn"] = energy_fn

    gen_pc = nn.Sequential(
        nn.Linear(config["input_size"],config["input_size"]),
        pc.PCLayer(**pc_layer_kwargs),
        activation_fn(),
        nn.Linear(config["input_size"], config["hidden_size"]),
        pc.PCLayer(**pc_layer_kwargs),
        activation_fn(),
        nn.Linear(config["hidden_size"], config["hidden2_size"]),
        pc.PCLayer(**pc_layer_kwargs),
        activation_fn(),
        nn.Linear(config["hidden2_size"], config["output_size"]),
    )
    gen_pc.train()
    if use_cuda:
        gen_pc.cuda()
    return gen_pc

def get_pc_trainer(gen_pc, config, is_mcpc=False, training=True):
    if is_mcpc:
        pc_trainer = pc.PCTrainer(gen_pc, 
            T=config["T_pc"], 
            update_x_at='all', 
            optimizer_x_fn=config["optimizer_x_fn_pc"],
            optimizer_x_kwargs=config["optimizer_x_kwargs_pc"],
            early_stop_condition = "False",
            update_p_at="never",   
            plot_progress_at=[]
        )
    else:
        pc_trainer = pc.PCTrainer(gen_pc, 
            T=config["T_pc"], 
            update_x_at='all', 
            optimizer_x_fn=config["optimizer_x_fn_pc"],
            optimizer_x_kwargs=config["optimizer_x_kwargs_pc"],
            early_stop_condition = "False",
            update_p_at= "last" if training else "never",   
            optimizer_p_fn=config["optimizer_p_fn"],
            optimizer_p_kwargs=config["optimizer_p_kwargs"],
            plot_progress_at=[]
        )
    return pc_trainer

def get_mcpc_trainer(gen_pc, config, training=True):
    mcpc_trainer = pc.PCTrainer(
        gen_pc,
        T=config["mixing"]+config["sampling"],
        update_x_at='all',
        optimizer_x_fn=optim.SGD,
        optimizer_x_kwargs=config["optimizer_x_kwargs_mcpc"],
        update_p_at="last" if training else "never",
        accumulate_p_at=[i+config["mixing"] for i in range(config["sampling"])],
        optimizer_p_fn= config["optimizer_p_fn_mcpc"] if training else optim.SGD,
        optimizer_p_kwargs=config["optimizer_p_kwargs_mcpc"] if training else {"lr": 0.0},
        plot_progress_at=[]
    )
    return mcpc_trainer

def get_mcpc_trainer_optimised(gen_pc, config, training=True):
    mcpc_trainer = pc.PCTrainer(
        gen_pc,
        T=config["K"],
        update_x_at='all',
        optimizer_x_fn=optim.SGD,
        optimizer_x_kwargs=config["optimizer_x_kwargs_mcpc"],
        update_p_at="last" if training else "never",
        optimizer_p_fn= config["optimizer_p_fn_mcpc"] if training else optim.SGD,
        optimizer_p_kwargs=config["optimizer_p_kwargs_mcpc"] if training else {"lr": 0.0},
        plot_progress_at=[]
    )
    return mcpc_trainer


def random_step(t,_pc_trainer, var=2.):
    """
    var: needs to be 2. for mathematically correct learning.
    """
    xs = _pc_trainer.get_model_xs()
    optimizer = _pc_trainer.get_optimizer_x()
    # optimizer.zero_grad()
    for x in xs:
        x.grad.normal_(0.,np.sqrt(var/optimizer.defaults['lr']))
    optimizer.step()


def clean_dir(dir):
    for file_name in os.listdir(dir):
        # construct full file path
        file = dir + file_name
        if os.path.isfile(file):
            # print('Deleting file:', file)
            os.remove(file)


def sample_multivariate_Gauss(mean, cov, use_cuda=False):
    L = torch.linalg.cholesky(cov)
    rand = torch.randn((mean.shape[0],mean.shape[1], 1))
    rand = torch.matmul(L, rand).view(mean.shape)
    if use_cuda:
        rand = rand.cuda()
    return (mean + rand).detach()


def sample_pc(num_samples, model, config, use_cuda=False, is_return_hidden=False):
    temp = torch.zeros((num_samples,config["input_size"]))
    if use_cuda:
        temp = temp.cuda()

    for layer_idx in range(len(model)):
        if isinstance(model[layer_idx], pc.PCLayer):
            temp = sample_multivariate_Gauss(temp,torch.eye(temp.shape[1]), use_cuda=use_cuda)
        else:
            temp = model[layer_idx](temp)
    
    if is_return_hidden:
        return temp.detach()

    if config["loss_fn"] == fe_fn:
        temp = sample_multivariate_Gauss(temp, config["input_var"]*torch.eye(temp.shape[1]), use_cuda=use_cuda)
    elif config["loss_fn"] == bernoulli_fn:
        temp = temp.sigmoid()
        temp = (torch.rand_like(temp)<=temp).double()

    return temp.detach()
    


def get_fid(gen_pc, config, use_cuda, n_samples = 5000, is_test=False):
    samples = sample_pc(n_samples, gen_pc, config, use_cuda=use_cuda, is_return_hidden=True)
    images = samples.view(-1,28,28)
    if config["loss_fn"] == fe_fn:
        images = (images > 0).type_as(images)
    if config["loss_fn"] == bernoulli_fn:
        images = images.sigmoid()

    tf = tempfile.NamedTemporaryFile()
    
    new_folder = False
    while not new_folder:
        try:
            new_folder=True
            os.makedirs(tf.name+"_")
            print(tf.name+"_")
        except OSError:
            print("ERROR")
            tf = tempfile.NamedTemporaryFile()
            new_folder=False
    
    for img_idx in range(len(images)):
        save_image(images[img_idx], tf.name+"_"+"\\"+str(img_idx)+".png")

    if is_test:
        result = subprocess.run('python -m pytorch_fid test_img.npz '+ tf.name+"_", stdout=subprocess.PIPE)
    else:
        result = subprocess.run('python -m pytorch_fid val_img.npz '+ tf.name+"_", stdout=subprocess.PIPE)
    shutil.rmtree(tf.name+"_")
    return float(str(result.stdout).split(" ")[2].split("\\r")[0])


def fe_fn_mask(output, _target, _var, perc=0.5):
    return (1/_var)*0.5*(output[:,-round(output.shape[1]*perc):] - _target[:,-round(output.shape[1]*perc):]).pow(2).sum()

def bernoulli_fn_mask(output, _target, _var=None, perc=0.5):
    loss = nn.BCEWithLogitsLoss(reduction='sum')
    return loss(output[:,-round(output.shape[1]*perc):], _target[:,-round(output.shape[1]*perc):])


class Bias(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self):
        return self.bias

class Bias_supervised(nn.Module):
    def __init__(self, num_features, offset=0.):
        super().__init__()
        self.bias = nn.Parameter(offset * torch.ones(num_features))

    def forward(self, x):
        return self.bias + x

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

