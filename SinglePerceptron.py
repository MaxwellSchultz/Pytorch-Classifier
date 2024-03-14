import math
import numpy as np
import onnx
import pandas as pd
import pprint
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import wandb

wandb.login()

conf = pd.read_table('task1_topics/task1.config', header=None, delim_whitespace=True)

N_TRAIN = conf.loc[0,1]
N_DEV = conf.loc[1,1]
D = conf.loc[2,1]
C = conf.loc[3,1]

print("N_TRAIN",N_TRAIN)
print("N_DEV",N_DEV)
print("D",D)
print("C",C)

# device config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = dict(
    n_train = N_TRAIN,
    n_dev = N_DEV,
    num_features = D,
    epochs = 20,
    classes = C,
    batch_size = 120,
    learning_rate = 0.075,
    hidden_size=10,
    optimizer='sgd',
    dataset="task1_topics")

sweep_config = {
    'method': 'random'
    }

metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric

parameters_dict = {
    'optimizer': {
        'values': ['adam', 'sgd']
        }
    # 'hidden_size': {
    #     'values': [128, 256, 512]
    #     }
    }

parameters_dict.update({
    'epochs': {
        'value': 16}
    ,
    'n_train': {
        'value': int(N_TRAIN)},
    'n_dev': {
        'value': int(N_DEV)},
    'num_features': {
        'value': int(D)},
    'classes': {
        'value': int(C)},
    'hidden_size': {
        'values': [5,10,15]}
    })

parameters_dict.update({
    'learning_rate': {
        # a flat distribution between 0 and 0.1
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
    },
    'batch_size': {
        # integers between 32 and 256
        # with evenly-distributed logarithms 
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 256,
    }
    })

sweep_config['parameters'] = parameters_dict

pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-single")

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        # self.relu = nn.ReLU()
        #self.leaky_relu = nn.LeakyReLU()
        
        self.sigmoid = nn.Sigmoid()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self,x):
        out = self.l1(x)
        out = self.sigmoid(out)
        out = self.l2(out)
        
        return out
    
class SparseCatDataset(Dataset):
    def __init__(self, samples, labels):
        # Load data from files
        self.samples = samples
        self.labels = labels

        # Preprocess data if necessary
        # e.g., convert to tensors, normalize, etc.

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        return sample, label

def model_pipeline(hyperparams=None):
    
    with wandb.init(project="single-perceptron-model", config=hyperparams):
        config = wandb.config
        
        model, train_loader, test_loader, criterion, optimizer = make(config)
        
        train(model, train_loader, criterion, optimizer, config)
        
        test(model, test_loader)

    return model

def make(config):
    # get data
    # train, test = get_data("task1_topics/train.sparseX","task1_topics/train.CT",config.n_train,config.num_features), get_data("task1_topics/dev.sparseX","task1_topics/dev.CT",config.n_dev,config.num_features)
    train, test = get_data("task1_topics/train.sparseX","task1_topics/train.CT",config.n_train,config.num_features,slice=2), get_data("task1_topics/dev.sparseX","task1_topics/dev.CT",config.n_train,config.num_features,slice=2)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)
    
    # get model 
    model = NeuralNet(D,config.hidden_size,C).to(device)
    
    # generate loss and optimizer
    criterion = nn.CrossEntropyLoss()
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer

def get_data(features_file, labels_file, N, D, min_slice=0, max_slice=None, slice=5):
    
    features = np.loadtxt(features_file, dtype=np.int64)
    labels = np.loadtxt(labels_file, dtype=np.int64)
    
    features = torch.tensor(features)
    labels = torch.tensor(labels)
    
    # convert sparse data into indices and values
    indices = torch.tensor([[entry[0], entry[1]] for entry in features], dtype=torch.long).t()
    values = torch.tensor([entry[2] for entry in features], dtype=torch.float)
    
    # create a sparse tensor
    sparse_features = torch.sparse.FloatTensor(indices, values, torch.Size([N, D]))
    
    # convert sparse tensor to dense tensor
    dense_tensor = sparse_features.to_dense()
    
    dataset = SparseCatDataset(dense_tensor,labels)
    
    if max_slice is None: 
        max_slice = len(dataset)
    
    sub_dataset = torch.utils.data.Subset(dataset, indices=range(int(min_slice),int(max_slice),slice))
    
    return sub_dataset

def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return loader

def train(model, loader, criterion, optimizer, config):
    # tell wandb what to watch
    wandb.watch(model, criterion, log="all", log_freq=10)
    
    # run training, tracked by wandb
    total_batches = len(loader) * config.epochs
    example_ct = 0
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        for _, (samples, labels) in enumerate(loader):
            
            loss = train_batch(samples, labels, model, optimizer, criterion)
            example_ct += len(samples)
            batch_ct += 1
            
            # report metrics every 25th batch
            if ((batch_ct+1) % 25) == 0:
                train_log(loss, example_ct, epoch)
                
def train_batch(samples, labels, model, optimizer, criterion):
    samples, labels = samples.to(device), labels.to(device)
    
    # forward pass
    outputs = model(samples)
    loss = criterion(outputs, labels)
    
    # backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # step
    optimizer.step()
    
    return loss

def train_log(loss, example_ct, epoch):
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
    
def test(model, test_loader):
    
    # run on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        for samples, labels in test_loader:
            samples, labels = samples.to(device), labels.to(device)
            outputs = model(samples)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print(f"Accuracy of the model on the {total} " + 
                f"test samples: {correct/total:%}")
    
        wandb.log({"test_accuracy": correct/total})
        
    # save the model in ONNX format
    torch.onnx.export(model, samples, "model.onnx")
    wandb.save("model.onnx")

# wandb.agent(sweep_id, model_pipeline, count=5) 
model = model_pipeline(config)
    
    
    
    
    
    