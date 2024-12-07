import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import copy
from os.path import join
import random
import torch
import deepdish as dd
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d, Dropout
from torch_geometric.nn import NNConv, Set2Set, GCNConv, global_add_pool, global_mean_pool,GATConv,GINConv
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset #easily fits into cpu memory
from torch.utils.data import Subset
import pickle
# NOTE: change this to the path for model.py file with dynaformer
import model as dynaformer_model
from utils import remove_random_edges
from torch_geometric.data import Data, Batch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from graphLambda_model import Net
writer = SummaryWriter(log_dir="logs/fused_"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

with open('CS224W_final_project/refined-set-2020-5-5-5_train_val.pkl', 'rb') as f:
  dataset = pickle.load(f)

for i in range(len(dataset)):
  dataset[i] = Data(**dataset[i].__dict__)  # allowing to use different pyg version
  dataset[i].x = dataset[i].x.to(torch.float32)

train_ids, val_ids = train_test_split([i for i in range(len(dataset))], test_size=0.3, random_state=42)
train_loader = DataLoader(Subset(dataset, train_ids), batch_size=32, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_ids), batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
p = 0.9
loss_function = torch.nn.MSELoss()
def train(model, train_loader,epoch,device,optimizer):
    model.train()
    loss_all = 0
    error = 0
    batch_idx = 0
    for data in tqdm(train_loader):
        batch = data.to_data_list()
        for i in range(len(batch)):
            batch[i] = remove_random_edges(batch[i], p)
        data = Batch.from_data_list(batch)
        data.to_data_list()
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_function(model(data), data.y)
        writer.add_scalar("Train Loss (batch)", loss, (epoch - 1) * len(train_loader) + batch_idx)
        loss.backward()
        if isinstance(loss_function, torch.nn.MSELoss):           
            loss_all += torch.sqrt(loss).item() * len(data.y)
        else:
            loss_all += loss.item() * len(data.y)
        # error += (model(data) - data.y).abs().sum().item()  # MAE
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        batch_idx += 1
    return loss_all / len(train_loader.dataset)


@torch.no_grad()
def test(model, loader,device):
    model.eval()
    loss_all = 0
    # print("Expected")
    # for data in loader:
    #     print(data.y)
    # print("Predicted")

    for data in loader:
        data = data.to(device)
        # error += (model(data) - data.y).abs().sum().item()  # MAE
        # print(model(data))
        if isinstance(loss_function, torch.nn.MSELoss):           
            loss_all += torch.sqrt(loss_function(model(data), data.y)).item() * len(data.y)
        else:
            loss_all += loss_function(model(data), data.y).item() * len(data.y)
    return loss_all / len(loader.dataset)


@torch.no_grad()
def test_predictions(model, loader):
    model.eval()
    pred = []
    true = []
    for data in loader:
        data = data.to(device)
        pred += model(data).detach().cpu().numpy().tolist()
        true += data.y.detach().cpu().numpy().tolist()
    return pred, true

best_val_error = None
best_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_errors, valid_errors,test_errors = [], [],[]
model = Net(dataset[0].num_node_features, dataset[0].num_edge_features, 32).to(device)
lr = 1e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                factor=0.5, patience=2,
                                min_lr=1e-7)
for epoch in range(1, 1001):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train(model, train_loader,epoch,device,optimizer)
    writer.add_scalar("Train error (epoch)", loss, epoch)
    # writer.add_scalar("Train error", train_error, epoch)
    val_error = test(model, val_loader,device)
    writer.add_scalar("Val error (epoch)", val_error, epoch)
    # train_errors.append(train_error)
    valid_errors.append(val_error)

    if best_val_error is None or val_error <= best_val_error:
        best_val_error = val_error
        best_model = copy.deepcopy(model)

    print('Epoch: {:03d}, LR: {:.7f}, Loss: {:.7f}, Val error: {:.7f}'
        .format(epoch, lr, loss, val_error))
print('leng of test errors = ', len(test_errors))
