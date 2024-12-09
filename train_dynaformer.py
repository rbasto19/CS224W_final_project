import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch.utils.data import Subset
from torch_geometric.utils.convert import to_networkx
import networkx as nx
from networkx import all_pairs_shortest_path
from sklearn.model_selection import train_test_split
import sys
import os
current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, 'models'))
# NOTE: change this to the path for layers and model.py file with dynaformer
import models.layers as layers
import models.model as model
import pickle
from tqdm import tqdm
from torch_geometric.nn.pool import global_mean_pool
from copy import deepcopy
from torch_geometric.utils import remove_isolated_nodes
import torch_geometric.transforms as T
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from copy import deepcopy
from utils import remove_random_edges
import numpy as np

writer = SummaryWriter(log_dir="logs/dynaformer"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

MD = True
if MD:
    MD_str = '_MD'
    dataset = []
    with open('/Users/rbasto/Downloads/md-refined2019-5-5-5/md-refined2019-5-5-5_test.pkl', 'rb') as f:
        dataset_temp = pickle.load(f)
    for i in range(len(dataset_temp)):
        for j in range(len(dataset_temp[i])):
            dataset_temp[i][j] = Data(**dataset_temp[i][j].__dict__)  # allowing to use different pyg version
            dataset_temp[i][j].x = dataset_temp[i][j].x.to(torch.float32)
            dataset.append(dataset_temp[i][j])
    
else:
    MD_str = ''
    with open('/Users/rbasto/Stanford projects/CS224W/refined-set-2020-5-5-5_train_val.pkl', 'rb') as f:
        dataset = pickle.load(f)
    for i in range(len(dataset)):
        dataset[i] = Data(**dataset[i].__dict__)  # allowing to use different pyg version
        dataset[i].x = dataset[i].x.to(torch.float32)

# NOTE: change these hyperparameters if necessary
hidden_dim = 32
graph_model = model.Graphormer(
    num_layers=2,
    input_node_dim=dataset[0].num_node_features,
    node_dim=hidden_dim,
    input_edge_dim=dataset[0].num_edge_features,
    edge_dim=hidden_dim,
    output_dim=1,
    n_heads=16,
    ff_dim=2 * hidden_dim,
    max_in_degree=4,
    max_out_degree=4,
    max_path_distance=4,
    num_heads_spatial=4
)

batch_size = 8
train_ids, test_ids = train_test_split([i for i in range(len(dataset))], test_size=0.3, random_state=42)
train_loader = DataLoader(Subset(dataset, train_ids), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(Subset(dataset, test_ids), batch_size=batch_size, shuffle=True)
# NOTE: change these hyperparameters if necessary
optimizer = torch.optim.AdamW(graph_model.parameters(), lr=1e-6)
loss_function = torch.nn.MSELoss()
p = 0.9

best_val_error = None
for epoch in range(100):
    graph_model.train()
    batch_loss = 0.0
    batch_idx = 0
    for batch in tqdm(train_loader):
        batch = batch.to_data_list()
        for i in range(len(batch)):
            batch[i] = remove_random_edges(batch[i], p)
        batch = Batch.from_data_list(batch)
        batch.to_data_list()
        y = batch.y
        optimizer.zero_grad()
        output = global_mean_pool(graph_model(batch), batch.batch)
        # print(output)
        loss = loss_function(output.flatten(), y.flatten())
        writer.add_scalar("Batch Loss", loss.item(), epoch * len(train_loader) + batch_idx)
        batch_loss += loss.item() * len(y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(graph_model.parameters(), 1)
        optimizer.step()
        batch_idx += 1
    print("TRAIN_LOSS", batch_loss / len(train_loader.dataset))
    writer.add_scalar("Train Loss", batch_loss / len(train_loader.dataset), epoch)
    writer.add_scalar("Train Error", np.sqrt(batch_loss / len(train_loader.dataset)), epoch)
    graph_model.eval()
    batch_loss = 0.0
    for batch in tqdm(test_loader):
        y = batch.y
        # print("Expected")
        # print(y)
        # print("Predicted")
        with torch.no_grad():
            # print(graph_model(batch))
            output = global_mean_pool(graph_model(batch), batch.batch)
            # print(output)
            loss = loss_function(output.flatten(), y.flatten())

        batch_loss += loss.item() * len(y)

    val_loss = batch_loss / len(test_loader.dataset)
    print("EVAL LOSS", val_loss)
    writer.add_scalar("Eval Loss", val_loss, epoch)
    writer.add_scalar("Eval Error", np.sqrt(val_loss), epoch)

    if best_val_error is None or val_loss <= best_val_error:
        best_val_error = val_loss
        best_model = deepcopy(model)
        torch.save(best_model.state_dict(), 'model_checkpoints/dynaformer{MD_str}_hdim_{dim}_batch_{batch}.pt'.format(MD_str=MD_str, dim=hidden_dim, batch=batch_size))
