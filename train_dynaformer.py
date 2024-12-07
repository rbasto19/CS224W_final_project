import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch.utils.data import Subset
from torch_geometric.utils.convert import to_networkx
import networkx as nx
from networkx import all_pairs_shortest_path
from sklearn.model_selection import train_test_split
# NOTE: change this to the path for layers and model.py file with dynaformer
import CS224W_final_project.layers as layers
import CS224W_final_project.model as model
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

writer = SummaryWriter(log_dir="logs/dynaformer"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

device = torch.device("mps")

with open('CS224W_final_project/refined-set-2020-5-5-5_test.pkl', 'rb') as f:
  dataset = pickle.load(f)

for i in range(len(dataset)):
  dataset[i] = Data(**dataset[i].__dict__)  # allowing to use different pyg version

transform = T.Compose([T.remove_isolated_nodes.RemoveIsolatedNodes()])
def remove_random_edges(graph, p):    
    graph = deepcopy(graph)
    num_edges = int(graph.edge_index.size()[1] / 2)
    keep_edge = (torch.rand(num_edges) > p).reshape(-1,1)
    keep_edge = torch.hstack((keep_edge, keep_edge)).flatten()
    graph.edge_index = graph.edge_index.T[keep_edge].T
    graph.edge_attr = graph.edge_attr[keep_edge]
    graph = transform(graph)
    return graph

graph_model = model.Graphormer(
    num_layers=2,
    input_node_dim=dataset[0].num_node_features,
    node_dim=512,
    input_edge_dim=dataset[0].num_edge_features,
    edge_dim=512,
    output_dim=1,
    n_heads=32,
    ff_dim=512,
    max_in_degree=4,
    max_out_degree=4,
    max_path_distance=4,
    num_heads_spatial=4
)

train_ids, test_ids = train_test_split([i for i in range(len(dataset))], test_size=0.3, random_state=42)
train_loader = DataLoader(Subset(dataset, train_ids), batch_size=8, shuffle=True)
test_loader = DataLoader(Subset(dataset, test_ids), batch_size=8, shuffle=True)

optimizer = torch.optim.AdamW(graph_model.parameters(), lr=1e-6)
loss_function = torch.nn.L1Loss()
p = 0.9

for epoch in range(10):
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

    print("EVAL LOSS", batch_loss / len(test_loader.dataset))
    writer.add_scalar("Eval Loss", batch_loss / len(test_loader.dataset), epoch)