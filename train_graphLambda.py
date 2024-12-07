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
import model as dynaformer_model
from utils import remove_random_edges
from torch_geometric.data import Data, Batch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

writer = SummaryWriter(log_dir="logs/lambda_"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

with open('CS224W_final_project/refined-set-2020-5-5-5_train_val.pkl', 'rb') as f:
  dataset = pickle.load(f)

for i in range(len(dataset)):
  dataset[i] = Data(**dataset[i].__dict__)  # allowing to use different pyg version
  dataset[i].x = dataset[i].x.to(torch.float32)

class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Net, self).__init__()
        #GCN-representation
        self.input_dim = input_dim
        self.hidden_dim_middle = 2 * hidden_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim_end = int(hidden_dim / 2)
        self.out_dims = [self.hidden_dim, 3*self.hidden_dim, self.hidden_dim_end]
        
        self.conv1 = GCNConv(self.input_dim, self.hidden_dim_middle, cached=False )
        self.bn01 = BatchNorm1d(self.hidden_dim_middle)
        self.conv2 = GCNConv(self.hidden_dim_middle, self.hidden_dim, cached=False )
        self.bn02 = BatchNorm1d(self.hidden_dim)
        self.conv3 = GCNConv(self.hidden_dim, self.hidden_dim, cached=False)
        self.bn03 = BatchNorm1d(self.out_dims[0])
        #GAT-representation
        self.gat1 = GATConv(self.input_dim, self.hidden_dim_middle,heads=3)
        self.bn11 = BatchNorm1d(self.hidden_dim_middle*3)
        self.gat2 = GATConv(self.hidden_dim_middle*3, self.hidden_dim,heads=3)
        self.bn12 = BatchNorm1d(self.hidden_dim*3)
        self.gat3 = GATConv(self.hidden_dim*3, self.hidden_dim,heads=3)
        self.bn13 = BatchNorm1d(self.out_dims[1])
        #GIN-representation
        fc_gin1=Sequential(Linear(self.input_dim, self.hidden_dim_middle), ReLU(), Linear(self.hidden_dim_middle, self.hidden_dim_middle))
        self.gin1 = GINConv(fc_gin1)
        self.bn21 = BatchNorm1d(self.hidden_dim_middle)
        fc_gin2=Sequential(Linear(self.hidden_dim_middle, self.hidden_dim), ReLU(), Linear(self.hidden_dim, self.hidden_dim))
        self.gin2 = GINConv(fc_gin2)
        self.bn22 = BatchNorm1d(self.hidden_dim)
        fc_gin3=Sequential(Linear(self.hidden_dim, self.hidden_dim_end), ReLU(), Linear(self.hidden_dim_end, self.hidden_dim_end))
        self.gin3 = GINConv(fc_gin3)
        self.bn23 = BatchNorm1d(self.out_dims[2])
        #Fully connected layers for concatinating outputs
        self.fc1=Linear(sum(self.out_dims), self.hidden_dim)
        self.dropout1=Dropout(p=0.2,)
        self.fc2=Linear(self.hidden_dim, self.hidden_dim_end)
        self.dropout2=Dropout(p=0.2,)
        self.fc3=Linear(self.hidden_dim_end, 1)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        y=x
        z=x
        #GCN-representation
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn01(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn02(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn03(x)
        x = global_mean_pool(x, data.batch)
        #GAT-representation
        y = F.relu(self.gat1(y, edge_index))
        y = self.bn11(y)
        y = F.relu(self.gat2(y, edge_index))
        y = self.bn12(y)
        y = F.relu(self.gat3(y, edge_index))
        y = self.bn13(y)
        y = global_mean_pool(y, data.batch)
        #GIN-representation
        z = F.relu(self.gin1(z, edge_index))
        z = self.bn21(z)
        z = F.relu(self.gin2(z, edge_index))
        z = self.bn22(z)
        z = F.relu(self.gin3(z, edge_index))
        z = self.bn23(z)
        z = global_mean_pool(z, data.batch)
        #Concatinating_representations
        cr=torch.cat((x,y,z),1)
        cr = F.relu(self.fc1(cr))
        cr = self.dropout1(cr)
        cr = F.relu(self.fc2(cr))
        cr = self.dropout2(cr)
        cr = self.fc3(cr)
        # print(cr)
        cr = F.relu(cr).view(-1)
        return cr  

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
        # batch = data.to_data_list()
        # for i in range(len(batch)):
        #     batch[i] = remove_random_edges(batch[i], p)
        # data = Batch.from_data_list(batch)
        # data.to_data_list()
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

device = 'cpu'
best_val_error = None
best_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_errors, valid_errors,test_errors = [], [],[]
model = Net(dataset[0].num_node_features, 64).to(device)
lr = 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                factor=0.95, patience=2,
                                min_lr=1e-6)
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

    print('Epoch: {:03d}, LR: {:.7f}, Loss: {:.7f}, Validation MAE: {:.7f}'
        .format(epoch, lr, loss, val_error))
print('leng of test errors = ', len(test_errors))
