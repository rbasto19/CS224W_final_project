import torch
from copy import deepcopy
import torch_geometric.transforms as T

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