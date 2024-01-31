import torch
import torch.nn as nn
from torch.nn import Linear

from torch_geometric.nn import GCN 
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv
from torch_geometric.nn import global_add_pool, global_max_pool
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import Sequential
from torch_geometric.nn.dense import dense_diff_pool, DenseGCNConv
from torch_geometric.utils import to_dense_adj, to_dense_batch, dense_to_sparse

class GNN_clf(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_classes=2,
                 conv=GCN, pool=global_add_pool,
                 num_layers = 2):
        super().__init__()
        self.conv = conv(input_channels, hidden_channels, 
                         num_layers=num_layers,
                           out_channels=num_classes)
        self.pool = pool
        self.config = {'hidden_channels': hidden_channels,
                       'nb_conv_layers': num_layers,
                       'pooling': self.pool,
                       'conv':self.conv}
        
    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self,x, edge_index, batch):
        """
        data : DataLoader
        """
        x = self.conv(x, edge_index)
        x = self.pool(x, batch)
        return x


class GNN_reg(torch.nn.Module):
    """
    Module generique pour intégrer n'importe quel modele de pytorch geometric (https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#models)
    """
    def __init__(self, input_channels, hidden_channels,
                 conv=GCN, pool=global_add_pool,
                 num_layers = 2):
        super().__init__()
        self.conv = conv(input_channels, hidden_channels, 
                         num_layers=num_layers,
                           out_channels=1)
        self.pool = pool
        self.config = {'hidden_channels': hidden_channels,
                       'nb_conv_layers': num_layers,
                       'pooling': self.pool,
                       'conv':self.conv}
        
    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self,x, edge_index, batch):
        """
        data : DataLoader
        """
        x = self.conv(x, edge_index)
        x = self.pool(x, batch)
        return x


        
class GCN_reg(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, pool=global_add_pool):
        super().__init__()
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)
        self.pool = pool
        nn.init.xavier_uniform_(self.conv1.lin.weight)
        self.config = {'hidden_channels': hidden_channels,
                       'nb_conv_layers': 2,
                       'pooling': self.pool}

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        x = self.pool(x, batch)
        x = self.lin(x)
        return x


class GAT_reg(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        self.conv1 = GATv2Conv(input_channels, hidden_channels)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)
        self.pool = global_add_pool
        self.config = {'hidden_channels': hidden_channels,
                       'nb_conv_layers': 2,
                       'pooling': self.pool}

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        x = self.pool(x, batch)
        x = self.lin(x)
        return x


class TopKPool_reg(torch.nn.Module):
    """
    """

    def __init__(self, input_channels, hidden_channels, num_nodes=7):
        super().__init__()
        self.gnn1_embed = GCNConv(input_channels, hidden_channels)
        self.gnn2_embed = GCNConv(hidden_channels, hidden_channels)
        self.pool = TopKPooling(hidden_channels)
        self.lin = Linear(hidden_channels, 1)
        self.config = {'hidden_channels': hidden_channels,
                       'nb_conv_layers': 2,
                       'pooling': self.pool,
                       'nb_nodes': num_nodes}

    def reset_parameters(self):
        self.gnn1_embed.reset_parameters()
        self.gnn2_embed.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index, batch):
        x_l1 = self.gnn1_embed(x, edge_index)
        x_l1.relu()
        x_p1, edge_index_p1, _, batch, _, _ = self.pool(
            x_l1, edge_index, batch=batch)
        x_l2 = self.gnn2_embed(x_p1, edge_index_p1)
        x_l2 = global_add_pool(x_l2, batch)
        x_out = self.lin(x_l2)
        return x_out


class DiffPool_reg(torch.nn.Module):
    """
    Diffpool avec deux convs. Diffpool est entre les deux
    adapté de https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial16/Tutorial16.ipynb
    """

    def __init__(self, input_channels, hidden_channels, num_nodes=5):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_nodes = num_nodes

        self.gnn_pool = DenseGCNConv(input_channels, self.num_nodes)

        self.gnn1_embed = DenseGCNConv(input_channels, self.hidden_channels)
        self.gnn2_embed = DenseGCNConv(
            self.hidden_channels, self.hidden_channels)
        self.pool = dense_diff_pool
        self.lin = Linear(self.hidden_channels, 1)
        self.config = {'hidden_channels': hidden_channels,
                       'nb_conv_layers': 2,
                       'pooling': self.pool,
                       'nb_nodes': num_nodes}

    def reset_parameters(self):
        self.gnn_pool.reset_parameters()
        self.gnn1_embed.reset_parameters()
        self.gnn2_embed.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index, batch):
        x_dense, mask = to_dense_batch(x, batch)
        nb_graphs, _, _ = x_dense.shape

        adj = to_dense_adj(edge_index, batch)

        s = self.gnn_pool(x_dense, adj, mask)
        x_l1 = self.gnn1_embed(x_dense, adj, mask)

        x_p1, adj_p1, _, _ = self.pool(x_l1, adj, s, mask=mask)

        x_l2 = self.gnn2_embed(x_p1, adj_p1)

        batch_reduced = torch.LongTensor(
            sum([[i]*self.num_nodes for i in range(nb_graphs)], []))

        x_l2 = global_add_pool(
            x_l2.reshape(-1, self.hidden_channels), batch_reduced)
        x_out = self.lin(x_l2)
        return x_out
