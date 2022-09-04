from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch
from greycdata.datasets import AlkaneDataset

dataset = AlkaneDataset(root='/tmp/Alkane')

print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

for data in dataset:
    print(data)
    print(data.y)

data = dataset[10]  # Get the first graph object.

print(data.__dict__)
print(data)
print('=============================================================')

# Gather some statistics about the first graph.
print(f'property : {data.y}')
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
