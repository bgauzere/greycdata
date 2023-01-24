#!/usr/bin/env python
# coding: utf-8

# # Basic GNN Regressor on GREYC chemistry datasets
# inspired and derived from https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html


import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from greycdata.datasets import GreycDataset
from mygnn.learner import Learner, Task
from mygnn.models import GCN_reg

RATIO_TRAIN = .9


def main():
    dataset = GreycDataset(name='Acyclic', root='data/Acyclic')

    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    data = dataset[0]  # Get a graph object.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    print(data.x)

    dataset = dataset.shuffle()

    size_train = int(len(dataset)*RATIO_TRAIN)
    train_dataset = dataset[:size_train]
    test_dataset = dataset[size_train:]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(dataset[0].y)

    for step, data in enumerate(train_loader):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data)
        print()

    model = GCN_reg(input_channels=dataset.num_features,
                    hidden_channels=128)

    learner = Learner(model, mode=Task.REGRESSION)
    losses = []
    losses = learner.train(train_loader, nb_epochs=1000)
    plt.plot(losses)
    plt.show()
    rmse_train = learner.score(train_loader)
    print(f"RMSE on train set :{rmse_train:.2f}")
    rmse_test = learner.score(test_loader)
    print(f"RMSE on test set :{rmse_test:.2f}")


if __name__ == '__main__':
    main()
