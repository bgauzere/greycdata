#!/usr/bin/env python
# coding: utf-8

# # Basic GNN Regressor on GREYC chemistry datasets
# inspired and derived from https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html


import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from greycdata.datasets import GreycDataset
from mygnn.learner import Learner, Task

from mygnn.models import GNN_clf

RATIO_TRAIN = .9


def main():
    dataset = GreycDataset(name='MAO', root='data/MAO')

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

    # In[7]:

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(dataset[0].y)

    for step, data in enumerate(train_loader):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data)
        print()

    model = GNN_clf(input_channels=dataset.num_features,
                    hidden_channels=128)

    learner = Learner(model,max_nb_epochs=100)
    losses = []

    learner.train(train_loader)
    plt.plot(learner.losses["train"],label="train")
    plt.plot(learner.losses["valid"],label="valid")
    plt.legend()
    plt.show()
    acc_train = learner.score(train_loader)
    print(f"Accuracy on train set :{acc_train:.2f}")
    acc_test = learner.score(test_loader)
    print(f"Accuracy on test set :{acc_test:.2f}")


if __name__ == '__main__':
    main()
