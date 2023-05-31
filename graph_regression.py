#!/usr/bin/env python
# coding: utf-8

# # Basic GNN Regressor on GREYC chemistry datasets
# inspired and derived from https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html


import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch.nn import Linear, ReLU
from torch_geometric.nn import Sequential, GCNConv
from torch_geometric.nn import global_add_pool, global_max_pool
import wandb

from greycdata.datasets import GreycDataset
from mygnn.learner import Learner, Task
from mygnn.models import GCN_reg, GAT_reg, DiffPool_reg, TopKPool_reg


RATIO_TRAIN = .9
MAX_NB_EPOCHS = 10000
PATIENCE = 500


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

    models = {}
    params = {'hidden_channels': 128}

    models["GCN"] = GCN_reg(input_channels=dataset.num_features,
                            hidden_channels=params['hidden_channels'])
    models["GAT"] = GAT_reg(input_channels=dataset.num_features,
                            hidden_channels=params['hidden_channels'])
    # models["GCN_maxpool"] = GCN_reg(input_channels=dataset.num_features,
    #                                 hidden_channels=params['hidden_channels'],
    #                                 pool=global_max_pool)
    # models["TopKPool"] = TopKPool_reg(input_channels=dataset.num_features,
    #                                   hidden_channels=params['hidden_channels'])

    models["DiffPool"] = DiffPool_reg(input_channels=dataset.num_features,
                                      hidden_channels=params['hidden_channels'])
    results = {name: {"train": [], "test": []} for name in models}

    for name, model in models.items():
        learner = Learner(model, mode=Task.REGRESSION,
                          max_nb_epochs=MAX_NB_EPOCHS,
                          patience=PATIENCE)
        model_params = model.config
        config_model = {"architecture": name,
                        "dataset": str(dataset),
                        "ratio_train": RATIO_TRAIN,
                        "max_nb_epochs": MAX_NB_EPOCHS,
                        "patience": PATIENCE}
        group_name = wandb.util.generate_id()
        for xp in range(10):
            wandb.init(project='gnn_greyc', config=model_params |
                       config_model, group=group_name+"_"+name)
            dataset = dataset.shuffle()

            size_train = int(len(dataset)*RATIO_TRAIN)
            train_dataset = dataset[:size_train]
            test_dataset = dataset[size_train:]

            train_loader = DataLoader(
                train_dataset, batch_size=8, shuffle=True)
            test_loader = DataLoader(
                test_dataset, batch_size=32, shuffle=False)
            learner.reset()
            # Pas de valid/test pour l'instant !
            _ = learner.train(
                train_loader, valid_loader=test_loader,
                wandb_log=True)
            rmse_train = learner.score(train_loader)
            rmse_test = learner.score(test_loader)

            # logging/printing
            print(f"RMSE on train set :{rmse_train:.2f}")
            print(f"RMSE on test set :{rmse_test:.2f}")

            wandb.log({"test/rmse": rmse_test,
                       "train/rmse": rmse_train})

            results[name]["train"].append(rmse_train)
            results[name]["test"].append(rmse_test)
            wandb.finish()

        results[name]["mean_train"] = np.mean(results[name]["train"])
        results[name]["std_train"] = np.std(results[name]["train"])
        results[name]["mean_test"] = np.mean(results[name]["test"])
        results[name]["std_test"] = np.std(results[name]["test"])

        # wandb.log({"test/mean_rmse": results[name]["mean_test"],
        #            "test/std_rmse": results[name]["std_test"],
        #            "train/mean_rmse": results[name]["mean_train"],
        #            "train/std_rmse": results[name]["std_train"]})

    with open("results.pickle", "wb") as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()
