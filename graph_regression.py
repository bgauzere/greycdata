#!/usr/bin/env python
# coding: utf-8

# # Basic GNN Regressor on GREYC chemistry datasets
# inspired and derived from https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html


import json
import numpy as np
import pickle
from torch_geometric.loader import DataLoader
import wandb
import random
import torch

from greycdata.datasets import GreycDataset
from mygnn.learner import Learner, Task
from mygnn.models import GCN_reg, GAT_reg, DiffPool_reg, TopKPool_reg


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

    seed = random.randint(0, 32492)
    torch.manual_seed(seed)
    np.random.seed(seed)

    with open("config.json", "r") as f:
        config_learning = json.load(f)

    # config_learning = {
    #     "hidden_channels": 128,
    #     "ratio_train": RATIO_TRAIN,
    #     "max_nb_epochs": MAX_NB_EPOCHS,
    #     "patience": PATIENCE,
    #     "dataset": str(dataset),
    #     "learning_rate": 0.03,
    #     "batch_size_train": 32

    # }

    models_dict = config_learning["models"]
    for name, config in models_dict.items():
        class_model = globals()[config["class"]]
        models[name] = class_model(**config["init_params"])

    # "GCN"] = GCN_reg(input_channels=dataset.num_features,
    #                         hidden_channels=config_learning['hidden_channels'])
    # models["GAT"] = GAT_reg(input_channels=dataset.num_features,
    #                         hidden_channels=config_learning['hidden_channels'])
    # models["GCN_maxpool"] = GCN_reg(input_channels=dataset.num_features,
    #                                 hidden_channels=params['hidden_channels'],
    #                                 pool=global_max_pool)
    # models["TopKPool"] = TopKPool_reg(input_channels=dataset.num_features,
    #                                   hidden_channels=params['hidden_channels'])

    # models["DiffPool"] = DiffPool_reg(input_channels=dataset.num_features,
    #                                   hidden_channels=config_learning['hidden_channels'])

    results = {name: {"train": [], "valid": [], "test": []} for name in models}

    for name, model in models.items():
        learner = Learner(model, mode=Task.REGRESSION,
                          max_nb_epochs=config_learning["max_nb_epochs"],
                          patience=config_learning["patience"],
                          ratio_train_valid=[
                              config_learning["ratio_train"],
                              1-config_learning["ratio_train"]],
                          learning_rate=config_learning["learning_rate"])
        model_params = model.config
        group_name = wandb.util.generate_id()

        # Pour un split de test, va générer 3 perfs de test et 10 de valid
        dataset = dataset.shuffle()

        size_train = int(len(dataset)*config_learning["ratio_train"])
        train_dataset = dataset[:size_train]
        test_dataset = dataset[size_train:]

        train_loader = DataLoader(
            train_dataset, batch_size=config_learning["batch_size_train"], shuffle=True)
        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False)

        config_xp = {"architecture": name, "mode": "valid", "seed": seed}

        # Train/Valid to evaluate perf of hyperparameters
        for xp_valid in range(10):

            wandb.init(project='gnn_greyc', config=model_params |
                       config_learning | config_xp,
                       group=group_name+"_"+name)
            learner.reset()
            # pas de train/valid predétermine
            learner.train(
                train_loader,
                wandb_log=True)

            rmse_train, rmse_valid = learner.best_score()

            results[name]["train"].append(rmse_train)
            results[name]["valid"].append(rmse_valid)
            wandb.log({"train/best_rmse": rmse_train,
                       "valid/best_rmse": rmse_valid})
            wandb.finish()

        results[name]["mean_train"] = np.mean(results[name]["train"])
        results[name]["std_train"] = np.std(results[name]["train"])
        results[name]["mean_valid"] = np.mean(results[name]["valid"])
        results[name]["std_valid"] = np.std(results[name]["valid"])

        # Evaluate on test set
        config_xp["mode"] = "test"
        for xp_test in range(3):
            wandb.init(project='gnn_greyc', config=model_params |
                       config_learning | config_xp,
                       group=group_name+"_"+name)

            learner.reset()
            learner.train(
                train_loader,
                wandb_log=True)
            # eval
            rmse_train, rmse_valid = learner.best_score()
            rmse_test = learner.score(test_loader)
            # log
            results[name]["train"].append(rmse_train)
            results[name]["valid"].append(rmse_valid)
            results[name]["test"].append(rmse_test)
            wandb.log({"train/rmse": rmse_train,
                       "valid/rmse": rmse_valid,
                       "test/rmse": rmse_test})

            wandb.finish()

    with open("results.pickle", "wb") as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()
