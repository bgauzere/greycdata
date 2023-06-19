from enum import Enum, auto
import numpy as np
from tqdm import tqdm
import torch
from mygnn.utils import rmse, nb_correct_classif
import wandb
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader


class Task(Enum):
    REGRESSION = auto()
    CLASSIF = auto()


class Learner():
    """
    Class to encapsulate learning process, either classif or regression

    """

    def __init__(self, model, mode=Task.CLASSIF, optimizer=None,
                 max_nb_epochs=10000,
                 patience=100,
                 ratio_train_valid=[.9, .1],
                 learning_rate=0.03):
        """Initialisation of the learner

        Parameters

        ----------

        model : torch.nn.Module

        the gnn to use

        optimizer :

        the  optimizer to update the weights (generally Adam)
        mode : "classif"|"reg"
            Torch.nn loss (crossentropy for classif)

        Examples
        --------
        FIXME: Add docs.

        """

        self.model = model
        self.best_model = None  # meilleur modele appris
        self.mode = mode
        self.ratio_train_valid = ratio_train_valid
        if self.mode == Task.CLASSIF:
            self.criterion = torch.nn.CrossEntropyLoss()
        elif self.mode == Task.REGRESSION:
            self.criterion = torch.nn.MSELoss(reduction="sum")
        else:
            raise Exception(f"Unknown mode !! {mode}")
        self.lr = learning_rate
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.optimizer = optimizer
        self.max_nb_epochs = max_nb_epochs
        self.patience = patience

    def reset(self):
        self.model.reset_parameters()

    def train(self, train_loader, valid_loader=None,  verbose=True, wandb_log=False):
        """
        train the model for one epoch and return the loss

        Parameters

        ----------

        loader : torch_geometric.loader.DataLoader
        """

        if verbose:
            iter_display = tqdm
        else:
            def iter_display(x): return
        if valid_loader is None:
            train_subset, valid_subset = random_split(train_loader.dataset,
                                                      self.ratio_train_valid)

            train_loader = DataLoader(train_subset,
                                      batch_size=train_loader.batch_size,
                                      shuffle=True)
            valid_loader = DataLoader(valid_subset,
                                      batch_size=len(valid_subset),
                                      shuffle=False)

        self.losses = {"valid": [],
                       "train": []}
        self.scores = {"valid": [],
                       "train": []}

        self.model.train()
        for epoch in iter_display(range(0, self.max_nb_epochs)):
            self.model.train()
            loss_train = 0.0
            for data in train_loader:  # Iterate in batches over the training dataset.
                # Perform a single forward pass.
                loss = self._compute_loss_batch(data)
                loss_train += loss.item()
                loss.backward()  # Derive gradients.
                self.optimizer.step()  # Update parameters based on gradients.
                self.optimizer.zero_grad()  # Clear gradients.
            # Calcul de la loss en valid
            loss_train = loss_train/len(train_loader.dataset)
            self.model.eval()
            loss_valid = 0.0
            for data_valid in valid_loader:
                loss = self._compute_loss_batch(data_valid)
                loss_valid += loss.item()
            loss_valid = loss_valid/len(valid_loader.dataset)

            score_train = self.score(train_loader, current=True)
            score_valid = self.score(valid_loader, current=True)

            if verbose:
                tqdm.write(f"{epoch} : {loss_train}", end="\r")
            if wandb_log:
                wandb.log({"train/loss": loss_train,
                           "valid/loss": loss_valid,
                           "valid/score": score_valid,
                           "train/score": score_train})

            # sauvegarde du meilleur modele
            if (epoch > 1):
                if (loss_valid < min(self.losses['valid'])):
                    torch.save(self.model, 'best_model.pth')
                    self.best_model = torch.load('best_model.pth')
                    self.best_model.eval()
                    self.best_epoch = epoch

            self.losses['train'].append(loss_train)
            self.losses['valid'].append(loss_valid)
            self.scores['train'].append(score_train)
            self.scores['valid'].append(score_valid)

            # test patience (a tester sur val/train et loss/acc)
            if (epoch > self.patience) and np.argmin(self.losses['valid'][-self.patience:]) == 0:
                # Pas de meilleur loss sur les patience dernieres epochs
                # reste à retourner le bon modele !
                return

        return

    def min_loss(self):
        """
        returns optimal losses on train and valid according to patience
        """
        return self.losses['train'][self.best_epoch], self.losses['valid'][self.best_epoch]

    def best_score(self):
        """
        returns optimal losses on train and valid according to patience
        """
        return self.scores['train'][self.best_epoch], self.scores['valid'][self.best_epoch]

    def _compute_loss_batch(self, data):
        '''
        Compute the loss for a given batch for the current model (not the best one)
        '''
        out = self.model(data.x, data.edge_index, data.batch)
        # FIXME: .reshape(-1,1) our la reg
        y_gt = data.y
        if self.mode == Task.REGRESSION:
            y_gt = data.y.reshape(-1, 1)
        loss = self.criterion(out, y_gt)

        return loss

    def _predict_batch(self, data_batch, model=None):
        """
        test mode. Use the best model by default otherwise model if specified
        """
        if model is None:
            model = self.best_model
        out = model(
            data_batch.x, data_batch.edge_index, data_batch.batch)
        if self.mode == Task.CLASSIF:
            out = out.argmax(dim=1)
        return out

    def predict(self, loader, model):
        """
        returns the predictions for data in loader, one item by batch
        """
        model.eval()  # redondant
        predictions = []
        # Iterate in batches over the training/test dataset.
        for data in loader:
            predictions.append(self._predict_batch(data, model))
        return predictions

    def score(self, loader, current=False):
        model = self.best_model
        if current:
            model = self.model
        if self.mode == Task.CLASSIF:
            return self._score_clf(loader, model)
        return self._score_reg(loader, model)

    def _score_reg(self, loader, model):
        pred = []
        gt = []
        for data in loader:
            pred.extend(self._predict_batch(data, model))
            gt.extend(data.y)

        return rmse(np.array([i.item() for i in gt]),
                    np.array([i.item() for i in pred]))

    def _score_clf(self, loader, model):
        preds = self.predict(loader, model)
        correct = 0.0
        for batch, data in enumerate(loader):
            correct += nb_correct_classif(preds[batch], data.y)
            # Check against ground-truth labels.
        # Derive ratio of correct predictions.
        return correct / len(loader.dataset)
