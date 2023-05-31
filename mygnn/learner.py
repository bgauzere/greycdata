from enum import Enum, auto
import numpy as np
from tqdm import tqdm
import torch
from mygnn.utils import rmse, nb_correct_classif
import wandb
from torch.utils.data import random_split


class Task(Enum):
    REGRESSION = auto()
    CLASSIF = auto()


class Learner():
    """
    Class to encapsulate learning process, either classif or regression

    """

    def __init__(self, model, mode=Task.CLASSIF, optimizer=None,
                 max_nb_epochs=10000,
                 patience=100):
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
        if self.mode == Task.CLASSIF:
            self.criterion = torch.nn.CrossEntropyLoss()
        elif self.mode == Task.REGRESSION:
            self.criterion = torch.nn.MSELoss(reduction="sum")
        else:
            raise Exception(f"Unknown mode !! {mode}")

        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
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
            train_loader, valid_loader = random_split(train_loader, [.9, .1])
        losses = {"valid": [],
                  "train": []}

        self.model.train()
        for epoch in iter_display(range(1, self.max_nb_epochs)):
            self.model.train()
            loss_epoch = 0.0
            for data in train_loader:  # Iterate in batches over the training dataset.
                # Perform a single forward pass.
                loss = self._compute_loss_batch(data)
                loss_epoch += loss.item()
                loss.backward()  # Derive gradients.
                self.optimizer.step()  # Update parameters based on gradients.
                self.optimizer.zero_grad()  # Clear gradients.
            # Calcul de la loss en valid
            loss_epoch = loss_epoch/len(train_loader.dataset)
            self.model.eval()
            loss_valid = 0.0
            for data_valid in valid_loader:
                loss = self._compute_loss_batch(data_valid)
                loss_valid += loss.item()
            loss_valid = loss_valid/len(valid_loader.dataset)
            if verbose:
                tqdm.write(f"{epoch} : {loss_epoch}", end="\r")
            if wandb_log:
                wandb.log({"train/loss": loss_epoch,
                           "valid/loss": loss_valid})

            # sauvegarde du meilleur modele
            if (epoch > 1):
                if (loss_valid < min(losses['valid'])):
                    torch.save(self.model, 'best_model.pth')
                    self.best_model = torch.load('best_model.pth')
                    self.best_model.eval()
            losses['train'].append(loss_epoch)
            losses['valid'].append(loss_valid)

            # test patience (a tester sur val/train et loss/acc)
            if (epoch > self.patience) and np.argmin(losses['valid'][-self.patience:]) == 0:
                # Pas de meilleur loss sur les patience dernieres epochs
                # reste à retourner le bon modele !
                return losses

        return losses

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

    def _predict_batch(self, data_batch):
        """
        test mode. Use the best model
        """
        out = self.best_model(
            data_batch.x, data_batch.edge_index, data_batch.batch)
        if self.mode == Task.CLASSIF:
            out = out.argmax(dim=1)
        return out

    def predict(self, loader):
        """
        returns the predictions for data in loader, one item by batch
        """
        self.best_model.eval()  # redondant
        predictions = []
        # Iterate in batches over the training/test dataset.
        for data in loader:
            predictions.append(self._predict_batch(data))
        return predictions

    def score(self, loader):
        if self.mode == Task.CLASSIF:
            return self._score_clf(loader)
        return self._score_reg(loader)

    def _score_reg(self, loader):
        pred = []
        gt = []
        for data in loader:
            pred.extend(self._predict_batch(data))
            gt.extend(data.y)

        return rmse(np.array([i.item() for i in gt]),
                    np.array([i.item() for i in pred]))

    def _score_clf(self, loader):
        preds = self.predict(loader)
        correct = 0.0
        for batch, data in enumerate(loader):
            correct += nb_correct_classif(preds[batch], data.y)
            # Check against ground-truth labels.
        # Derive ratio of correct predictions.
        return correct / len(loader.dataset)
