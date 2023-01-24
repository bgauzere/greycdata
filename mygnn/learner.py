from enum import Enum, auto
import numpy as np
from tqdm import tqdm
import torch
from mygnn.utils import rmse, nb_correct_classif


class Task(Enum):
    REGRESSION = auto()
    CLASSIF = auto()


class Learner():
    """
    Class to encapsulate learning process, either classif or regression

    """

    def __init__(self, model, mode=Task.CLASSIF, optimizer=None):
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

    def train(self, loader, nb_epochs=1000, verbose=True):
        """
        train the model for one epoch and return the loss

        Parameters

        ----------

        loader : torch_geometric.loader.DataLoader
        """

        def iter_display(x): return x
        if verbose:
            iter_display = tqdm
        losses = []
        self.model.train()
        for epoch in iter_display(range(1, nb_epochs)):
            loss_epoch = 0.0
            for data in loader:  # Iterate in batches over the training dataset.
                # Perform a single forward pass.
                out = self.model(data.x, data.edge_index,
                                 data.batch)
                # Compute the loss.
                # FIXME: .reshape(-1,1) our la reg
                y_gt = data.y
                if self.mode == Task.REGRESSION:
                    y_gt = data.y.reshape(-1, 1)
                loss = self.criterion(out, y_gt)
                loss_epoch += loss.item()
                loss.backward()  # Derive gradients.
                self.optimizer.step()  # Update parameters based on gradients.
                self.optimizer.zero_grad()  # Clear gradients.
                loss_epoch += loss.item()
            if verbose:
                tqdm.write(f"{epoch} : {loss_epoch}", end="\r")
            losses.append(loss_epoch)
        return losses

    def _predict_batch(self, data_batch):
        out = self.model(data_batch.x, data_batch.edge_index,
                         data_batch.batch)
        if self.mode == Task.CLASSIF:
            out = out.argmax(dim=1)
        return out

    def predict(self, loader):
        """
        returns the predictions for data in loader, one item by batch
        """
        self.model.eval()
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
