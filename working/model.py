import pytorch_lightning as pl
import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from torch.nn import Linear, CrossEntropyLoss, functional as F


class MyModel(pl.LightningModule):
    def __init__(self, net, learning_rate, decay_factor):
        super().__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # optimizer paramters
        self.lr = learning_rate
        self.decay_factor = decay_factor
        self.net = net

    def forward(self, x):
        return self.net(x)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = LambdaLR(optimizer, lambda epoch: self.decay_factor**epoch)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train/batch/loss", loss, prog_bar=False)

        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()
        acc = accuracy_score(y_true, y_pred)
        self.log("train/batch/acc", acc)
        self.training_step_outputs.append({"loss": loss, "y_true": y_true, "y_pred": y_pred})
        return {"loss": loss, "y_true": y_true, "y_pred": y_pred}

    def on_train_epoch_end(self):
        loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in self.training_step_outputs:
            loss = np.append(loss, results_dict["loss"].cpu().detach().numpy())
            y_true = np.append(y_true, results_dict["y_true"])
            y_pred = np.append(y_pred, results_dict["y_pred"])
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        self.log("train/epoch/loss", loss.mean())
        self.log("train/epoch/acc", acc)
        self.log("train/epoch/auc", auc)
        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        print(y_hat)
        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()
        print("======  prediction =====")
        print(y_pred)
        print("===== GT =====")
        print(y_true)
        self.validation_step_outputs.append({"loss": loss, "y_true": y_true, "y_pred": y_pred})

        return {"loss": loss, "y_true": y_true, "y_pred": y_pred}
    
    def on_validation_epoch_end(self):
        loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in self.validation_step_outputs:
            loss = np.append(loss, results_dict["loss"].cpu().detach().numpy())
            y_true = np.append(y_true, results_dict["y_true"])
            y_pred = np.append(y_pred, results_dict["y_pred"])
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        self.log("val/loss", loss.mean())
        self.log("val/acc", acc)
        self.log("val/auc", auc)
        self.validation_step_outputs.clear()  # free memory
