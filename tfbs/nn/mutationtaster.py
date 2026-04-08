import math
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class BaseModel(pl.LightningModule):
    def __init__(
            self,
            input_length: int,
            learning_rate: float = 1e-3,
            classify: bool = False,
            **kwargs):
        super(BaseModel, self).__init__()
        self.save_hyperparameters()
        self.input_length = input_length
        self.learning_rate = learning_rate
        self.classify = classify
        if self.classify:
            self.loss = F.binary_cross_entropy
            self.accuracy = torchmetrics.Accuracy(task="binary")
            self.auroc = torchmetrics.AUROC(task="binary")
        else:
            self.loss = F.mse_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.classify:
            y_pred = torch.round(y_hat)
            acc = self.accuracy(y_pred, y)
            auc = self.auroc(y_hat, y)
            self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_auc', auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.classify:
            y_pred = torch.round(y_hat)
            acc = self.accuracy(y_pred, y)
            auc = self.auroc(y_hat, y)
            self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('test_auc', auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def create_windows(self, x, step_size=1):
        """
        Create overlapping windows from the input tensor.
        """
        window_size = self.input_length
        windows = []
        for i in range(0, x.size(2) - window_size + 1, step_size):
            windows.append(x[:, :, i:i + window_size])
        return windows
    


