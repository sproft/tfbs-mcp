import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

######KEEEP FOR NOW AS BACKUP
class my_cnn(nn.Module):
    def __init__(self, input_dropout):
        super(my_cnn, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.layers(x)


class CNNModule(pl.LightningModule):
    def __init__(self, input_dropout):
        super(CNNModule, self).__init__()
        self.save_hyperparameters()
        self.loss_fn = F.mse_loss
        self.model = my_cnn(input_dropout)

    def forward(self, x):
        return self.model(x)

    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #self.log_dict(self.model.hparams)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)