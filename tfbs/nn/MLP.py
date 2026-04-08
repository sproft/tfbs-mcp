import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F


#load a pytorch model via a checkpitnt file
# Define the MLP Model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, input_dropout, dropout):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            torch.nn.Dropout(p=input_dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)

# Create a Lightning Module
class MLPModule(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, input_dropout, dropout):
        super(MLPModule, self).__init__()
        self.save_hyperparameters()
        self.model = MLP(input_dim, hidden_dim, output_dim, input_dropout, dropout)
        self.loss_fn = F.mse_loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log_dict(self.model.hparams)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

