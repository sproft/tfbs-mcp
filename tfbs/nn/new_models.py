import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

class BaseModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, classify=False):
        super(BaseModel, self).__init__()
        self.save_hyperparameters()
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

class SimpleCNN(BaseModel):
    def __init__(self, input_channels=4, sequence_length=14, learning_rate=1e-3, classify=False):
        super(SimpleCNN, self).__init__(learning_rate, classify)
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * (sequence_length // 2), 128)  # Adjusted for sequence length
        self.fc2 = nn.Linear(128, 1)  # Output a single float value

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        if self.classify:
            x = torch.sigmoid(self.fc2(x))
        else:
            x = self.fc2(x)
        return x

class MLP(BaseModel):
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=1e-3, classify=False):
        super(MLP, self).__init__(learning_rate, classify)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.classify:
            x = torch.sigmoid(self.fc3(x))
        else:
            x = self.fc3(x)
        return x

class BPNet(BaseModel):
    def __init__(self, input_length, learning_rate=1e-3, classify=False):
        super(BPNet, self).__init__(learning_rate, classify)
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=21, padding=10)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=21, padding=10)
        self.dilated_conv1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=21, padding=20, dilation=2)
        self.dilated_conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=21, padding=40, dilation=4)
        self.dilated_conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=21, padding=80, dilation=8)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * (input_length // 2), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.dilated_conv1(x))
        x = F.relu(self.dilated_conv2(x))
        x = F.relu(self.dilated_conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.classify:
            x = torch.sigmoid(self.fc3(x))
        else:
            x = self.fc3(x)
        return x

class MyModel(BaseModel):
    def __init__(self, learning_rate=1e-3, classify=False):
        super(MyModel, self).__init__(learning_rate, classify)
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=100, kernel_size=5, padding="valid")
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.3)
        self.gru1 = nn.GRU(100, 100, 1, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc1 = nn.Linear(1000, 100)
        self.dropout2 = nn.Dropout(0.6)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = x.permute(0, 2, 1)  # Change shape from (batch_size, channels, seq_len) to (batch_size, seq_len, channels)
        x, _ = self.gru1(x)
        x = x.contiguous().view(x.size(0), -1)  # Flatten the GRU output
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        if self.classify:
            x = torch.sigmoid(self.fc2(x))
        else:
            x = self.fc2(x)
        return x