import math
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

# This import is for the EquiNet model specifically.
# Ensure you have the 'equirc' package installed if you plan to use it.
# pip install equirc
try:
    from equirc.pytorch_rclayers import (RegToIrrepConv, IrrepToIrrepConv, 
                                         IrrepActivationLayer, IrrepConcatLayer, 
                                         IrrepBatchNorm, ToKmerLayer)
except ImportError:
    print("Warning: 'equirc' package not found. The EquiNet model will not be available.")
    # Define placeholder classes if the import fails so the rest of the script doesn't break
    class ToKmerLayer(nn.Module): pass
    class RegToIrrepConv(nn.Module): pass
    class IrrepBatchNorm(nn.Module): pass
    class IrrepActivationLayer(nn.Module): pass
    class IrrepToIrrepConv(nn.Module): pass
    class IrrepConcatLayer(nn.Module): pass


class BaseModel(pl.LightningModule):
    """
    A base model for PyTorch Lightning that handles training, validation, and test steps.
    It's designed to be flexible for both classification and regression tasks and
    now supports models with single or multiple inputs.
    """
    def __init__(
            self,
            input_length: int,
            learning_rate: float = 1e-3,
            classify: bool = False,
            **kwargs):
        """
        Args:
            input_length (int): The number of input features or the length of the input sequence.
            learning_rate (float): The learning rate for the optimizer.
            classify (bool): If True, the model is set up for binary classification. 
                             Otherwise, it's set up for regression.
        """
        super().__init__()
        # save_hyperparameters() is essential for logging and loading models with tuned parameters
        self.save_hyperparameters()
        self.input_length = input_length
        self.learning_rate = learning_rate
        self.classify = classify
        
        # Initialize metrics based on the task (classification or regression)
        if self.classify:
            self.loss = F.binary_cross_entropy
            self.accuracy = torchmetrics.Accuracy(task="binary")
            self.auroc = torchmetrics.AUROC(task="binary")
        else:
            self.loss = F.mse_loss

    def _handle_batch(self, batch):
        """Handles different batch structures (single vs. multiple inputs)."""
        x, y = batch
        if isinstance(x, (list, tuple)):
            y_hat = self.forward(*x)
        else:
            y_hat = self.forward(x)
        loss = self.loss(y_hat, y.view_as(y_hat).float())
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        """Performs a single training step."""
        loss, _, _ = self._handle_batch(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Performs a single validation step."""
        loss, y_hat, y = self._handle_batch(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Log classification metrics if applicable
        if self.classify:
            # Squeeze the prediction tensor to match the target tensor's shape
            y_hat_squeezed = y_hat.squeeze()
            self.accuracy(y_hat_squeezed, y)
            self.auroc(y_hat_squeezed, y)
            self.log('val_acc', self.accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_auc', self.auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Performs a single test step."""
        loss, y_hat, y = self._handle_batch(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Log classification metrics if applicable
        if self.classify:
            y_hat_squeezed = y_hat.squeeze()
            self.accuracy(y_hat_squeezed, y)
            self.auroc(y_hat_squeezed, y)
            self.log('test_acc', self.accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('test_auc', self.auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """Configures the optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def create_windows(self, x, step_size=1):
        """
        Create overlapping windows from the input tensor.
        The window size is determined by self.input_length.
        """
        window_size = self.input_length
        if x.size(2) < window_size:
            return []
        windows = x.unfold(dimension=2, size=window_size, step=step_size)
        # Reshape to (num_windows, batch_size, channels, window_size)
        windows = windows.permute(2, 0, 1, 3) 
        return [w for w in windows]


class FlexibleMLP(BaseModel):
    """
    A flexible Multi-Layer Perceptron (MLP) for tabular data, adaptable to any number of input features.
    """
    def __init__(self, hidden_dims: list = [64, 32], output_dim: int = 1, dropout_p: float = 0.5, **kwargs):
        """
        Args:
            hidden_dims (list): A list of integers specifying the number of neurons in each hidden layer.
            output_dim (int): The number of output units.
            dropout_p (float): The dropout probability.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()
        
        layers = []
        input_d = self.input_length 
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_d, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            input_d = h_dim
        
        layers.append(nn.Linear(input_d, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the model."""
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.model(x)
        
        if self.classify:
            return torch.sigmoid(x)
        return x


class FlexibleCNN(BaseModel):
    """
    A simple 1D Convolutional Neural Network. This can be useful if your data has some sequential nature.
    """
    def __init__(self, input_channels=1, num_channels=[32, 64], kernel_sizes=[3, 3], dense_size=128, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        
        self.convs = nn.ModuleList()
        in_ch = input_channels
        
        for i in range(len(num_channels)):
            self.convs.append(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=num_channels[i],
                    kernel_size=kernel_sizes[i],
                    padding='same'
                )
            )
            in_ch = num_channels[i]

        self.pool = nn.AdaptiveMaxPool1d(1) # Global max pooling
        fc_input_size = num_channels[-1]
        
        self.fc1 = nn.Linear(fc_input_size, dense_size)
        self.fc2 = nn.Linear(dense_size, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        for conv in self.convs:
            x = F.relu(conv(x))
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        
        if self.classify:
            return torch.sigmoid(self.fc2(x))
        return self.fc2(x)


class VCNN(BaseModel):
    """Refactored VCNN with tunable hyperparameters."""
    def __init__(self, input_channels=4, conv_channels=[32, 16], kernel_size=3, pool_output_size=100, dense_size=128, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=conv_channels[0], kernel_size=kernel_size, padding='same')
        self.conv2 = nn.Conv1d(in_channels=conv_channels[0], out_channels=conv_channels[1], kernel_size=kernel_size, padding='same')
        self.pool = nn.AdaptiveMaxPool1d(pool_output_size) 
        self.fc1 = nn.Linear(pool_output_size * conv_channels[1], dense_size)
        self.fc2 = nn.Linear(dense_size, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        if self.classify:
            return torch.sigmoid(self.fc2(x))
        return self.fc2(x)


class VCNNBpnet(BaseModel):
    """Refactored VCNNBpnet with tunable hyperparameters."""
    def __init__(self, input_channels=4, num_channels=64, kernel_size=21, dilations=[1, 1, 2, 4, 8], pool_output_size=128, dense_sizes=[128, 64], **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.conv_layers = nn.ModuleList()
        in_ch = input_channels
        for i, dilation in enumerate(dilations):
            padding = (kernel_size - 1) // 2 * dilation
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_ch, 
                    out_channels=num_channels, 
                    kernel_size=kernel_size, 
                    padding=padding, 
                    dilation=dilation
                )
            )
            in_ch = num_channels

        self.pool = nn.AdaptiveMaxPool1d(pool_output_size)
        
        self.dense_layers = nn.ModuleList()
        fc_in_size = pool_output_size * num_channels
        for size in dense_sizes:
            self.dense_layers.append(nn.Linear(fc_in_size, size))
            fc_in_size = size
        
        self.output_layer = nn.Linear(fc_in_size, 1)

    def forward(self, x):
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        for dense in self.dense_layers:
            x = F.relu(dense(x))
        
        if self.classify:
            return torch.sigmoid(self.output_layer(x))
        return self.output_layer(x)


class MultiCNN(BaseModel):
    """Refactored MultiCNN with tunable hyperparameters."""
    def __init__(self, input_channels=4, num_numerical_features=1, conv_channels=[32, 64], kernel_size=3, pool_size=2, num_dense_dims=[8, 1], combined_dense_size=128, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        # CNN Path
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=conv_channels[0], kernel_size=kernel_size, padding='same')
        self.conv2 = nn.Conv1d(in_channels=conv_channels[0], out_channels=conv_channels[1], kernel_size=kernel_size, padding='same')
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        cnn_output_len = self.input_length // pool_size
        
        # Dense Path for numerical features
        self.dense_num_layers = nn.ModuleList()
        num_in_dim = num_numerical_features
        for dim in num_dense_dims:
            self.dense_num_layers.append(nn.Linear(num_in_dim, dim))
            num_in_dim = dim
        
        # Combined fully connected layer
        combined_input_size = conv_channels[1] * cnn_output_len + num_dense_dims[-1]
        self.fc1 = nn.Linear(combined_input_size, combined_dense_size)
        self.fc2 = nn.Linear(combined_dense_size, 1)

    def forward(self, x_seq, x_num):
        # Process sequence data
        x = F.relu(self.conv1(x_seq))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)

        # Process numerical data
        for i, layer in enumerate(self.dense_num_layers):
            x_num = layer(x_num)
            if i < len(self.dense_num_layers) - 1: # Apply ReLU to all but the last layer
                x_num = F.relu(x_num)

        # Concatenate and process
        x_combined = torch.cat((x, x_num), dim=1)
        x_combined = F.relu(self.fc1(x_combined))
        
        if self.classify:
            return torch.sigmoid(self.fc2(x_combined))
        return self.fc2(x_combined)


class CNN(BaseModel):
    """This model was already well-parameterized."""
    def __init__(self,
                 input_channels: int = 4,
                 num_channels: list = [32, 64],
                 paddings: list = [1, 1],
                 dilations: list = [1, 1],
                 kernel_sizes: list = [3, 3],
                 max_pool: bool = True,
                 max_pool_size: int = 2,
                 max_pool_stride: int = 2,
                 dense_size: int = 128,
                 dropout_conv: float = 0.25,
                 dropout_fc: float = 0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        assert len(num_channels) == len(paddings) == len(dilations) == len(kernel_sizes), \
            "All layer parameter lists must have the same length"

        self.convs = nn.ModuleList()
        in_ch = input_channels
        seq_len = self.input_length

        for i in range(len(num_channels)):
            conv = nn.Conv1d(
                in_channels=in_ch,
                out_channels=num_channels[i],
                kernel_size=kernel_sizes[i],
                padding=paddings[i],
                dilation=dilations[i]
            )
            self.convs.append(nn.Sequential(
                conv,
                nn.ReLU(),
                nn.Dropout(dropout_conv)
            ))
            seq_len = math.floor(
                (seq_len + 2 * paddings[i] - dilations[i] * (kernel_sizes[i] - 1) - 1) / 1 + 1
            )
            in_ch = num_channels[i]

        if max_pool:
            self.pool = nn.MaxPool1d(kernel_size=max_pool_size, stride=max_pool_stride)
            pooled_length = math.floor((seq_len - (max_pool_size - 1) - 1) / max_pool_stride + 1)
        else:
            self.pool = nn.Identity()
            pooled_length = seq_len
        
        self.fc1 = nn.Linear(num_channels[-1] * pooled_length, dense_size)
        self.dropout_fc = nn.Dropout(dropout_fc)
        self.fc2 = nn.Linear(dense_size, 1)

    def forward(self, x):
        for conv_block in self.convs:
            x = conv_block(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        
        if self.classify:
            return torch.sigmoid(self.fc2(x))
        return self.fc2(x)


class SimpleCNN(BaseModel):
    """Refactored SimpleCNN with tunable hyperparameters."""
    def __init__(self, input_channels=4, conv_channels=[32, 64], kernel_size=3, pool_size=2, dense_size=128, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=conv_channels[0], kernel_size=kernel_size, padding='same')
        self.conv2 = nn.Conv1d(in_channels=conv_channels[0], out_channels=conv_channels[1], kernel_size=kernel_size, padding='same')
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        
        pooled_length = self.input_length // pool_size
        self.fc1 = nn.Linear(conv_channels[1] * pooled_length, dense_size)
        self.fc2 = nn.Linear(dense_size, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        if self.classify:
            return torch.sigmoid(self.fc2(x))
        return self.fc2(x)
    
    def forward_with_windows(self, x):
        """Processes a long sequence by sliding a window over it."""
        windows = self.create_windows(x, step_size=1)
        if not windows:
            return torch.tensor([], device=x.device)
            
        window_batch = torch.cat(windows, dim=0)
        outputs = self.forward(window_batch)
        
        num_windows = len(windows)
        batch_size = x.size(0)
        outputs = outputs.view(num_windows, batch_size, -1)
        return torch.max(outputs, dim=0)[0]


class EquiNet(BaseModel):
    """
    Reverse-Complement Equivariant Network for DNA Sequences.
    This model was already well-parameterized.
    """
    def __init__(self,
                 filters=((8, 8), (8, 8), (8, 8)),
                 kernel_sizes=(3, 3, 3),
                 pool_size=2,
                 pool_length=2,
                 out_size=1,
                 placeholder_bn=False,
                 kmers=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.kmers = int(kmers)
        self.to_kmer = ToKmerLayer(k=self.kmers)

        reg_in = self.to_kmer.features // 2
        first_kernel_size = kernel_sizes[0]
        first_a, first_b = filters[0]
        self.last_a, self.last_b = filters[-1]
        
        self.reg_irrep = RegToIrrepConv(reg_in=reg_in, a_out=first_a, b_out=first_b, kernel_size=first_kernel_size)
        self.first_bn = IrrepBatchNorm(a=first_a, b=first_b, placeholder=placeholder_bn)
        self.first_act = IrrepActivationLayer(a=first_a, b=first_b)

        self.irrep_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()
        
        seq_len = self.input_length - (first_kernel_size - 1)
        for i in range(1, len(filters)):
            prev_a, prev_b = filters[i - 1]
            next_a, next_b = filters[i]
            self.irrep_layers.append(IrrepToIrrepConv(
                a_in=prev_a, b_in=prev_b, a_out=next_a, b_out=next_b, kernel_size=kernel_sizes[i],
            ))
            self.bn_layers.append(IrrepBatchNorm(a=next_a, b=next_b, placeholder=placeholder_bn))
            self.activation_layers.append(IrrepActivationLayer(a=next_a, b=next_b))
            seq_len -= (kernel_sizes[i] - 1)

        self.concat = IrrepConcatLayer(a=self.last_a, b=self.last_b)
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_length)
        self.flattener = nn.Flatten()
        
        pooled_length = math.floor((seq_len - (pool_size - 1) - 1) / pool_length + 1)
        input_size = pooled_length * (self.last_a + self.last_b)
        
        self.dense = nn.Linear(input_size, out_size)
        self.final_activation = nn.Sigmoid()

    def forward(self, x, windows=False):
        if windows:
            return self.forward_with_windows(x)
        
        x = self.to_kmer(x)
        x = self.reg_irrep(x)
        x = self.first_bn(x)
        x = self.first_act(x)

        for irrep_layer, bn_layer, act_layer in zip(self.irrep_layers, self.bn_layers, self.activation_layers):
            x = irrep_layer(x)
            x = bn_layer(x)
            x = act_layer(x)

        x = self.concat(x)
        x = self.pool(x)
        x = self.flattener(x)
        x = self.dense(x)
        if self.classify:
            x = self.final_activation(x)
        return x

    def forward_with_windows(self, x):
        windows = self.create_windows(x, step_size=1)
        if not windows:
            return torch.tensor([], device=x.device)
        
        window_batch = torch.cat(windows, dim=0)
        outputs = self.forward(window_batch, windows=False)
        
        num_windows = len(windows)
        batch_size = x.size(0)
        outputs = outputs.view(num_windows, batch_size, -1)
        return torch.max(outputs, dim=0)[0]


class MLP(BaseModel):
    """Refactored MLP with tunable hyperparameters."""
    def __init__(self, input_channels=4, hidden_dims=[8, 8], output_dim=1, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.layers = nn.ModuleList()
        in_dim = self.input_length * input_channels
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        self.layers.append(nn.Linear(in_dim, output_dim))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        
        if self.classify and self.layers[-1] is not torch.sigmoid:
             return torch.sigmoid(x)
        return x


class BPNet(BaseModel):
    """Refactored BPNet with tunable hyperparameters."""
    def __init__(self, input_channels=4, num_channels=64, kernel_size=21, dilations=[1, 1, 2, 4, 8], pool_size=2, dense_sizes=[128, 64], **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.conv_layers = nn.ModuleList()
        in_ch = input_channels
        for dilation in dilations:
            # Calculate padding to keep sequence length the same with dilation
            padding = (kernel_size - 1) // 2 * dilation
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_ch, 
                    out_channels=num_channels, 
                    kernel_size=kernel_size, 
                    padding=padding, 
                    dilation=dilation
                )
            )
            in_ch = num_channels
        
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        pooled_length = self.input_length // pool_size

        self.dense_layers = nn.ModuleList()
        fc_in_size = num_channels * pooled_length
        for size in dense_sizes:
            self.dense_layers.append(nn.Linear(fc_in_size, size))
            fc_in_size = size
        
        self.output_layer = nn.Linear(fc_in_size, 1)

    def forward(self, x):
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        for dense in self.dense_layers:
            x = F.relu(dense(x))
        
        if self.classify:
            return torch.sigmoid(self.output_layer(x))
        return self.output_layer(x)


class BPNetReal(BaseModel):
    """
    This model was already well-parameterized.
    A simplified implementation of BPNet for compatibility with the BaseModel training loop.
    This version only predicts the total counts and does not output a profile.
    A full implementation would require a custom training step to handle multi-task loss.
    """
    def __init__(self, n_filters=64, n_layers=8, 
                 n_control_tracks=0, count_output_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.n_control_tracks = n_control_tracks

        self.iconv = nn.Conv1d(4, n_filters, kernel_size=21, padding=10)
        self.irelu = nn.ReLU()

        self.rconvs = nn.ModuleList()
        for i in range(self.n_layers):
            conv = nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=2**(i+1), dilation=2**(i+1))
            self.rconvs.append(conv)
        
        self.final_relu = nn.ReLU()
        
        n_count_control = 1 if n_control_tracks > 0 else 0
        self.linear = nn.Linear(n_filters + n_count_control, 1, bias=count_output_bias)

    def forward(self, x, control=None):
        x = self.irelu(self.iconv(x))
        
        for i in range(self.n_layers):
            x_conv = self.rconvs[i](x)
            x_relu = self.final_relu(x_conv)
            x = x + x_relu # Residual connection

        x_pooled = x.mean(dim=2)
        
        if self.n_control_tracks > 0 and control is not None:
            control_sum = torch.log1p(control.sum(dim=(1,2))).unsqueeze(-1)
            x_pooled = torch.cat([x_pooled, control_sum], dim=1)

        counts = self.linear(x_pooled)
        return counts


class RNN(BaseModel):
    """Refactored RNN with tunable hyperparameters."""
    def __init__(self, input_channels=4, conv_out_channels=100, kernel_size=5, pool_size=2, dropout_conv=0.3, gru_hidden_size=100, gru_num_layers=1, bidirectional=True, dense_size=100, dropout_fc=0.6, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=conv_out_channels, kernel_size=kernel_size, padding="valid")
        self.maxpool1 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.dropout1 = nn.Dropout(dropout_conv)
        
        self.gru1 = nn.GRU(conv_out_channels, gru_hidden_size, num_layers=gru_num_layers, batch_first=True, bidirectional=bidirectional)
        
        conv_output_length = (self.input_length - kernel_size + 1) // pool_size
        num_directions = 2 if bidirectional else 1
        
        self.fc1 = nn.Linear(conv_output_length * gru_hidden_size * num_directions, dense_size)
        self.dropout2 = nn.Dropout(dropout_fc)
        self.fc2 = nn.Linear(dense_size, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = x.permute(0, 2, 1) # (Batch, Seq, Features) for GRU
        x, _ = self.gru1(x)
        x = x.contiguous().view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        if self.classify:
            return torch.sigmoid(self.fc2(x))
        return self.fc2(x)


class my_cnn(BaseModel):
    """Refactored my_cnn to be dynamically built and tunable."""
    def __init__(self, input_channels=4, conv_channels=[16, 32], kernel_size=3, dense_size=128, **kwargs):
        super(my_cnn, self).__init__(**kwargs)
        self.save_hyperparameters()

        self.layers = nn.ModuleList()
        in_ch = input_channels
        seq_len = self.input_length

        # Convolutional layers
        for out_ch in conv_channels:
            self.layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size))
            self.layers.append(nn.ReLU())
            in_ch = out_ch
            seq_len = seq_len - (kernel_size - 1) # Update sequence length after conv 'valid' padding
        
        self.layers.append(nn.Flatten())
        
        # Dense layers
        flattened_size = seq_len * conv_channels[-1]
        self.layers.append(nn.Linear(flattened_size, dense_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(dense_size, 1))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        output = self.model(x)
        if self.classify:
            return torch.sigmoid(output)
        return output
