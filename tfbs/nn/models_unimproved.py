import math
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from equirc.pytorch_rclayers import RegToRegConv, RegToIrrepConv, IrrepToIrrepConv, IrrepActivationLayer, \
    IrrepConcatLayer, IrrepBatchNorm, RegBatchNorm, RegConcatLayer, ToKmerLayer


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

class VCNN(BaseModel):
    def __init__(self, input_channels=4, **kwargs):
        super(VCNN, self).__init__(**kwargs)
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(100)
        self.fc1 = nn.Linear(100*16,128)  # Adjusted for sequence length
        self.fc2 = nn.Linear(128, 1)  # Output a single float value

    def forward(self, x):
        x = self.remove_trailing_zeros(x)
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        if self.classify:
            x = torch.sigmoid(self.fc2(x))
        else:
            x = self.fc2(x)
        return x
    
    def remove_trailing_zeros(self, tensor):
        """
        Remove trailing zero vectors from a tensor of shape [B, 4, L].
        Trailing entries with all zeros in the 4 channels are removed.

        Args:
            tensor (torch.Tensor): Input tensor of shape [B, 4, L].

        Returns:
            torch.Tensor: Tensor with trailing zero vectors removed.
        """
        # Compute a mask to identify non-zero positions along the last dimension
        mask = (tensor != 0).any(dim=1)  # Shape: [B, L]

        # Find the maximum index of non-zero entries for each batch
        max_nonzero_idx = mask.sum(dim=1)  # Shape: [B]

        # Slice the tensor to remove trailing zeros
        trimmed_tensors = [tensor[b, :, :max_nonzero_idx[b]] for b in range(tensor.size(0))]

        # Pad the trimmed tensors back to a batch (if needed)
        trimmed_tensors=torch.stack(trimmed_tensors, dim=0)
        return trimmed_tensors

class VCNN_BPNET(BaseModel):
    def __init__(self, **kwargs):
        super(VCNN_BPNET, self).__init__(input_channels=4, **kwargs)
        self.conv1 = nn.Conv1d(in_channels=self.input_channels, out_channels=64, kernel_size=21, padding=10)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=21, padding=10)
        self.dilated_conv1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=21, padding=20, dilation=2)
        self.dilated_conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=21, padding=40, dilation=4)
        self.dilated_conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=21, padding=80, dilation=8)
        self.pool = nn.AdaptiveMaxPool1d(128)
        self.fc1 = nn.Linear(264*64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.remove_trailing_zeros(x)
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
    
    def remove_trailing_zeros(self, tensor):
        """
        Remove trailing zero vectors from a tensor of shape [B, 4, L].
        Trailing entries with all zeros in the 4 channels are removed.

        Args:
            tensor (torch.Tensor): Input tensor of shape [B, 4, L].

        Returns:
            torch.Tensor: Tensor with trailing zero vectors removed.
        """
        # Compute a mask to identify non-zero positions along the last dimension
        mask = (tensor != 0).any(dim=1)  # Shape: [B, L]

        # Find the maximum index of non-zero entries for each batch
        max_nonzero_idx = mask.sum(dim=1)  # Shape: [B]

        # Slice the tensor to remove trailing zeros
        trimmed_tensors = [tensor[b, :, :max_nonzero_idx[b]] for b in range(tensor.size(0))]

        # Pad the trimmed tensors back to a batch (if needed)
        trimmed_tensors=torch.stack(trimmed_tensors, dim=0)
        return trimmed_tensors


class MultiCNN(BaseModel):
    def __init__(self, input_channels=4, num_num=1,**kwargs):
        super(MultiCNN, self).__init__(**kwargs)
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * (self.input_length // 2) + 1, 128)  # Adjusted for sequence length
        self.fc2 = nn.Linear(128, 1)  # Output a single float value

        # additional number
        self.dense_num = nn.Linear(3, 8)
        self.dense_num2 = nn.Linear(8, 1)

    def forward(self, x, x_num):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor

        # Process the additional number
        x_num = F.relu(self.dense_num(x_num))
        x_num = F.relu(self.dense_num2(x_num))

        # Concatenate the CNN output with the processed number
        x = torch.cat((x, x_num), dim=1)
        x = F.relu(self.fc1(x))
        
        if self.classify:
            x = torch.sigmoid(self.fc2(x))
        else:
            x = self.fc2(x)
        return x


class CNN(BaseModel):
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
        super(CNN, self).__init__(**kwargs)
        assert len(num_channels) == len(paddings) == len(dilations) == len(kernel_sizes), \
            "All layer parameter lists must have the same length"

        self.convs = nn.ModuleList()
        self.conv_dropouts = nn.ModuleList()
        in_ch = input_channels
        seq_len = self.input_length

        # Dynamically create convolutional layers and dropout after each conv
        for i in range(len(num_channels)):
            self.convs.append(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=num_channels[i],
                    kernel_size=kernel_sizes[i],
                    padding=paddings[i],
                    dilation=dilations[i]
                )
            )
            self.conv_dropouts.append(nn.Dropout(dropout_conv))
            seq_len = math.floor(
                (seq_len + 2 * paddings[i] - dilations[i] * (kernel_sizes[i] - 1) - 1) + 1
            )
            in_ch = num_channels[i]

        self.max_pool = max_pool
        if self.max_pool:
            self.pool = nn.MaxPool1d(kernel_size=max_pool_size, stride=max_pool_stride)
            pooled_length = seq_len // max_pool_stride
        else:
            self.pool = nn.Identity()
            pooled_length = seq_len
        self.fc1 = nn.Linear(num_channels[-1] * pooled_length, dense_size)
        self.dropout_fc = nn.Dropout(dropout_fc)
        self.fc2 = nn.Linear(dense_size, 1)


    def forward(self, x):
        for conv in self.convs:
            x = F.relu(conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        # Apply sigmoid activation if classifying
        # Otherwise, return the raw output
        if self.classify:
            x = torch.sigmoid(self.fc2(x))
        else:
            x = self.fc2(x)
        return x

class SimpleCNN(BaseModel):
    def __init__(self, input_channels=4, **kwargs):
        super(SimpleCNN, self).__init__(**kwargs)
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * (self.input_length // 2), 128)  # Adjusted for sequence length
        self.fc2 = nn.Linear(128, 1)  # Output a single float value

    def forward(self, x, mask=None):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        if self.classify:
            x = torch.sigmoid(self.fc2(x))
        else:
            x = self.fc2(x)
        return x
    
    def forward_with_windows(self, x):
        # Split the input into overlapping windows
        windows = self.create_windows(x,step_size=1)
        outputs = []

        for window in windows:
            out = F.relu(self.conv1(window))
            out = self.pool(F.relu(self.conv2(out)))
            out = out.view(out.size(0), -1)  # Flatten the tensor
            out = F.relu(self.fc1(out))
            if self.classify:
                out = torch.sigmoid(self.fc2(out))
            else:
                out = self.fc2(out)
            outputs.append(out)

       # Stack the outputs from all windows
        outputs = torch.stack(outputs, dim=1)

        # Return the maximum value across the windows
        return torch.max(outputs, dim=1)[0]
    

class EquiNet(BaseModel):
    """
    Reverse-Complement Equivariant Network for DNA Sequences.
    Requires the 'equirc' package.
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
    
class EquiNet_broken(BaseModel):
    def __init__(self,
                 filters=((8, 8), (8, 8), (8, 8)),
                 kernel_sizes=(3, 3, 3),
                 pool_size=2,
                 pool_length=2,
                 out_size=1,
                 placeholder_bn=False,
                 kmers=1,
                 **kwargs):
        """
        This network takes as inputs windows of 1000 base pairs one hot encoded and outputs a binary or linear prediction

        First maps the regular representation to irrep setting
        Then goes from one setting to another.

        Taken from 
        @article{mallet2021reverse,
        title={Reverse-Complement Equivariant Networks for DNA Sequences},
        author={Mallet, Vincent and Vert, Jean-Philippe},
        journal={Advances in Neural Information Processing Systems},
        volume={34},
        year={2021}
        }
        """
        super(EquiNet, self).__init__(**kwargs)

        self.kmers = int(kmers)
        self.to_kmer = ToKmerLayer(k=self.kmers)

        # First mapping goes from the input to an irrep feature space
        reg_in = self.to_kmer.features // 2
        first_kernel_size = kernel_sizes[0]
        first_a, first_b = filters[0]
        self.last_a, self.last_b = filters[-1]
        self.reg_irrep = RegToIrrepConv(reg_in=reg_in,
                                        a_out=first_a,
                                        b_out=first_b,
                                        kernel_size=first_kernel_size)
        self.first_bn = IrrepBatchNorm(a=first_a, b=first_b, placeholder=placeholder_bn)
        self.first_act = IrrepActivationLayer(a=first_a, b=first_b)

        # Now add the intermediate layers : sequence of conv, BN, activation
        self.irrep_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()
        for i in range(1, len(filters)):
            prev_a, prev_b = filters[i - 1]
            next_a, next_b = filters[i]
            self.irrep_layers.append(IrrepToIrrepConv(
                a_in=prev_a,
                b_in=prev_b,
                a_out=next_a,
                b_out=next_b,
                kernel_size=kernel_sizes[i],
            ))
            self.bn_layers.append(IrrepBatchNorm(a=next_a, b=next_b, placeholder=placeholder_bn))
            self.activation_layers.append(IrrepActivationLayer(a=next_a,
                                                               b=next_b))

        self.concat = IrrepConcatLayer(a=self.last_a, b=self.last_b)
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_length)
        self.flattener = nn.Flatten()

        # Calculate the size of the input to the dense layer
        # Start with the input length and apply the effects of pooling
        pooled_length = (self.input_length - (kernel_sizes[0] - 1)) // pool_length
        input_size = pooled_length * (self.last_a + self.last_b)

        print(self.last_a, self.last_b)


        # old was 64
        # 69=496
        # 70=512
        # FOR CONTEXT IT IS 496
        # NEED TO FIND FORMULA
        num_dense = 16 * self.input_length - 608
        self.dense = nn.Linear(num_dense,out_size)
        self.final_activation = nn.Sigmoid()

    def forward(self, x, windows=False):
        if windows:
            return self.forward_with_windows(x)
        else:
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
        # Split the input into overlapping windows
        windows = self.create_windows(x,step_size=1)
        outputs = []

        for window in windows:
            out = self.forward(window, windows=False)
            outputs.append(out)

       # Stack the outputs from all windows
        outputs = torch.stack(outputs, dim=1)

        # Return the maximum value across the windows
        return torch.max(outputs, dim=1)[0]

class MLP_csv(BaseModel):
    def __init__(self, hidden_dim = 8, output_dim = 1, **kwargs):
        super(MLP_csv, self).__init__(**kwargs)
        self.fc1 = nn.Linear(self.input_length, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Ensure input is 2D (batch_size, input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.classify:
            x = torch.sigmoid(self.fc3(x))
        else:
            x = self.fc3(x)
        return x

class MLP(BaseModel):
    def __init__(self, channel_num = 4, hidden_dim = 8, output_dim = 1, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.fc1 = nn.Linear(self.input_length*channel_num, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Ensure input is 2D (batch_size, input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.classify:
            x = torch.sigmoid(self.fc3(x))
        else:
            x = self.fc3(x)
        return x

class BPNet(BaseModel):
    def __init__(self, **kwargs):
        super(BPNet, self).__init__(**kwargs)
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=21, padding=10)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=21, padding=10)
        self.dilated_conv1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=21, padding=20, dilation=2)
        self.dilated_conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=21, padding=40, dilation=4)
        self.dilated_conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=21, padding=80, dilation=8)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * (self.input_length // 2), 128)
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



class BPNet_real(BaseModel):
	"""A basic BPNet model with stranded profile and total count prediction.

	This is a reference implementation for BPNet models. It exactly matches the
	architecture in the official ChromBPNet repository. It is very similar to
	the implementation in the official basepairmodels repository but differs in
	when the activation function is applied for the resifual layers. See the
	BasePairNet object below for an implementation that matches that repository. 

	The model takes in one-hot encoded sequence, runs it through: 

	(1) a single wide convolution operation 

	THEN 

	(2) a user-defined number of dilated residual convolutions

	THEN

	(3a) profile predictions done using a very wide convolution layer 
	that also takes in stranded control tracks 

	AND

	(3b) total count prediction done using an average pooling on the output
	from 2 followed by concatenation with the log1p of the sum of the
	stranded control tracks and then run through a dense layer.

	This implementation differs from the original BPNet implementation in
	two ways:

	(1) The model concatenates stranded control tracks for profile
	prediction as opposed to adding the two strands together and also then
	smoothing that track 

	(2) The control input for the count prediction task is the log1p of
	the strand-wise sum of the control tracks, as opposed to the raw
	counts themselves.

	(3) A single log softmax is applied across both strands such that
	the logsumexp of both strands together is 0. Put another way, the
	two strands are concatenated together, a log softmax is applied,
	and the MNLL loss is calculated on the concatenation. 

	(4) The count prediction task is predicting the total counts across
	both strands. The counts are then distributed across strands according
	to the single log softmax from 3.


	Parameters
	----------
	n_filters: int, optional
		The number of filters to use per convolution. Default is 64.

	n_layers: int, optional
		The number of dilated residual layers to include in the model.
		Default is 8.

	n_outputs: int, optional
		The number of profile outputs from the model. Generally either 1 or 2 
		depending on if the data is unstranded or stranded. Default is 2.

	n_control_tracks: int, optional
		The number of control tracks to feed into the model. When predicting
		TFs, this is usually 2. When predicting accessibility, this is usualy
		0. When 0, this input is removed from the model. Default is 2.

	alpha: float, optional
		The weight to put on the count loss.

	profile_output_bias: bool, optional
		Whether to include a bias term in the final profile convolution.
		Removing this term can help with attribution stability and will usually
		not affect performance. Default is True.

	count_output_bias: bool, optional
		Whether to include a bias term in the linear layer used to predict
		counts. Removing this term can help with attribution stability but
		may affect performance. Default is True.

	name: str or None, optional
		The name to save the model to during training.

	trimming: int or None, optional
		The amount to trim from both sides of the input window to get the
		output window. This value is removed from both sides, so the total
		number of positions removed is 2*trimming.

	verbose: bool, optional
		Whether to display statistics during training. Setting this to False
		will still save the file at the end, but does not print anything to
		screen during training. Default is True.
	"""

	def __init__(self, n_filters=64, n_layers=8, n_outputs=2, 
		n_control_tracks=2, alpha=1, profile_output_bias=True, 
		count_output_bias=True, name=None, trimming=None, verbose=True):
		super(BPNet, self).__init__()
		self.n_filters = n_filters
		self.n_layers = n_layers
		self.n_outputs = n_outputs
		self.n_control_tracks = n_control_tracks

		self.alpha = alpha
		self.name = name or "bpnet.{}.{}".format(n_filters, n_layers)
		self.trimming = trimming or 2 ** n_layers

		self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=21, padding=10)
		self.irelu = torch.nn.ReLU()

		self.rconvs = torch.nn.ModuleList([
			torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=2**i, 
				dilation=2**i) for i in range(1, self.n_layers+1)
		])
		self.rrelus = torch.nn.ModuleList([
			torch.nn.ReLU() for i in range(1, self.n_layers+1)
		])

		self.fconv = torch.nn.Conv1d(n_filters+n_control_tracks, n_outputs, 
			kernel_size=75, padding=37, bias=profile_output_bias)
		
		n_count_control = 1 if n_control_tracks > 0 else 0
		self.linear = torch.nn.Linear(n_filters+n_count_control, 1, 
			bias=count_output_bias)

		#self.logger = Logger(["Epoch", "Iteration", "Training Time",
	#		"Validation Time", "Training MNLL", "Training Count MSE", 
	#		"Validation MNLL", "Validation Profile Pearson", 
	#		"Validation Count Pearson", "Validation Count MSE", "Saved?"], 
	#		verbose=verbose)

class RNN(BaseModel):
    def __init__(self, **kwargs):
        super(RNN, self).__init__(**kwargs)
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=100, kernel_size=5, padding="valid")
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.3)
        #check gru dropout
        self.gru1 = nn.GRU(100, 100, 1, batch_first=True, bidirectional=True)
        conv_output_length = (self.input_length - 5 + 1) // 2
        self.fc1 = nn.Linear(conv_output_length * 200, 100)  # Adjusted for GRU output dimension
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
    
class my_cnn(BaseModel):
    def __init__(self, **kwargs):
        super(my_cnn, self).__init__(**kwargs)
        self.layers = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((self.input_length - 2 * 2) * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.layers(x)











######KEEEP FOR NOW AS BACKUP
class my_cnn_old(nn.Module):
    def __init__(self,input_length=14):
        super(my_cnn_old, self).__init__()
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
    def __init__(self, input_length=14, learning_rate=1e-3):
        self.classify=False
        self.learning_rate=learning_rate
        super(CNNModule, self).__init__()
        self.save_hyperparameters()
        self.loss = F.mse_loss
        self.model = my_cnn_old(input_length=input_length)

    def forward(self, x):
        return self.model(x)

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