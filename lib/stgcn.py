import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.3):
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.dropout = nn.Dropout(dropout)
        self.swish = Swish()  
        self.elu = nn.ELU()

    def forward(self, X):

        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.leaky_relu(temp + self.conv3(X))
        out = self.dropout(out)
        out = out.permute(0, 2, 3, 1)
        return out

class STGCNBlock(nn.Module):

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes, dropout=0.3):

        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels, out_channels=out_channels, dropout=dropout)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels, spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels, out_channels=out_channels, dropout=dropout)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        t2 = F.leaky_relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        t3 = self.dropout(t3)
        return self.batch_norm(t3)


class LSTM(nn.Module):
    
    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, lstm_hidden_dim=64, num_lstm_layers=5):

        super(LSTM, self).__init__()
        self.input_size = 1
        self.hidden_size = lstm_hidden_dim
        self.output_size = num_timesteps_output
        self.num_layers = num_lstm_layers
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=lstm_hidden_dim, 
                            num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, A_hat, X_joint, X_motion):

        batch_size, num_nodes, num_timesteps, feature_dim = X_joint.shape
        lstm_out = []

        for node_idx in range(num_nodes):
            node_input = X_joint[:, node_idx, :, :]  
            lstm_out_node, _ = self.lstm(node_input)  
            lstm_out_node = lstm_out_node[:, -1, :].unsqueeze(1)  
            linear_out = self.linear(lstm_out_node)  
            lstm_out.append(linear_out)
        lstm_out = torch.cat(lstm_out, dim=1)  
        final_out = lstm_out
        return final_out


class TwoStreamSTGCN(nn.Module):

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, dropout=0.3, lstm_hidden_dim=64, num_lstm_layers=5):
        super(TwoStreamSTGCN, self).__init__()
        self.block1_joint = STGCNBlock(in_channels=num_features, out_channels=512,
                                       spatial_channels=256, num_nodes=num_nodes, dropout=dropout)
        self.block2_joint = STGCNBlock(in_channels=512, out_channels=512,
                                       spatial_channels=256, num_nodes=num_nodes, dropout=dropout)
        self.block1_motion = STGCNBlock(in_channels=num_features, out_channels=512,
                                        spatial_channels=256, num_nodes=num_nodes, dropout=dropout)
        self.block2_motion = STGCNBlock(in_channels=512, out_channels=512,
                                        spatial_channels=256, num_nodes=num_nodes, dropout=dropout)
        self.last_temporal_joint = TimeBlock(in_channels=512, out_channels=512, dropout=dropout)
        self.last_temporal_motion = TimeBlock(in_channels=512, out_channels=512, dropout=dropout)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 512 * 2, num_timesteps_output)

        self.input_size = 1
        self.hidden_size = lstm_hidden_dim
        self.output_size = num_timesteps_output
        self.num_layers = num_lstm_layers
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=lstm_hidden_dim, 
                            num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, A_hat, X_joint, X_motion):

        out1_joint = self.block1_joint(X_joint, A_hat)
        out2_joint = self.block2_joint(out1_joint, A_hat)
        out3_joint = self.last_temporal_joint(out2_joint)

        out1_motion = self.block1_motion(X_motion, A_hat)
        out2_motion = self.block2_motion(out1_motion, A_hat)
        out3_motion = self.last_temporal_motion(out2_motion)

        out3 = torch.cat((out3_joint, out3_motion), dim=-1)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))

        batch_size, num_nodes, num_timesteps, feature_dim = X_joint.shape
        lstm_out = []

        for node_idx in range(num_nodes):
            node_input = X_joint[:, node_idx, :, :]  
            lstm_out_node, _ = self.lstm(node_input)  
            lstm_out_node = lstm_out_node[:, -1, :].unsqueeze(1) 
            linear_out = self.linear(lstm_out_node) 
            lstm_out.append(linear_out)

        lstm_out = torch.cat(lstm_out, dim=1)  
        final_out = (out4 + lstm_out) / 2  

        return final_out

