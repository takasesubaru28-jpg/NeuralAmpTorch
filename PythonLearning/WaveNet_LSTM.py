import torch
import torch.nn as nn
import torch.nn.functional as F
from WaveNet import *

class WaveNetLSTMAmplifier(nn.Module):
    def __init__(self, input_size, res_hidden_units, kernel_size, num_wn_layers, lstm_hidden_units, num_lstm_layers, param_dim=7):
        super(WaveNetLSTMAmplifier, self).__init__()

        self.input_conv = nn.Conv1d(1 + param_dim, res_hidden_units, 1)
        self.wn_layers = nn.ModuleList([
            WaveNetLayer(res_hidden_units, kernel_size, dilation=2**i)
            for i in range(num_wn_layers)
        ])
        
        self.lstm = nn.LSTM(
            input_size=res_hidden_units, 
            hidden_size=lstm_hidden_units, 
            num_layers=num_lstm_layers, 
            batch_first=True
        )
        
        self.out_linear = nn.Linear(lstm_hidden_units, 1)

    def forward(self, x, p):

        batch_size, seq_len, _ = x.size()
        p_expanded = p.unsqueeze(1).expand(-1, seq_len, -1)
        x_with_p = torch.cat([x, p_expanded], dim=-1)
        
        h = x_with_p.transpose(1, 2)
        h = self.input_conv(h)
        
        skip_connections = []
        for layer in self.wn_layers:
            h, skip = layer(h)
            skip_connections.append(skip)

        wn_out = torch.stack(skip_connections).sum(dim=0)
        wn_out = F.leaky_relu(wn_out)
        
        lstm_input = wn_out.transpose(1, 2)
        
        lstm_out, _ = self.lstm(lstm_input)
        
        out = self.out_linear(lstm_out)
        
        return torch.tanh(out)