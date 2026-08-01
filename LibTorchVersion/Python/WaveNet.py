import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveNetLayer(nn.Module):
    def __init__(self, hidden_units, kernel_size, dilation):
        super(WaveNetLayer, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(hidden_units, hidden_units * 2, kernel_size, dilation=dilation, padding=0)
        self.res_conv = nn.Conv1d(hidden_units, hidden_units, 1)
        self.skip_conv = nn.Conv1d(hidden_units, hidden_units, 1)

    def forward(self, x):
        # Padding を左側だけに寄せる (Causal)
        # F.pad(x, (left, right)) なので、(self.padding, 0) とします
        x_padded = F.pad(x, (self.padding, 0)) 
        
        combined = self.conv(x_padded)
        # すでに causal になっているので :-self.padding のスライスは不要

        d_tanh, d_sigmoid = torch.chunk(combined, 2, dim=1)
        gated = torch.tanh(d_tanh) * torch.sigmoid(d_sigmoid)        
        res_out = self.res_conv(gated) + x
        skip_out = self.skip_conv(gated)
        
        return res_out, skip_out

class WaveNetAmplifire(nn.Module):
    def __init__(self, input_size, res_hidden_units, kernel_size, num_layers, param_dim=7):
        super(WaveNetAmplifire, self).__init__()

        self.input_conv = nn.Conv1d(1 + param_dim, res_hidden_units, 1)
        
        self.layers = nn.ModuleList([
            WaveNetLayer(res_hidden_units, kernel_size, dilation=2**i)
            for i in range(num_layers)
        ])
        
        self.out_conv1 = nn.Conv1d(res_hidden_units, res_hidden_units, 1)
        self.out_conv2 = nn.Conv1d(res_hidden_units, 1, 1)

    def forward(self, x, p):

        batch_size, seq_len, _ = x.size()
        p_expanded = p.unsqueeze(1).expand(-1, seq_len, -1)
        x_with_p = torch.cat([x, p_expanded], dim=-1)
        x_with_p = x_with_p.transpose(1, 2)
        h = self.input_conv(x_with_p)
        skip_connections = []
        for layer in self.layers:
            h, skip = layer(h)
            skip_connections.append(skip)

        out = torch.stack(skip_connections).sum(dim=0)
        out = F.leaky_relu(out)
        out = self.out_conv1(out)
        out = F.leaky_relu(out)
        out = self.out_conv2(out)        
        out = out.transpose(1, 2)
        
        return torch.tanh(out)