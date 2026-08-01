import torch
import torch.nn as nn

class LSTMAmplifier(nn.Module):
    def __init__(self, input_size, hidden_units=64, param_dim=7, num_layers=2):
        super().__init__()
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=1 + param_dim, 
            hidden_size=hidden_units, 
            num_layers=num_layers, 
            batch_first=True
        )

        self.init_h = nn.Linear(param_dim, hidden_units * num_layers)
        self.init_c = nn.Linear(param_dim, hidden_units * num_layers)

        self.out_fc = nn.Sequential(
            nn.Linear(hidden_units, 1)
        )

    def forward(self, x, p):
        batch_size, seq_len, _ = x.size()

        p_expanded = p.unsqueeze(1).expand(-1, seq_len, -1)
        x_with_p = torch.cat([x, p_expanded], dim=-1)

        h0 = self.init_h(p).view(self.num_layers, batch_size, self.hidden_units).contiguous()
        c0 = self.init_c(p).view(self.num_layers, batch_size, self.hidden_units).contiguous()

        lstm_out, _ = self.lstm(x_with_p, (h0, c0))
        
        out = self.out_fc(lstm_out)
        out = torch.tanh(out)
        return out  