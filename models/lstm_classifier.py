import torch
import torch.nn as nn
class LSTM(nn.Module):
    def __init__(self, input_len, hidden_size, num_classes, n_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size 
        self.n_layers = n_layers       
        self.lstm = nn.LSTM(input_len, hidden_size, n_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, num_classes)  

    def forward(self, X):
        hidden_states = torch.zeros(self.n_layers, X.size(0), self.hidden_size, device=X.device)
        cell_states = torch.zeros(self.n_layers, X.size(0), self.hidden_size, device=X.device)

        output, hide = self.lstm(X, (hidden_states, cell_states))
        output = self.output_layer(output[:,-1,:])
        return output